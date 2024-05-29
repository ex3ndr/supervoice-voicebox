import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from torch.cuda.amp import autocast
from .debug import debug_if_invalid
from .tensors import RMSNorm

class Transformer(nn.Module):
    def __init__(self, 
        n_heads,
        n_layers,
        n_dim,
        n_dim_head,
        n_dim_ffn,
        n_non_bias_tokens,
        att_dropout, 
        ffn_dropout,
        position_embedding = 'alibi', # or rotary
        enable_skip_connections = True
    ):
        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_non_bias_tokens = n_non_bias_tokens
        self.enable_skip_connections = enable_skip_connections

        # Attention blocks
        self.layers = torch.nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(AttentionBlock(
                n_heads = n_heads, 
                n_dim = n_dim, 
                n_dim_head = n_dim_head, 
                n_dim_ffn = n_dim_ffn,
                att_dropout = att_dropout,
                ffn_dropout = ffn_dropout
            ))
        
        # Skip connections
        self.skip_combiners = torch.nn.ModuleList([])
        if enable_skip_connections:
            for i in range(n_layers//2):
                self.skip_combiners.append(torch.nn.Linear(n_dim * 2, n_dim))

        # Output normalization
        self.output_norm = RMSNorm(n_dim)

        # Positional embedding
        self.position_embedding = position_embedding
        if position_embedding == 'alibi':
            pass
        elif position_embedding == 'rotary':
            theta = 50000
            self.register_buffer('inv_freq', 1.0 / (theta ** (torch.arange(0, n_dim_head, 2).float() / n_dim)))
        else:
            raise ValueError(f"Unknown position embedding: {position_embedding}")


    def forward(self, x):
        batch, seq_len, *_ = x.shape

        # Embeddings
        alibi = None
        rotational = None

        # Compute ALiBi
        # This computes ALiBi bias mask, excluding non-bias tokens which are expected to be appended to the end of the sequence
        # Inspired by: https://github.com/ofirpress/attention_with_linear_biases/issues/5
        if self.position_embedding == 'alibi':
            alibi = get_alibi_mask(seq_len - self.n_non_bias_tokens, self.n_heads, x.device)
            if self.n_non_bias_tokens > 0:
                alibi = torch.nn.functional.pad(alibi, (0, self.n_non_bias_tokens, 0, self.n_non_bias_tokens), value=0)

        # Compute rotary embeddings
        if self.position_embedding == 'rotary':
            t = torch.arange(seq_len, device = self.inv_freq.device, dtype = self.inv_freq.dtype)
            freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
            rotational =  torch.cat((freqs, freqs), dim = -1)

        # Run through attention blocks
        connections = []
        for i in range(self.n_layers):

            # Skip connection
            if self.n_layers - (self.n_layers // 2) < i and self.enable_skip_connections:
                s = connections.pop()
                x = torch.cat([x, s], dim = -1)
                x = self.skip_combiners[i - (self.n_layers // 2)](x)

            # Attention
            x = self.layers[i](x, alibi = alibi, rotational = rotational)

            # Skip connection
            if i <= self.n_layers // 2:
                connections.append(x)

        # Output normalization
        x = self.output_norm(x)

        # Result
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_heads, n_dim, n_dim_head, n_dim_ffn, att_dropout, ffn_dropout):
        super(AttentionBlock, self).__init__()

        self.n_heads = n_heads
        self.n_dim_head = n_dim_head
        self.att_dropout = att_dropout

        # Attention input layer norm
        self.attention_ln = RMSNorm(n_dim)

        # Input -> Query/Key/Value for each head in single tensor for speedup
        self.attention = nn.Linear(n_dim, 3 * n_dim_head * n_heads, bias=False)
        torch.nn.init.normal_(self.attention.weight, mean=0.0, std=0.02)

        # Attention dropout
        # self.attention_dropout = nn.Dropout(att_dropout)

        # Output flatten multiple heads into single tensor
        self.attention_output = nn.Linear(n_dim_head * n_heads, n_dim, bias=False)
        torch.nn.init.normal_(self.attention_output.weight, mean=0.0, std=0.02)

        # Attention dropout
        # self.attention_output_dropout = nn.Dropout(dropout)

        # MLP part
        self.mlp_ln = RMSNorm(n_dim)
        self.mlp_input = nn.Linear(n_dim, n_dim_ffn)
        self.mlp_output = nn.Linear(n_dim_ffn, n_dim)
        self.mlp_output_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x, alibi = None, rotational = None):

        B, T, C = x.size() # batch size, sequence length, context width

        # Residual
        residual = x

        # Input normalization
        y = self.attention_ln(x)

        # Calculation Q/K/V for each head
        q, k, v = self.attention(y).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), (q, k, v))
        
        # Rotary embedding
        if rotational is not None:
            q = apply_rotary_pos_emb(rotational, q)
            k = apply_rotary_pos_emb(rotational, k)

        # Dot product attention
        # with torch.backends.cuda.sdp_kernel(enable_mem_efficient=True, enable_math=False): # Math backend is broken on mixed precision
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = alibi if alibi is not None else None, dropout_p=self.att_dropout if self.training else 0.0) # Using ALiBi as a mask

        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.n_dim_head) # re-assemble all head outputs side by side

        # Output
        y = self.attention_output(y)
        # y = self.attention_output_dropout(y)

        # Residual
        y = residual + y
        residual = y

        # MLP
        y = self.mlp_ln(y)
        y = self.mlp_input(y)
        y = F.gelu(y)
        y = self.mlp_output(y)
        y = self.mlp_output_dropout(y)
        y = residual + y

        return y

#
# Convolutional positional embedding
#

class ConvPositionEmbed(nn.Module):
    def __init__(self, n_dim, kernel_size):
        super().__init__()
        self.dw_conv1d = nn.Sequential(nn.Conv1d(n_dim, n_dim, kernel_size, groups = n_dim, padding = kernel_size // 2), nn.GELU())

    def forward(self, x, mask = None):

        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = rearrange(x, 'b c n -> b n c')

        if mask is not None:
            out = out.masked_fill(~mask, 0.)

        return out

#
# ALiBi implementation
#

slopes_cache = {}
def get_slopes_power_of_2(n_heads, device):
    global slopes_cache
    key = str(n_heads) + "_" + str(device)
    if key not in slopes_cache:
        start = (2**(-2**-(math.log2(n_heads)-3)))
        ratio = start
        slopes_cache[key] = torch.tensor([start*ratio**i for i in range(n_heads)], requires_grad=False, device = device) * -1
    return slopes_cache[key]

# alibi_cache = {}
# def get_alibi_mask(seq_len, n_heads, device):
#     global alibi_cache
#     key = str(seq_len) + "_" + str(n_heads) + "_" + str(device)

#     if key not in alibi_cache:
#         slopes = get_slopes_power_of_2(n_heads, device)
#         context_position = torch.arange(seq_len, device = device)[:, None]
#         memory_position = torch.arange(seq_len, device = device)[None, :]
#         relative_position = memory_position - context_position 
#         relative_position = torch.abs(relative_position).unsqueeze(0).expand(n_heads, -1,-1)
#         alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
#         alibi = alibi.view(1, n_heads, seq_len, seq_len)
#         alibi_cache[key] = alibi

#     return alibi_cache[key]

def get_alibi_mask(seq_len, n_heads, device):
    slopes = get_slopes_power_of_2(n_heads, device)
    context_position = torch.arange(seq_len, device = device)[:, None]
    memory_position = torch.arange(seq_len, device = device)[None, :]
    relative_position = memory_position - context_position 
    relative_position = torch.abs(relative_position).unsqueeze(0).expand(n_heads, -1,-1)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
    alibi = alibi.view(1, n_heads, seq_len, seq_len)
    return alibi


def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()