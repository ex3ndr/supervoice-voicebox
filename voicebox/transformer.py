import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from torch.cuda.amp import autocast

class Transformer(nn.Module):
    def __init__(self, 
        n_heads,
        n_layers,
        n_dim,
        n_dim_head,
        n_dim_ffn,
        dropout
    ):
        super(Transformer, self).__init__()
        self.n_layers = n_layers

        # Attention blocks
        self.layers = torch.nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(AttentionBlock(
                n_heads = n_heads, 
                n_dim = n_dim, 
                n_dim_head = n_dim_head, 
                n_dim_ffn = n_dim_ffn,
                dropout = dropout
            ))
        
        # Skip connections
        self.skip_combiners = torch.nn.ModuleList([])
        for i in range(n_layers//2):
            self.skip_combiners.append(torch.nn.Linear(n_dim * 2, n_dim))

        # Output normalization
        self.output_norm = nn.LayerNorm(n_dim, bias=False)


    def forward(self, x, rotary_embed = None):
        batch, seq_len, *_ = x.shape

        # Run through attention blocks
        connections = []
        for i in range(self.n_layers):

            # Skip connection
            if self.n_layers - (self.n_layers // 2) < i:
                s = connections.pop() * 2 ** -0.5
                x = torch.cat([x, s], dim = -1)
                x = self.skip_combiners[i - (self.n_layers // 2)](x)

            # Attention
            x = self.layers[i](x, rotary_embed = rotary_embed)

            # Skip connection
            if i <= self.n_layers // 2:
                connections.append(x)

        # Output normalization
        x = self.output_norm(x)

        # Result
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_heads, n_dim, n_dim_head, n_dim_ffn, dropout):
        super(AttentionBlock, self).__init__()

        self.n_heads = n_heads
        self.n_dim_head = n_dim_head
        self.dropout = dropout

        # Attention input layer norm
        self.attention_ln = nn.LayerNorm(n_dim, bias=False)

        # Input -> Query/Key/Value for each head in single tensor for speedup
        self.attention = nn.Linear(n_dim, 3 * n_dim_head * n_heads, bias=False)
        torch.nn.init.normal_(self.attention.weight, mean=0.0, std=0.02)

        # Output flatten multiple heads into single tensor
        self.attention_output = nn.Linear(n_dim_head * n_heads, n_dim, bias=False)
        torch.nn.init.normal_(self.attention_output.weight, mean=0.0, std=0.02)

        # Attention dropout
        self.attention_output_dropout = nn.Dropout(dropout)

        # MLP part
        self.mlp_ln = nn.LayerNorm(n_dim, bias=False)
        self.mlp_input = nn.Linear(n_dim, n_dim_ffn)
        self.mlp_output = nn.Linear(n_dim_ffn, n_dim)
        self.mlp_output_dropout = nn.Dropout(dropout)

    def forward(self, x, rotary_embed = None):

        # Residual
        residual = x

        # Input normalization
        y = self.attention_ln(x)

        # Calculation Q/K/V for each head
        q, k, v = self.attention(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), (q, k, v))

        # Rotary embedding
        if rotary_embed is not None:
            q = apply_rotary_pos_emb(rotary_embed, q)
            k = apply_rotary_pos_emb(rotary_embed, k)

        # Dot product attention
        B, T, C = x.size() # batch size, sequence length, context width
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.n_dim_head) # re-assemble all head outputs side by side

        # Output
        y = self.attention_output(y)
        y = self.attention_output_dropout(y)

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
# Rotary positional embedding
#

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 50000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled = False)
    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.arange(t, device = self.device)
        t = t.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()