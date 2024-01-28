import torch
from torch import nn
from einops import rearrange, repeat, reduce, pack, unpack

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
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(AttentionBlock(
                n_heads = n_heads, 
                n_dim = n_dim, 
                n_dim_head = n_dim_head, 
                n_dim_ffn = n_dim_ffn,
                dropout = attn_dropout
            ))
        
        # Skip connections
        self.skip_combiners = nn.ModuleList([])
        for i in range(n_layers//2):
            self.skip_combiners.append(nn.Linear(n_dim * 2, n_dim))


    def forward(self, x, mask = None):
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
            x = self.layers[i](x, mask = mask)

            # Skip connection
            if i <= self.n_layers // 2:
                connections.append(x)

        # Result
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, n_heads, n_dim, n_dim_head, n_dim_ffn, dropout):
        super(Attention, self).__init__()

        self.n_heads = n_heads
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

    def forward(self, x, mask = None):
        residual = x # Save for residual connection

        # Input normalization
        y = self.attention_ln(x)

        # Calculation Q/K/V for each head
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), (q, k, v))

        # Dot product attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output
        y = self.output(y)
        y = self.output_dropout(y)
        y = residual + y

        # MLP
        residual = y
        y = self.mlp_ln(y)
        y = self.mlp_input(y)
        y = F.gelu(y)
        y = self.mlp_output(y)
        y = residual + self.mlp_output_dropout(y)

        return y

class ConvPositionEmbed(Module):
    def __init__(self, n_dim, kernel_size):
        super().__init__()
        assert is_odd(kernel_size)

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