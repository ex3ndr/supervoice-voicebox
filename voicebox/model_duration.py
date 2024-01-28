import torch
from transformer import Transformer, ConvPositionEmbed

class DurationPredictor(torch.nn.Module):
    def __init__(self, n_tokens):
        super(DurationPredictor, self).__init__()
        
        # Embedding
        self.token_embedding = nn.Embedding(n_tokens, 511) # Keep one for duration

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = 512, kernel_size = 31)
        
        # Transformer
        self.transformer = Transformer(
            n_heads = 8,
            n_layers = 8,
            n_dim = 512,
            n_dim_head = 512, # ??
            n_dim_ffn = 2048,
            dropout = 0.1
        )

        # Prediction
        self.prediction = nn.Linear(n_dim, 1)

    def forward(self, x, y, mask):

        # Check shapes
        assert x.shape[0] == y.shape[0] == mask.shape[0] # Batch
        assert x.shape[1] == y.shape[1] == mask.shape[1] # Sequence length

        # Convert durations to log durations
        y = torch.log(y.float() + 1)

        # Convert phonemes to embeddings
        x = self.token_embedding(x)

        # Combine duration and phoneme embeddings
        z = torch.cat([x, y], dim = -1)

        # Apply convolutional positional encoder
        z = self.conv_embed(z, mask = mask) + z

        # Run through transformer
        z = self.transformer(z, mask = mask)

        # Predict durations
        z = self.prediction(z)

        # Convert predicted log durations back to durations
        z = torch.clamp(torch.round(z.exp() - 1), min=0).long()

        return z