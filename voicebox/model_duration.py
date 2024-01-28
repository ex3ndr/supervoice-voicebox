import torch
import torch.nn.functional as F
from .transformer import Transformer, RotaryEmbedding, ConvPositionEmbed

class DurationPredictor(torch.nn.Module):
    def __init__(self, n_tokens):
        super(DurationPredictor, self).__init__()
        
        # Embedding
        self.token_embedding = torch.nn.Embedding(n_tokens, 512) # Keep one for duration

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = 512, kernel_size = 31)

        # Rotational embedding
        self.rotary_embed = RotaryEmbedding(dim = 512)

        # Transformer input
        self.transformer_input = torch.nn.Linear(512 + 1, 512)
        
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
        self.prediction = torch.nn.Linear(512, 1)

    def forward(self, x, y, mask, target = None):

        #
        # Prepare
        #

        # Check shapes
        assert x.shape[0] == y.shape[0] == mask.shape[0] # Batch
        assert x.shape[1] == y.shape[1] == mask.shape[1] # Sequence length

        # Convert durations to log durations
        y = torch.log(y.float() + 1)

        # Mask out y
        y_masked = y.masked_fill(mask, 0.0)
        y_masked = y_masked.unsqueeze(-1) # (B, T) -> (B, T, 1)

        #
        # Compute
        #

        # Convert phonemes to embeddings
        x = self.token_embedding(x)

        # Combine duration and phoneme embeddings
        z = torch.cat([y_masked, x], dim = -1)

        # Apply transformer input layer
        z = self.transformer_input(z)

        # Apply convolutional positional encoder
        z = self.conv_embed(z) + z

        # Run through transformer
        rotary_embed = self.rotary_embed(z.shape[1])
        z = self.transformer(z, rotary_embed = rotary_embed)

        # Predict durations
        z = self.prediction(z)

        #
        # Output
        #

        # Convert predicted log durations back to durations
        predictions = torch.clamp(torch.round(z.exp() - 1), min=0).long()
        predictions = predictions.squeeze(-1) # (B, T, 1) -> (B, T)

        #
        # Loss
        #

        if target is not None:

            # Update shape (B, T) -> (B, T, 1)
            target = target.unsqueeze(-1)

            # Compute l1 loss
            loss = F.l1_loss(z, target, reduction = 'none')

            # Zero non-masked values
            loss = loss.masked_fill(~mask, 0.)

            # Number of masked frames
            n_masked_frames = mask.sum(dim = -1).clamp(min = 1e-5)

            # Mean loss of expectation over masked loss
            loss = loss.sum(dim = -1) / n_masked_frames

            # Expectation over loss of batch
            loss = loss.mean()

            return predictions, loss
        else:
            return predictions