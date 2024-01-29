import torch
import torch.nn.functional as F
from .transformer import Transformer, RotaryEmbedding, ConvPositionEmbed
from .cfm import sample_noisy_value
from einops import rearrange, reduce

class AudioModel(torch.nn.Module):
    def __init__(self, n_tokens):
        super(AudioModel, self).__init__()

        # Token embedding
        self.token_embedding = torch.nn.Embedding(n_tokens, 1024)

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = 1024, kernel_size = 31)

        # Rotational embedding
        self.rotary_embed = RotaryEmbedding(dim = 64)

        # Transformer input
        self.transformer_input = torch.nn.Linear(1024 + 80 + 80, 1024)

        # Transformer
        self.transformer = Transformer(
            n_heads = 16,
            n_layers = 24,
            n_dim = 1024,
            n_dim_head = 64,
            n_dim_ffn = 4096,
            dropout = 0.1
        )

        # Prediction
        self.prediction = torch.nn.Linear(1024, 80)

    def forward(self, x, y, mask, times = None, target = None):
        
        #
        # Prepare
        #

        # Check shapes
        assert x.shape[0] == y.shape[0] == mask.shape[0] # Batch
        assert x.shape[1] == y.shape[1] == mask.shape[1] # Sequence length

        # Mask out y
        y_masked = y.masked_fill(mask.unsqueeze(-1), 0.0) # Mask need to be reshaped: (B, T) -> (B, T, 1)

        # Calculate random times if not provided
        if times is None:
            times = torch.rand((x.shape[0],), dtype = y.dtype, device = y.device)

        #
        # Compute noizy audio (CFM)
        # 

        z = sample_noisy_value(y, times, sigma = 0.1) # What sigma to use?

        #
        # Compute
        #

        # Convert phonemes to embeddings
        x = self.token_embedding(x)

        # Combine phoneme embeddings, masked audio and noizy audio
        output = torch.cat([x, y_masked, z], dim = -1)

        # Apply transformer input layer
        output = self.transformer_input(output)

        # Apply convolutional positional encoder
        output = self.conv_embed(output) + output

        # Run through transformer
        rotary_embed = self.rotary_embed(output.shape[1])
        output = self.transformer(output, rotary_embed = rotary_embed)

        # Predict durations
        output = self.prediction(output)

        #
        # Loss
        #

        if target is not None:
            
            # Compute MSE loss
            loss = F.mse_loss(output, target, reduction = 'none')

            # Mean for each frame
            loss = reduce(loss, 'b n d -> b n', 'mean')

            # Mask out non target frames
            loss = loss.masked_fill(~mask, 0.)

            # Number of masked frames
            n_masked_frames = mask.sum(dim = -1).clamp(min = 1e-5)

            # Mean loss of expectation over masked loss
            loss = loss.sum(dim = -1) / n_masked_frames

            # Expectation over loss of batch
            loss = loss.mean()

            return output, loss
        else:
            return output