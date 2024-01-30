import torch
import torch.nn.functional as F
import math
from .transformer import Transformer, ConvPositionEmbed
from einops import rearrange, reduce, repeat
from torchdiffeq import odeint

class AudioModel(torch.nn.Module):
    def __init__(self, n_tokens):
        super(AudioModel, self).__init__()
        self.n_tokens = n_tokens

        # Architecture
        n_dim_head = 64

        # Token embedding
        self.token_embedding = torch.nn.Embedding(n_tokens, 1024)

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = 1024, kernel_size = 31)

        # Sinusoidal positional embedding for time
        self.sinu_pos_emb = LearnedSinusoidalPosEmb(1024)

        # Transformer input
        self.transformer_input = torch.nn.Linear(1024 + 80 + 80, 1024)

        # Transformer
        self.transformer = Transformer(
            n_heads = 16,
            n_layers = 12,
            n_dim = 1024,
            n_dim_head = n_dim_head,
            n_dim_ffn = 4096,
            n_non_bias_tokens = 1, # Exclude time embedding from attention bias
            dropout = 0.1
        )

        # Prediction
        self.prediction = torch.nn.Linear(1024, 80)

    def sample(self, x, y, mask, steps):
        
        #
        # Prepare
        #

        # Mask out y
        y_masked = y.masked_fill(mask.unsqueeze(-1), 0.0) # Mask need to be reshaped: (B, T) -> (B, T, 1)

        # Create noise
        noise = torch.randn_like(y).to(x.device)

        # Create time interpolation
        times = torch.linspace(0, 1, steps, device = x.device)

        # Solver
        def solver(t, z):
            return self.forward(x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0), mask.unsqueeze(0), times = t.unsqueeze(0)).squeeze(0)
        trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        # Output sample and full trajectory
        return trajectory[-1], trajectory

    def forward(self, x, y, z, mask, times, target = None):
        
        #
        # Prepare
        #

        # Check shapes
        assert x.shape[0] == y.shape[0] == z.shape[0] == mask.shape[0] # Batch
        assert x.shape[1] == y.shape[1] == z.shape[1] == mask.shape[1] # Sequence length

        # Mask out y
        y_masked = y.masked_fill(mask.unsqueeze(-1), 0.0) # Mask need to be reshaped: (B, T) -> (B, T, 1)

        #
        # Compute
        #

        # Convert phonemes to embeddings
        x = self.token_embedding(x)

        # Combine phoneme embeddings, masked audio and noizy audio
        output = torch.cat([x, y_masked, z], dim = -1)

        # Apply transformer input layer
        output = self.transformer_input(output)

        # Apply sinusoidal positional embedding
        sinu_times = self.sinu_pos_emb(times).unsqueeze(1)
        output = torch.cat([output, sinu_times], dim=1)

        # Apply convolutional positional encoder
        output = self.conv_embed(output) + output

        # Run through transformer
        output = self.transformer(output)

        # Predict durations
        output = self.prediction(output)
        output = output[:, :-1, :] # Cut to length

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


class LearnedSinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        self.weights = torch.nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered