import math
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torchdiffeq import odeint

from .transformer import Transformer, ConvPositionEmbed
from .debug import debug_if_invalid

class AudioPredictor(torch.nn.Module):
    def __init__(self, config):
        super(AudioPredictor, self).__init__()
        self.n_tokens = len(config.tokenizer.tokens)
        self.config = config.audio_predictor

        # Token embedding
        self.token_embedding = torch.nn.Embedding(self.n_tokens, self.config.n_embeddings)

        # Transformer input
        self.transformer_input = torch.nn.Linear(self.config.n_embeddings + 2 * config.audio.n_mels, self.config.n_dim)

        # Sinusoidal positional embedding for time
        self.sinu_pos_emb = LearnedSinusoidalPosEmb(self.config.n_dim)

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = self.config.n_dim, kernel_size = 31)

        # Transformer
        self.transformer = Transformer(
            n_heads = self.config.n_heads,
            n_layers = self.config.n_layers,
            n_dim = self.config.n_dim,
            n_dim_head = self.config.n_dim_head,
            n_dim_ffn = self.config.n_dim_ffn,
            n_non_bias_tokens = 1, # Exclude time embedding from attention bias
            att_dropout = 0,
            ffn_dropout = 0.1
        )

        # Prediction
        self.prediction = torch.nn.Linear(self.config.n_dim, config.audio.n_mels)

    def sample(self, *, tokens, audio, mask, steps):
        
        #
        # Prepare
        #

        # Mask out y
        audio_masked = audio.masked_fill(mask.unsqueeze(-1), 0.0) # Mask need to be reshaped: (B, T) -> (B, T, 1)

        # Create noise
        noise = torch.randn_like(audio_masked).to(audio_masked.device)

        # Create time interpolation
        times = torch.linspace(0, 1, steps, device = audio_masked.device)

        # Solver
        def solver(t, z):
            return self.forward(tokens = tokens.unsqueeze(0), audio = audio_masked.unsqueeze(0), audio_noizy = z.unsqueeze(0), mask = mask.unsqueeze(0), times = t.unsqueeze(0)).squeeze(0)
        trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        # Output sample and full trajectory
        return trajectory[-1], trajectory

    def forward(self, *, tokens, audio, audio_noizy, mask, times, target = None, debug = False, debug_save = False):
        
        #
        # Prepare
        #

        # Check shapes
        assert tokens.shape[0] == audio.shape[0] == audio_noizy.shape[0] == mask.shape[0] # Batch
        assert tokens.shape[1] == audio.shape[1] == audio_noizy.shape[1] == mask.shape[1] # Sequence length

        # Mask out audio
        audio_masked = audio.masked_fill(mask.unsqueeze(-1), 0.0) # Mask need to be reshaped: (B, T) -> (B, T, 1)

        if debug:
            debug_if_invalid(target, 'target', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)

        #
        # Compute
        #

        # Convert phonemes to embeddings
        tokens_embed = self.token_embedding(tokens)
        if debug:
            debug_if_invalid(tokens_embed, 'tokens_embed', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)

        # Combine phoneme embeddings, masked audio and noizy audio
        output = torch.cat([tokens_embed, audio_masked, audio_noizy], dim = -1)
        if debug:
            debug_if_invalid(output, 'output_1', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)

        # Apply transformer input layer
        output = self.transformer_input(output)
        if debug:
            debug_if_invalid(output, 'output_2', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)

        # Apply sinusoidal positional embedding
        sinu_times = self.sinu_pos_emb(times).unsqueeze(1)
        output = torch.cat([output, sinu_times], dim=1)
        if debug:
            debug_if_invalid(output, 'output_3', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)

        # Apply convolutional positional encoder
        output = self.conv_embed(output) + output
        if debug:
            debug_if_invalid(output, 'output_4', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)

        # Run through transformer
        output = self.transformer(output)
        if debug:
            debug_if_invalid(output, 'output_5', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)

        # Predict durations
        output = self.prediction(output)
        if debug:
            debug_if_invalid(output, 'output_6', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)


        # Cut to length
        output = output[:, :-1, :]

        #
        # Loss
        #

        if target is not None:
            
            # Compute MSE loss
            loss = F.mse_loss(output, target, reduction = 'none')

            # Check if loss is nan
            if torch.isnan(loss).any():
                debug_if_invalid(loss, 'loss_mse', self, { 'tokens': tokens, 'audio': audio, 'audio_noizy': audio_noizy, 'mask': mask, 'times': times, 'target': target }, debug_save)

            # Mean for each frame
            loss = reduce(loss, 'b n d -> b n', 'mean')

            # Mask out non target frames
            loss = loss.masked_fill(~mask, 0.)

            # Number of masked frames
            n_masked_frames = mask.sum(dim = -1).clamp(min = 1)

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