import math
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torchdiffeq import odeint

from .transformer import Transformer, ConvPositionEmbed
from .debug import debug_if_invalid
from .tensors import drop_using_mask, merge_mask

class AudioPredictor(torch.nn.Module):
    def __init__(self, config):
        super(AudioPredictor, self).__init__()
        self.n_tokens = len(config.tokenizer.tokens)
        self.n_style_tokens = config.tokenizer_style.tokens + 1 # +1 Padding
        self.config = config.audio_predictor

        # Token and embedding
        self.token_embedding = torch.nn.Embedding(self.n_tokens, self.config.n_embeddings)
        self.style_embedding = torch.nn.Embedding(self.n_style_tokens, self.config.n_embeddings)

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

    def sample(self, *, tokens, tokens_style, audio, mask, steps, alpha = None):
        
        #
        # Prepare
        #

        # Mask out audio
        source_audio = audio
        audio = drop_using_mask(source = audio, replacement = 0, mask = mask)

        # Create noise
        noise = torch.randn_like(audio)

        # Create time interpolation
        times = torch.linspace(0, 1, steps, device = audio.device)

        #
        # Solver
        # 

        # Overwrite audio segment with predicted audio according to mask
        def merge_predicted(predicted):
            return merge_mask(source = source_audio, replacement = predicted, mask = mask)

        def solver(t, z):

            # If alpha is not provided
            if alpha is None:
                return self.forward(
                    tokens = tokens.unsqueeze(0), 
                    tokens_style = tokens_style.unsqueeze(0), 
                    audio = audio.unsqueeze(0), 
                    audio_noizy = z.unsqueeze(0), 
                    times = t.unsqueeze(0)
                ).squeeze(0)

            # If alpha is provided - zero out tokens and audio and mix together
            tokens_empty = torch.zeros_like(tokens)
            audio_empty = torch.zeros_like(audio)

            # Mix together
            tokens_t = torch.stack([tokens_empty, tokens], dim = 0)
            tokens_style_t = torch.stack([tokens_empty, tokens_style], dim = 0)
            audio_t = torch.stack([audio_empty, audio], dim = 0)
            audio_noizy_t = torch.stack([z, z], dim = 0) # Just double it
            t_t = torch.stack([t, t], dim = 0) # Just double it

            # Inference
            predicted_mix = self.forward(
                tokens = tokens_t, 
                tokens_style = tokens_style_t, 
                audio = audio_t, 
                audio_noizy = audio_noizy_t, 
                times = t_t
            )
            predicted_conditioned = predicted_mix[1]
            predicted_unconditioned = predicted_mix[0]
            
            # CFG prediction
            # NOTE: One paper has a mistake in the formula and it used addition instead of subtraction
            prediction = (1 + alpha) * predicted_conditioned - alpha * predicted_unconditioned

            return prediction


        trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        #
        # Output sample and full trajectory
        #

        return merge_predicted(trajectory[-1]), trajectory

    def forward(self, *, 
        # Tokens
        tokens, 
        tokens_style,
        
        # Audio
        audio, 
        audio_noizy, 
        
        # Time
        times, 

        # Training    
        mask = None,
        target = None
    ):
        
        #
        # Prepare
        #

        if mask is None and target is not None:
            raise ValueError('Mask is required when target is provided')
        if target is None and mask is not None:
            raise ValueError('Mask is not required when target is not provided')

        # Check shapes
        assert tokens.shape[0] == audio.shape[0] == audio_noizy.shape[0] == tokens_style.shape[0] # Batch
        assert tokens.shape[1] == audio.shape[1] == audio_noizy.shape[1] == tokens_style.shape[1] # Sequence length
        if mask is not None:
            assert tokens.shape[0] == mask.shape[0]
            assert tokens.shape[1] == mask.shape[1]

        #
        # Compute
        #

        # Convert phoneme and style tokens to embeddings
        tokens_embed = self.token_embedding(tokens)
        tokens_embed += self.style_embedding(tokens_style)

        # Combine phoneme embeddings, masked audio and noizy audio
        output = torch.cat([tokens_embed, audio, audio_noizy], dim = -1)

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

        # Cut to length
        output = output[:, :-1, :]

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