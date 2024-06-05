import math
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torchdiffeq import odeint
from .tensors import drop_using_mask, merge_mask

class AudioPredictor(torch.nn.Module):
    def __init__(self, flow, config):
        super(AudioPredictor, self).__init__()
        self.n_tokens = len(config.tokenizer.tokens)
        self.config = config.audio_predictor
        self.flow = flow
        self.flow.transformer.cache_alibi = False

        # Token and embedding
        if self.config.use_original_conditioning:
            self.token_embedding = torch.nn.Embedding(self.n_tokens, config.audio.n_mels)
        else:
            self.token_embedding = torch.nn.Embedding(self.n_tokens, self.config.n_embeddings)
            self.conditioning = torch.nn.Linear(self.config.n_embeddings, self.config.n_dim)


    def sample(self, *, tokens, audio, mask, steps, alpha = None):
        
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
                    audio = audio.unsqueeze(0), 
                    noise = z.unsqueeze(0), 
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
                audio = audio_t, 
                noise = audio_noizy_t, 
                times = t_t
            )
            predicted_conditioned = predicted_mix[1]
            predicted_unconditioned = predicted_mix[0]
            
            # CFG prediction

            # There are different ways to do CFG, this is my very naive version, which worked for me:
            # prediction = (1 + alpha) * predicted_conditioned - alpha * predicted_unconditioned

            # Original paper uses a different one, but i found that it simply creates overexposed values
            # prediction = predicted_unconditioned + (predicted_conditioned - predicted_unconditioned) * alpha

            # This is from the latest paper that rescales original formula (https://arxiv.org/abs/2305.08891):
            prediction = predicted_conditioned + (predicted_conditioned - predicted_unconditioned) * alpha
            prediction_rescaled = predicted_conditioned.std() * (prediction / prediction.std())

            return prediction


        trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        #
        # Output sample and full trajectory
        #

        return merge_predicted(trajectory[-1]), trajectory

    def forward(self, *, 
        # Inputs
        tokens, 
        audio, 
        noise, 

        # Time
        times, 

        # Training    
        mask = None,
        target = None,
        token_scale = 1
    ):
        
        #
        # Prepare
        #

        if mask is None and target is not None:
            raise ValueError('Mask is required when target is provided')
        if target is None and mask is not None:
            raise ValueError('Mask is not required when target is not provided')

        # Check shapes
        assert tokens.shape[0] == audio.shape[0] == noise.shape[0] # Batch
        assert tokens.shape[1] == audio.shape[1] == noise.shape[1] # Sequence length
        if mask is not None:
            assert tokens.shape[0] == mask.shape[0]
            assert tokens.shape[1] == mask.shape[1]

        #
        # Conditioning
        #

        conditioning = None
        if self.config.use_original_conditioning:
            tokens_embed = self.token_embedding(tokens) * token_scale
            audio = audio + tokens_embed            
        else:
            tokens_embed = self.token_embedding(tokens)
            conditioning = self.conditioning(tokens_embed)

        #
        # Speech Flow
        #

        return self.flow(

            # Inputs
            audio = audio,
            noise = noise,
            condition = conditioning,

            # Time
            times = times,
            
            # Loss
            mask = mask,
            mask_loss = True,
            target = target
        )