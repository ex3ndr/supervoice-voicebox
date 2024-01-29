import torch
from einops import rearrange

def sample_noisy_value(source, times, sigma):
    
    # Random noize
    noise = torch.randn_like(source) # (B, T, C)

    # Random times [0...1)
    t = rearrange(times, 'b -> b 1 1')

    # sample xt (w in the paper)
    w = (1 - (1 - sigma) * t) * noise + t * source
    flow = source - (1 - sigma) * noise

    return flow