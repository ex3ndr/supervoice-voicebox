import torch
import torch.nn.functional as F
import random

class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma
    

def probability_binary_mask(shape, true_prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < true_prob


def debug_if_invalid(x):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print('Invalid tensor')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def drop_using_mask(source, replacement, mask):
    while mask.dim() < source.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, torch.full(source.shape, replacement, dtype = source.dtype, device = source.device), source)

def merge_mask(source, replacement, mask):
    while mask.dim() < source.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, replacement, source)

def interval_mask(batch_size, length, min_interval, max_interval, probability_all, device):
    tensor = torch.full((batch_size, length), False, device = device, dtype = torch.bool)
    for i in range(batch_size):
        interval_length = random.randint(min_interval, max_interval)
        if random.random() < probability_all or interval_length == length:
            tensor[i] = True
            continue
        start_point = random.randint(0, length - interval_length - 1)
        tensor[i, start_point:start_point + interval_length] = True 
    return tensor