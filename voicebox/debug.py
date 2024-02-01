import torch

def debug_if_invalid(x, name):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(x)
        print('Invalid tensor ' + name)