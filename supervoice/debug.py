import random
import torch

def debug_if_invalid(x, name):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(x)
        print('Invalid tensor ' + name)

def deterministic_random(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False