import random
import torch

def debug_if_invalid(x, name, model, ctx = None, save = False):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print('Invalid tensor ' + name)

        print(name, x)
        if save:
            torch.save(x, "debug_" + name + ".pt")
            torch.save(model.state_dict(), "debug_" + name + "_model.pt")

        if ctx is not None:
            for k in ctx:
                v = ctx[k]
                print(k, v)
                if save:
                    torch.save(v, "debug_" + name + "_" + k + ".pt")

        raise RuntimeError(name + " is NaN or Inf")

def deterministic_random(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False