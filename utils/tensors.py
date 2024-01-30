import torch
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma