import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ

class VAEModel(nn.Module):
    def __init__(self, config):
        super(VAEModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(100, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 100, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

    def encode(self, x):
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)
    
    def decode(self, x):
        return self.decoder(x.transpose(1, 2)).transpose(1, 2)
    
    def forward(self, x):
        y = self.encode(x)
        y = self.decode(y)
        loss = F.mse_loss(x, y)
        return loss