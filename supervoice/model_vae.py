import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ

class VAEModel(nn.Module):
    def __init__(self, config):
        super(VAEModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(100, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.quantizer = ResidualVQ(
            dim = 512,
            num_quantizers = 8,
            codebook_size = 1024,
            stochastic_sample_codes = True,
            sample_codebook_temp = 0.1,
            shared_codebook = True    
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 100, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

    def encode(self, x):
        y = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        y, _, _ = self.quantizer(y)
        return y
    
    def decode(self, x):
        return self.decoder(x.transpose(1, 2)).transpose(1, 2)
    
    def forward(self, x):
        y = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        y, _, l = self.quantizer(y)
        y = self.decode(y)
        loss = F.mse_loss(x, y) + l.mean()
        return loss