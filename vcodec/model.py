import torch
from vector_quantize_pytorch import GroupedResidualVQ

class VCodec(torch.nn.Module):
    def __init__(self, config):
        super(VCodec, self).__init__()

        # Assert
        assert config.vcodec.tokens % config.vcodec.groups == 0, "Number of tokens must be divisible by number of groups"
        assert config.vcodec.quantizers % config.vcodec.groups == 0, "Number of quantizers must be divisible by number of groups"

        # VQ
        self.residual_vq = GroupedResidualVQ(
            dim = config.audio.num_mels,
            num_quantizers = config.vcodec.quantizers // config.vcodec.groups,
            groups = config.vcodec.groups,
            codebook_size = config.vcodec.tokens // config.vcodec.groups,
            learnable_codebook = True,
            requires_projection = True,
        )

    def forward(self, x):

        # Residual VQ
        quantized, all_indices, commit_losses = self.residual_vq(x)

        # Reconstruction loss
        recon_loss = ((x - quantized) ** 2).mean(dim=-1)

        # Loss
        commit_losses = commit_losses.mean()
        loss = recon_loss + commit_losses

        return (quantized, loss, commit_losses, recon_loss)