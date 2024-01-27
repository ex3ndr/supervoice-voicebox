import torch

from voicebox_pytorch import (
    VoiceBox,
    ConditionalFlowMatcherWrapper,
    DurationPredictor
)

class VoiceBoxModule(torch.nn.Module):
    def __init__(self, config):
        super(VoiceBoxModule, self).__init__()

        self.box = VoiceBox(
            dim = config.audio.num_mels,
            num_cond_tokens = 500,
            depth = 2,
            dim_head = 64,
            heads = config.voicebox.n_heads,
        )

        self.cfm_wrapper = ConditionalFlowMatcherWrapper(voicebox = self.box)

    def forward(self, x):
        return self.cfm_wrapper(x)