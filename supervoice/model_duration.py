import torch
import torch.nn.functional as F
from .transformer import Transformer, ConvPositionEmbed

class DurationPredictor(torch.nn.Module):
    def __init__(self, config):
        super(DurationPredictor, self).__init__()
        self.config = config.duration_predictor
        self.n_tokens = len(config.tokenizer.tokens)
        
        # Embedding
        self.token_embedding = torch.nn.Embedding(self.n_tokens, self.config.n_embeddings)

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = self.config.n_embeddings, kernel_size = 31)

        # Transformer input
        self.transformer_input = torch.nn.Linear(self.config.n_embeddings + 1, self.config.n_dim)
        
        # Transformer
        self.transformer = Transformer(
            n_heads = self.config.n_heads,
            n_layers = self.config.n_layers,
            n_dim = self.config.n_dim,
            n_dim_head = self.config.n_dim_head,
            n_dim_ffn = self.config.n_dim_ffn,
            n_non_bias_tokens = 0,
            att_dropout = 0,
            ffn_dropout = 0.1
        )

        # Prediction
        self.prediction = torch.nn.Linear(self.config.n_dim, 1)

    def forward(self, *, tokens, durations, mask, target = None):

        #
        # Prepare
        #

        # Check shapes
        assert tokens.shape[0] == durations.shape[0] == mask.shape[0] # Batch
        assert tokens.shape[1] == durations.shape[1] == mask.shape[1] # Sequence length

        # Reshape inputs
        mask = mask.unsqueeze(-1) # (B, T) -> (B, T, 1)
        durations = durations.unsqueeze(-1) # (B, T) -> (B, T, 1)

        # Convert durations to log durations
        durations = torch.log(durations.float() + 1)

        # Mask out y
        durations_masked = durations.masked_fill(mask, 0.0)

        #
        # Compute
        #

        # Convert phonemes to embeddings
        tokens_embeddings = self.token_embedding(tokens)

        # Combine duration and phoneme embeddings
        output = torch.cat([durations_masked, tokens_embeddings], dim = -1)

        # Apply transformer input layer
        output = self.transformer_input(output)

        # Apply convolutional positional encoder
        output = self.conv_embed(output) + output

        # Run through transformer
        output = self.transformer(output)

        # Predict durations
        output_log = self.prediction(output)

        #
        # Output
        #

        # Convert predicted log durations back to durations
        output = torch.clamp(output_log.exp() - 1, min=0).long()
        output = output.squeeze(-1) # (B, T, 1) -> (B, T)

        #
        # Loss
        #

        if target is not None:

            # Convert target durations to log durations
            target = torch.log(target.float() + 1)

            # Update shape (B, T) -> (B, T, 1)
            target = target.unsqueeze(-1)

            # Compute l1 loss
            loss = F.l1_loss(output_log, target, reduction = 'none')

            # Zero non-masked values
            loss = loss.masked_fill(~mask, 0.)

            # Number of masked frames
            # n_masked_frames = mask.sum(dim = -1).clamp(min = 1e-5)
            n_masked_frames = mask.sum(dim = -1).clamp(min = 1)

            # Mean loss of expectation over masked loss
            loss = loss.sum(dim = -1) / n_masked_frames

            # Expectation over loss of batch
            loss = loss.mean()

            return output, loss
        else:
            return output