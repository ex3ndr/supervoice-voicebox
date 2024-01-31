import torch

class Tokenizer:
    def __init__(self, config):
        self.silence_token = config.tokenizer.silence_token
        self.unknown_token = config.tokenizer.unknown_token
        self.tokens = config.tokenizer.tokens
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.n_tokens = len(self.tokens)

    def __call__(self, tokens):
        return torch.tensor([self.token_to_id[token] for token in tokens])
