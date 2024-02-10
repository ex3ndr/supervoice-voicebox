import torch

class Tokenizer:
    def __init__(self, config):
        self.silence_token = config.tokenizer.silence_token
        self.unknown_token = config.tokenizer.unknown_token
        self.tokens = config.tokenizer.tokens
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.n_tokens = len(self.tokens)
        self.unknown_token_id = self.token_to_id[self.unknown_token]
        self.silence_token_id = self.token_to_id[self.silence_token]

    def __call__(self, tokens):
        return torch.tensor([(self.token_to_id[token] if token in self.token_to_id else self.unknown_token_id) for token in tokens])
