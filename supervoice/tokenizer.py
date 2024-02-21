import torch

class Tokenizer:
    def __init__(self, config):
        self.silence_token = config.tokenizer.silence_token
        self.unknown_token = config.tokenizer.unknown_token
        self.begin_token = config.tokenizer.begin_token
        self.end_token = config.tokenizer.end_token
        self.tokens = config.tokenizer.tokens
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.n_tokens = len(self.tokens)
        self.unknown_token_id = self.token_to_id[self.unknown_token]
        self.silence_token_id = self.token_to_id[self.silence_token]
        self.begin_token_id = self.token_to_id[self.begin_token]
        self.end_token_id = self.token_to_id[self.end_token]

    def __call__(self, tokens, force = False):
        if force:
            return torch.tensor([(self.token_to_id[token] if token in self.token_to_id else self.unknown_token_id) for token in tokens])
        else:
            missing = []
            for token in tokens:
                if token not in self.token_to_id:
                    missing.append(token)
            if missing:
                raise ValueError(f"Tokens not found: {missing}")
            return torch.tensor([self.token_to_id[token] for token in tokens])