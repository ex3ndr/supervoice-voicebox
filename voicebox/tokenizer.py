import torch

class Tokenizer:
    def __init__(self):
        self.silence_token = 'SIL'
        self.tokens = ['SIL', 'aj', 'aw', 'aː', 'b', 'bʲ', 'c', 'cʰ', 'cʷ', 'd', 'dʒ', 'dʲ', 'd̪', 'ej', 'f', 'fʲ', 'h', 'i', 'iː', 'j', 'k', 'kʰ', 'kʷ', 'l', 'm', 'mʲ', 'm̩', 'n', 'n̩', 'ow', 'p', 'pʰ', 'pʲ', 's', 'spn', 't', 'tʃ', 'tʰ', 'tʲ', 'tʷ', 't̪', 'v', 'vʲ', 'w', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔj', 'ə', 'ɚ', 'ɛ', 'ɝ', 'ɟ', 'ɟʷ', 'ɡ', 'ɪ', 'ɫ', 'ɫ̩', 'ɱ', 'ɲ', 'ɹ', 'ɾ', 'ɾʲ', 'ɾ̃', 'ʃ', 'ʉ', 'ʉː', 'ʊ', 'ʎ', 'ʒ', 'ʔ', 'θ']
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.n_tokens = len(self.tokens)

    def __call__(self, tokens):
        return torch.tensor([self.token_to_id[token] for token in tokens])
