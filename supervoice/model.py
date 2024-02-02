import torch
from .model_audio import AudioPredictor
from .model_duration import DurationPredictor
from .model_vocoder import Generator
from .tokenizer import Tokenizer
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

class SuperVoice(torch.nn.Module):
    def __init__(self, config):
        super(SuperVoice, self).__init__()

        # Create Tokenizer
        self.tokenizer = Tokenizer(config)

        # Create Audio Model
        self.audio_model = AudioPredictor(config)

        # Create Duration Model
        self.duration_model = DurationPredictor(config)

        # Create vocoder
        self.vocoder = Generator(config)

        # Create phonemizer
        self.phonemizer = EspeakBackend('en-us')

    def tts(self, text):

        # Phonemize
        text = Punctuation(';:,.!"?()-').remove(text)
        words = {w.lower() for w in text.strip().split(' ') if w}
        separator = Separator(phone='|', word=' ')

        # Phonemize
        phonemized = self.phonemizer.phonemize(words, separator = separator, strip = True)
        tokens = []
        for p in phonemized:
            tokens.append(self.tokenizer.silence_token)
            for t in p.split('|'):
                tokens.append(t)
        tokens.append(self.tokenizer.silence_token)

        # Tokenize
        # tokens = self.tokenizer(tokens)
        
        return tokens