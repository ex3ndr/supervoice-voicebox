import torch
from .model_audio import AudioPredictor
from .model_duration import DurationPredictor
from .tokenizer import Tokenizer
from ..vocoder.model import Generator
from phonemizer.backend import EspeakBackend

class VoiceBox(torch.nn.Module):
    def __init__(self, config):
        super(VoiceBox, self).__init__()

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
        phonemized = self.phonemizer.phonemize(phone='|', word=' ')
        words = phonemized.split(' ')
        pass