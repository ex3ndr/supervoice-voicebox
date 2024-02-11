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

    def tts(self, text, steps = 4):

        # Clean up text
        text = Punctuation(';:,.!"?()-').remove(text)
        words = [w.lower() for w in text.strip().split(' ') if w]
        separator = Separator(phone='|', word=' ')

        # Phonemize
        phonemized = self.phonemizer.phonemize(words, separator = separator, strip = True)
        tokens = []
        first = True
        for p in phonemized:
            if not first:
                tokens.append(self.tokenizer.silence_token)
            first = False
            for t in p.split('|'):
                tokens.append(t)

        # Tokenize
        map_tokens = {
            'aɪ': 'aj',
            'oʊ': 'ow',
            'uː': 'ʉː',
            'ɔ': 'ɒ',
            'ʌ': 'ɑː',
            'əl': 'ə',
            'eɪ': 'ej',
            'iə': 'ə',
            'ɛɹ': 'ɛ',
            'ɜː': 'ə'
        }
        tokens = [map_tokens[t] if t in map_tokens else t for t in tokens]
        tokens = self.tokenizer(tokens)

        # Predict duration
        with torch.no_grad():
            duration = self.duration_model(
                tokens = tokens.unsqueeze(0),
                durations = torch.zeros(tokens.shape[0]).unsqueeze(0),
                mask = torch.ones(tokens.shape[0]).bool().unsqueeze(0)
            )

        # Prepare token tensor
        tokens_t = []
        for (t, d) in zip(tokens.tolist(), duration.squeeze(0).tolist()):
            for i in range(d * 2):
                tokens_t.append(t)
        tokens_t = torch.tensor(tokens_t)

        # Append silence
        tokens_t = torch.nn.functional.pad(tokens_t, (5, 5))

        # Predict audio
        with torch.no_grad():
            spectogram, _ = self.audio_model.sample(
                tokens = tokens_t, 
                audio = torch.zeros((tokens_t.shape[0], 80)),  # Empty source audio
                mask = torch.ones((tokens_t.shape[0])).bool(), # Mask everything
                steps = steps
            )
        
        # Rescale spectogram
        spectogram = (spectogram * 2.2615) + (-5.8843)

        # Vocoder
        with torch.no_grad():
            audio = self.vocoder(spectogram.transpose(1,0).unsqueeze(0)).squeeze(0)
        
        return spectogram, audio