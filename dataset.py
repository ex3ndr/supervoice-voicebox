import glob
import torch
import torchaudio
import random
from audio import load_mono_audio

class SimpleAudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, files, sample_rate, samples_size, transformer = None):
        self.files = files
        self.sample_rate = sample_rate
        self.samples_size = samples_size
        self.transformer = transformer
    
    def __getitem__(self, index):

        # Load File
        filename = self.files[index]

        # Load audio
        audio = load_mono_audio(filename, self.sample_rate)

        # Pad or trim to target duration
        if audio.shape[0] >= self.samples_size:
            audio_start = random.randint(0, audio.shape[0] - self.samples_size)
            audio = audio[audio_start:audio_start+self.samples_size]
        elif audio.shape[0] < self.samples_size: # Rare or impossible case - just pad with zeros
            audio = torch.nn.functional.pad(audio, (0, self.samples_size - audio.shape[0]))

        # Transformer
        if self.transformer is not None:
            return self.transformer(audio)
        else:
            return audio

    def __len__(self):
        return len(self.files)