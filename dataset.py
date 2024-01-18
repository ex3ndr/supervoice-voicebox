import glob
import torch
import torchaudio
from audio import load_mono_audio

def search_wav_files(dir):
    with open(index, 'r', encoding='utf-8') as fi:
        return [os.path.join(wav_dir, x.split('|')[0] + '.wav') for x in fi.read().split('\n') if len(x) > 0]


class SimpleAudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, files, sample_rate, samples_size):
        self.files = files
        self.sample_rate = sample_rate
        self.samples_size = samples_size
    
    def __getitem__(self, index):

        # Load File
        filename = self.files[index]

        # Load audio
        audio = load_mono_audio(filename, self.sample_rate)

        # Pad or trim to target duration
        if audio.shape[0] > self.samples_size:
            audio = audio[:self.samples_size]
        elif audio.shape[0] < self.samples_size:
            audio = torch.nn.functional.pad(audio, (0, self.samples_size - audio.shape[0]))

        return audio

    def __len__(self):
        return len(self.files)