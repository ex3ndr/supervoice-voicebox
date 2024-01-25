import glob
import torch
import torchaudio
import random
import csv
from tqdm import tqdm
from .audio import load_mono_audio

class SimpleAudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, files, sample_rate, samples_size, vad = False, transformer = None):
        self.files = files
        self.sample_rate = sample_rate
        self.samples_size = samples_size
        self.transformer = transformer
        self.vad = None
        if vad:
            self.vad = torchaudio.transforms.Vad(sample_rate=self.sample_rate)
    
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

        # VAD
        if self.vad is not None:
            audio = audio.unsqueeze(0)

            # Trim front
            audio = self.vad(audio)

            # Trin end
            audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, self.sample_rate, [["reverse"]])
            audio = self.vad(audio)
            audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, self.sample_rate, [["reverse"]])

            audio = audio.squeeze(0)

        # Transformer
        if self.transformer is not None:
            return self.transformer(audio)
        else:
            return audio

    def __len__(self):
        return len(self.files)


def load_common_voice_files(path, split):
    res = []
    with open(path + f'{split}.tsv') as csvfile:
        cvs_reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        next(cvs_reader, None)  # skip the headers
        return [path + 'clips/' + row[1] for row in cvs_reader]