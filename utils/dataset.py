from glob import glob
import torch
import torchaudio
import random
import csv
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import math
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import textgrid
from supervoice.model_style import resolve_style
from supervoice.alignment import compute_alignments
from supervoice.config import config

class AudioFileListDataset(torch.utils.data.Dataset):
    
    def __init__(self, path, segment_size, limit=None, transformer = None):
        self.segment_size = segment_size
        self.transformer = transformer
        with open(path, 'r') as filelist:
            self.rows = list(filelist)
        random.shuffle(self.rows)
        if limit is not None:
            self.rows = self.rows[:limit]
    
    def __getitem__(self, index):

        # Load File
        r = self.rows[index]
        filename, length = r.split(',')

        # Load audio
        audio = torch.load(filename[:-3] + "pt", map_location="cpu").transpose(0, 1)

        # Pad or trim to target duration
        if audio.shape[0] >= self.segment_size:
            audio_start = random.randint(0, audio.shape[0] - self.segment_size)
            audio = audio[audio_start:audio_start+self.segment_size]
        elif audio.shape[0] < self.segment_size: # Rare or impossible case - just pad with zeros
            audio = torch.nn.functional.pad(audio, (0, 0, 0, self.segment_size - audio.shape[0]))

        # Transformer
        if self.transformer is not None:
            return self.transformer(audio)
        else:
            return audio

    def __len__(self):
        return len(self.rows)


class SpecAudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, files, segment_size, transformer = None):
        self.files = files
        self.segment_size = segment_size
        self.transformer = transformer
    
    def __getitem__(self, index):

        # Load File
        filename = self.files[index]

        # If in tensor mode
        audio = torch.load(filename)

        # Pad or trim to target duration
        if audio.shape[1] >= self.segment_size:
            audio_start = random.randint(0, audio.shape[1] - self.segment_size)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        elif audio.shape[1] < self.segment_size: # Rare or impossible case - just pad with zeros
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape[1]))

        # Transformer
        if self.transformer is not None:
            return self.transformer(audio)
        else:
            return audio

    def __len__(self):
        return len(self.files)

class PhonemesDataset(torch.utils.data.Dataset):
    def __init__(self, path, transformer, tokenizer):
        self.tokenizer = tokenizer
        self.transformer = transformer
        with open(path, 'r') as json_file:
            self.items = list(json_file)
    def __getitem__(self, index):
        data = json.loads(self.items[index])
        if self.transformer is not None:
            return self.transformer(data)
        else:
            return data
    def __len__(self):
        return len(self.items)

def load_mono_audio(src, sample_rate, device=None):

    # Load audio
    audio, sr = torchaudio.load(src)

    # Move to device
    if device is not None:
        audio = audio.to(device)

    # Resample
    if sr != sample_rate:
        audio = resampler(sr, sample_rate, device)(audio)
        sr = sample_rate

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Convert to single dimension
    audio = audio[0]

    return audio

def get_aligned_dataset_loader(names, voices, max_length, workers, batch_size, tokenizer, phoneme_duration, dtype = None):

    # Load datasets
    def load_dataset(name):
        dataset_dir = "datasets/" + name + "-aligned"
        dataset_audio_dir = "datasets/" + name + "-prepared"
        if voices is None:
            files = glob(dataset_dir + "/**/*.TextGrid")
        else:
            files = []
            for voice in voices:
                files += glob(dataset_dir + "/" + voice + "/*.TextGrid")
        files = [f[len(dataset_dir + "/"):-len(".TextGrid")] for f in files]

        # Load textgrids
        tg = [textgrid.TextGrid.fromFile(dataset_dir + "/" + f + ".TextGrid") for f in tqdm(files)]

        # Style tokens
        styles = [dataset_audio_dir + "/" + f + ".style.pt" for f in files]

        # Load audio
        files = [dataset_audio_dir + "/" + f + ".pt" for f in files]

        return tg, files, styles

    # Load all datasets
    files = []
    tg = []
    styles = []
    for name in names:
        t, f, s = load_dataset(name)
        files += f
        tg += t
        styles += s

    # Sort two lists by length together
    tg, files, styles = zip(*sorted(zip(tg, files, styles), key=lambda x: (-x[0].maxTime, x[1])))

    class AlignedDataset(torch.utils.data.Dataset):
        def __init__(self, textgrid, files, styles):
            self.files = files
            self.styles = styles
            self.textgrid = textgrid
        def __len__(self):
            return len(self.files)        
        def __getitem__(self, index):

            try:

                # Spectogram
                audio = torch.load(self.files[index])
                audio = audio.transpose(0, 1) # Reshape audio (C, T) -> (T, C)

                # Styles
                style = torch.load(self.styles[index])

                # Phonemes
                aligned_phonemes = compute_alignments(config, self.textgrid[index], style, audio.shape[0])

                # Unwrap phonemes
                phonemes = []
                styles = []
                for t in aligned_phonemes:
                    for i in range(t[1]):
                        phonemes.append(t[0])
                        styles.append(t[2])
                if len(phonemes) != audio.shape[0]:
                    raise Exception("Phonemes and audio length mismatch: " + str(len(phonemes)) + " != " + str(audio.shape[0]) + " in " + self.files[index])

                # Length
                l = len(phonemes)
                offset = 0
                if l  > max_length:
                    l = max_length
                    offset = random.randint(0, len(phonemes) - l)
        
                # Cut to size
                phonemes = phonemes[offset:offset+l]
                styles = styles[offset:offset+l]
                audio = audio[offset:offset+l]

                # Tokenize
                phonemes = tokenizer(phonemes)
                styles = torch.tensor(styles).long()

                # Cast
                if dtype is not None:
                    audio = audio.to(dtype)

                # Outputs
                return phonemes, styles, audio
            except Exception as e:
                print("Error in file: " + self.files[index])
                print(e)
                raise e

    # Create dataset
    dataset = AlignedDataset(tg, files, styles)

    def collate_to_shortest(batch):

        # Find minimum length
        min_len = min([b[0].shape[0] for b in batch])

        # Pad
        padded = []
        for b in batch:
            if b[0].shape[0] > min_len:
                offset = random.randint(0, b[0].shape[0] - min_len)
                padded.append((
                    b[0][offset:offset + min_len],
                    b[1][offset:offset + min_len],
                    b[2][offset:offset + min_len]
                ))
            else:
                padded.append((
                    b[0],
                    b[1],
                    b[2]
                ))
        return torch.stack([b[0] for b in padded]), torch.stack([b[1] for b in padded]), torch.stack([b[2] for b in padded])

    return DataLoader(dataset, num_workers=workers, shuffle=False, batch_size=batch_size, pin_memory=True, collate_fn=collate_to_shortest)

def get_aligned_dataset_dumb_loader(path, max_length, workers, batch_size, tokenizer, phoneme_duration, dtype = None):

    # Dataset
    def transformer(data):
        return torch.zeros(data.shape[0]).long(), data
    dataset = AudioFileListDataset(path, max_length, transformer)

    # Loader
    return DataLoader(dataset, num_workers=workers, shuffle=False, batch_size=batch_size, pin_memory=True)

def get_audio_spectogram_loader(path, max_length, workers, batch_size, limit = None, dtype = None):

    # Dataset
    dataset = AudioFileListDataset(path, max_length, limit=limit)

    # Loader
    return DataLoader(dataset, num_workers=workers, shuffle=False, batch_size=batch_size, pin_memory=False)


def get_phonemes_dataset(path, max_length, workers, batch_size, tokenizer, phoneme_duration, dtype = None):

    # Transform dataset
    def transformer(data):

        # Convert to phonemes and durations
        phonemes, durations = [], []
        last_time = 0
        last_silence = True
        for word in data['w']:

            # Extract data
            start = word['t'][0]
            end = word['t'][1]

            # Process word or silence
            if word['w'] is None:
                durations.append(round((end - start) / phoneme_duration))
                phonemes.append(tokenizer.silence_token)
                last_silence = True
            else:
                if not last_silence: # Add empty silence
                    durations.append(0)
                    phonemes.append(tokenizer.silence_token)
                last_silence = False
                for phone in word['p']:
                    if phone['p'] is not None:
                        phonemes.append(phone['p'])
                        durations.append(round((phone['t'][1] - phone['t'][0]) / phoneme_duration))

        # Convert to tensor
        phonemes = tokenizer(phonemes)
        durations = torch.tensor(durations)

        # Cast
        if dtype is not None:
            durations = durations.to(dtype)
        
        # Outputs
        return phonemes, durations

    # Create dataset
    dataset = PhonemesDataset(path, transformer, tokenizer)

    # Collator
    def collate_to_shortest(batch):

        # Find minimum length
        min_len = min([b[0].shape[0] for b in batch])

        # Pad
        padded = []
        for b in batch:
            if b[0].shape[0] > min_len:
                offset = random.randint(0, b[0].shape[0] - min_len)
                padded.append((
                    b[0][offset:offset + min_len],
                    b[1][offset:offset + min_len]
                ))
            else:
                padded.append((
                    b[0],
                    b[1]
                ))
        return torch.stack([b[0] for b in padded]), torch.stack([b[1] for b in padded])

    return DataLoader(dataset, num_workers=workers, shuffle=False, batch_size=batch_size, pin_memory=True, collate_fn=collate_to_shortest)