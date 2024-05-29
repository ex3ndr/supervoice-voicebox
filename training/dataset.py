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

def create_single_sampler(datasets):

    # Load datasets
    def load_dataset(name):

        # Load ids
        dataset_dir = "datasets/" + name + "-aligned"
        dataset_audio_dir = "datasets/" + name + "-prepared"
        with open("datasets/" + name + ".txt", "r") as f:
            ids = f.read().splitlines()

        # Load textgrids
        tg = [dataset_dir + "/" + f + ".TextGrid" for f in ids]

        # Style tokens
        styles = [dataset_audio_dir + "/" + f + ".style.pt" for f in ids]

        # Load audio
        files = [dataset_audio_dir + "/" + f + ".pt" for f in ids]

        return tg, files, styles

    # Load all datasets
    files = []
    tg = []
    styles = []
    for name in datasets:
        t, f, s = load_dataset(name)
        files += f
        tg += t
        styles += s

    # Sampler
    def sample():

        # Random file
        index = random.randint(0, len(files) - 1)

        # Load spectogram
        audio = torch.load(files[index])
        audio = audio.transpose(0, 1) # Reshape audio (C, T) -> (T, C)

        # Load style
        style = torch.load(styles[index])

        # Load textgrid
        ttg = textgrid.TextGrid.fromFile(tg[index])

        return (audio, style, ttg)

    return sample

def create_batch_sampler(base_sampler, tokenizer, *, frames, dtype = None):

    def fetch_single():
        
        # Sample
        audio, style, textgrid = base_sampler()

        # Alignment
        aligned_phonemes = compute_alignments(config, textgrid, style, audio.shape[0])

        # Unwrap phonemes
        phonemes = []
        styles = []
        for t in aligned_phonemes:
            for i in range(t[1]):
                phonemes.append(t[0])
                styles.append(t[2])
        assert len(phonemes) == audio.shape[0]

        # Prepare tensors
        phonemes = tokenizer(phonemes)
        styles = torch.tensor(styles).long()

        # Cast
        if dtype is not None:
            audio = audio.to(dtype)

        return phonemes, styles, audio


    def sample():
        max_length = 0
        phonemes, styles, audio = [], [], []

        # Find first sample that fits
        p, s, a = fetch_single()
        if a.shape[0] <= frames:
            phonemes.append(p)
            styles.append(s)
            audio.append(a)
            max_length = a.shape[0]
        else:
            # If the first sample is too long, return it truncated (should never happen thought)
            return p.unsqueeze(0), s.unsqueeze(0), a.unsqueeze(0), [a.shape[0]]

        # Collect additional samples
        while (len(phonemes) + 1) * max_length <= frames:

            # Sample element
            p, s, a = fetch_single()
            sample_frames = a.shape[0]
            new_max = max(max_length, sample_frames)

            # Check for overflow
            if (len(phonemes) + 1) * new_max > frames:
                break

            # Append
            max_length = new_max
            phonemes.append(p)
            styles.append(s)
            audio.append(a)

        # Padding to max_length
        phonemes_padded = []
        styles_padded = []
        audio_padded = []
        lengths = []
        for i in range(len(phonemes)):
            lengths.append(phonemes[i].shape[0])
            if (audio[i].shape[0] < max_length):
                phonemes_padded.append(torch.nn.functional.pad(phonemes[i], (0, max_length - phonemes[i].shape[0]), "constant", 0))
                styles_padded.append(torch.nn.functional.pad(styles[i], (0, max_length - styles[i].shape[0]), "constant", 0))
                audio_padded.append(torch.nn.functional.pad(audio[i], (0, 0, 0, max_length - audio[i].shape[0]), "constant", 0))
            else:
                phonemes_padded.append(phonemes[i])
                styles_padded.append(styles[i])
                audio_padded.append(audio[i])

        # Concatenate
        phonemes = torch.stack(phonemes_padded, dim=0)
        styles = torch.stack(styles_padded, dim=0)
        audio = torch.stack(audio_padded, dim=0)
        
        return phonemes, styles, audio, lengths

    return sample

def create_async_loader(sampler, num_workers = 1):

    # Dataset
    class AsyncDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = AsyncDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader