from glob import glob
import torch
import torchaudio
import random
import csv
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import textgrid

class SimpleAudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, files, sample_rate, segment_size, vad = False, transformer = None):
        self.files = files
        self.sample_rate = sample_rate
        self.segment_size = segment_size
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
        if audio.shape[0] >= self.segment_size:
            audio_start = random.randint(0, audio.shape[0] - self.segment_size)
            audio = audio[audio_start:audio_start+self.segment_size]
        elif audio.shape[0] < self.segment_size: # Rare or impossible case - just pad with zeros
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.shape[0]))

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


def load_common_voice_files(path, split):
    res = []
    with open(path + f'{split}.tsv') as csvfile:
        cvs_reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        next(cvs_reader, None)  # skip the headers
        return [path + 'clips/' + row[1] for row in cvs_reader]


def get_aligned_dataset_loader(names, max_length, workers, batch_size, tokenizer):

    # Load datasets
    def load_dataset(name):
        dataset_dir = "datasets/" + name + "-aligned"
        dataset_audio_dir = "datasets/" + name + "-prepared"
        files = glob(dataset_dir + "/**/*.TextGrid")
        files = [f[len(dataset_dir + "/"):-len(".TextGrid")] for f in files]

        # Load textgrids
        tg = [textgrid.TextGrid.fromFile(dataset_dir + "/" + f + ".TextGrid") for f in tqdm(files)]

        # Load audio
        files = [dataset_audio_dir + "/" + f + ".pt" for f in files]    

        return tg, files

    # Load all datasets
    files = []
    tg = []
    for name in names:
        t, f = load_dataset(name)
        files += f
        tg += t

    # Sort two lists by length together
    tg, files = zip(*sorted(zip(tg, files), key=lambda x: (-x[0].maxTime, x[1])))

    # Text grid extraction
    def extract_textgrid(src):

        # Prepare
        token_duration = 0.01
        tokens = src[1]
        time = 0
        output_tokens = []
        output_durations = []

        # Iterate over tokens
        for t in tokens:

            # Resolve durations
            ends = t.maxTime
            duration = math.floor((ends - time) / token_duration)
            time = ends

            # Resolve token
            tok = t.mark
            if tok == '':
                tok = tokenizer.silence_token
            if tok == 'spn':
                tok = tokenizer.unknown_token

            # Apply
            output_tokens.append(tok)
            output_durations.append(duration)

        # Outputs
        return output_tokens, output_durations

    class AlignedDataset(torch.utils.data.Dataset):
        def __init__(self, textgrid, files):
            self.files = files
            self.textgrid = textgrid
        def __len__(self):
            return len(self.files)        
        def __getitem__(self, index):

            # Load textgrid and audio
            tokens, durations = extract_textgrid(self.textgrid[index])
            audio = torch.load(self.files[index])
        
            # Reshape audio (C, T) -> (T, C)
            audio = audio.transpose(0, 1)

            # Phonemes
            phonemes = []
            for t in range(len(tokens)):
                tok = tokens[t]
                for i in range(durations[t]):
                    phonemes.append(tok)

            # Length
            l = len(phonemes)
            offset = 0
            if l > max_length:
                l = max_length
                offset = random.randint(0, len(phonemes) - l)
        
            # Cut to size
            phonemes = phonemes[offset:offset+l]
            audio = audio[offset:offset+l]

            # Tokenize
            phonemes = tokenizer(phonemes)

            # Outputs
            return phonemes, audio

    # Create dataset
    dataset = AlignedDataset(tg, files)

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


