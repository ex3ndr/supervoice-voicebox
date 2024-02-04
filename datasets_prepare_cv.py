# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import csv
import os
import multiprocessing
import glob
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from utils.dataset import load_common_voice_files
from supervoice.audio import load_mono_audio, spectogram
from utils.audio import trim_silence
from train_config import config
import torchaudio
import os

#
# Parameters
#

PARAM_WORKERS = 4

#
# Execution
#

def speaker_directory(speaker):
    return str(speaker).zfill(8)

def execute_parallel(args):
    files, index, collection, lang = args
    file, text, speaker = files[index]
    device = "cuda:" + str(index % 2)

    # Format filename from index (e.g. 000001)
    target_name = str(index).zfill(8)

    # Load audio
    waveform = load_mono_audio(file, 16000, device=device)

    # Trim silence
    waveform = trim_silence(waveform, 16000)

    # Spectogram
    spec = spectogram(waveform, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.sample_rate)

    # Save
    target_dir = os.path.join("datasets", "cv-" + lang + "-prepared", collection, speaker_directory(speaker))
    torchaudio.save(os.path.join(target_dir, target_name + ".wav"), waveform.unsqueeze(0).cpu(), 16000)
    torch.save(spec.cpu(), os.path.join(target_dir, target_name + ".pt"))
    with open(os.path.join(target_dir, target_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    

def execute_run(collection, lang):

    # Indexing files
    print("Starting processing " + collection + " for " + lang + "...")
    files = []
    speakers = {}
    with open("external_datasets/common-voice-16.0-" + lang + "/" + lang + "/" + collection + '.tsv', 'r') as file:
        
        # Open file
        tsv_reader = csv.reader(file, delimiter='\t')
        next(tsv_reader)

        # Loop through the rows
        for row in tsv_reader:

            # Speaker ID
            speaker = 'cv_' + collection + '_'+ lang + '_'+ row[0]

            # Path
            file = "external_datasets/common-voice-16.0-" + lang + "/" + lang + "/clips/" + row[1]

            # Text
            text = row[2]

            # Extract speaker
            if speaker not in speakers:
                speakers[speaker] = len(speakers)
            speaker = speakers[speaker]

            # Append
            files.append((file, text, speaker))

    # Results
    print("Files: " + str(len(files)))

    # Creating directories
    for speaker in speakers:
        Path("datasets/cv-" + lang + "-prepared/" + collection + "/" + speaker_directory(speakers[speaker])).mkdir(parents=True, exist_ok=True)

    # Indexes loop
    print("Preparing files...")
    with multiprocessing.Manager() as manager:
        files = manager.list(files)
        args_list = [(files, i, collection, lang) for i in range(len(files))]
        with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
            for result in tqdm(pool.imap_unordered(execute_parallel, args_list, chunksize=32), total=len(files)):
                pass

if __name__ == "__main__":
    for collection in ['train', 'test']:
        execute_run(collection, "en")