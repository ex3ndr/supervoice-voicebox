# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
import multiprocessing
import glob
import torch
import torchaudio
import csv
from pathlib import Path
from tqdm import tqdm
from supervoice.audio import load_mono_audio, spectogram
from supervoice.model_style import export_style
from supervoice.config import config
from training.audio import trim_silence, improve_audio, dowload_enhancer
import pyworld as pw
import json

#
# Parameters
#

PARAM_WORKERS = max(torch.cuda.device_count() * 4, 4)

CLEANUP_SYMBOLS = [
    "\n",
    "\r",
    "\t",
    "-",
    "\"",
    # "\'" # Breaks the text with apostrophes
]

#
# Execution
#

def speaker_directory(speaker):
    return str(speaker).zfill(8)

def execute_parallel(args):
    process_id = multiprocessing.current_process()._identity[0]
    files, vad, collection_dir, aligned_dir, index = args
    file, text, speaker, alignment = files[index]
    device = "cuda:" + str(process_id % torch.cuda.device_count())

    # Format filename from index (e.g. 000001)
    target_name = str(index).zfill(8)

    # Load audio
    waveform = load_mono_audio(file, config.audio.sample_rate, device=device)

    # Trim silence
    if vad:
        waveform = trim_silence(waveform, config.audio.sample_rate)

    # Enhance audio
    # waveform = improve_audio(waveform, config.audio.sample_rate)

    # Spectogram
    spec = spectogram(waveform, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.mel_norm, config.audio.mel_scale, config.audio.sample_rate)

    # Calculate style tokens
    style = export_style(config, waveform, spec)

    # Clean up text
    for symbol in CLEANUP_SYMBOLS:
        text = text.replace(symbol, " ")
    text = " ".join(text.split()) # Remove multiple spaces
    text = text.strip()

    # Save
    target_dir = os.path.join(collection_dir, speaker_directory(speaker))
    torchaudio.save(os.path.join(target_dir, target_name + ".wav"), waveform.unsqueeze(0).cpu(), config.audio.sample_rate)
    torch.save(spec.cpu(), os.path.join(target_dir, target_name + ".pt"))
    torch.save(style.cpu(), os.path.join(target_dir, target_name + ".style.pt"))
    with open(os.path.join(target_dir, target_name + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    
    # Copy alignments if provided
    if alignment is not None:
        target_dir = os.path.join(aligned_dir, speaker_directory(speaker))
        with open(alignment, "r") as f:
            alignment = f.read()
        with open(os.path.join(target_dir, target_name + ".TextGrid"), "w", encoding="utf-8") as f:
            f.write(alignment)

def load_vctk_corpus():
    files = []
    speakers = {}

    # Load vctk corpus
    for file in glob.glob("external_datasets/vctk-corpus-0.92/**/*.flac"):
        directory, filename = os.path.split(file)
        _, speaker = os.path.split(directory)
        speaker = "vctk_" + speaker

        # Check if file is a valid audio file
        if file.endswith("_mic1.flac") or file.endswith("_mic2.flac"):

            # Load text
            base = filename[:-10]
            text_file = os.path.join(directory, base + ".txt")
            if os.path.exists(text_file):
                with open(text_file, "r") as f:
                    text = f.read()
            else:
                continue # not found

            # Extract speaker
            if speaker not in speakers:
                speakers[speaker] = len(speakers)
            speaker = speakers[speaker]

            # Append
            files.append((file, text, speaker, None))
        else:
            print("Strange filename:", filename)    

    return { 'files': files, 'speakers': speakers, 'vad': False }

def load_libritts_corpus(collections):
    files = []
    speakers = {}

    # Ignore bad samples
    ignored = {}
    ignore_files = [
        "train-clean-100_bad_sample_list.txt", 
        "train-clean-360_bad_sample_list.txt", 
        "train-other-500_bad_sample_list.txt", 
        "test-clean_bad_sample_list.txt", 
        "test-other_bad_sample_list.txt",
        "dev-clean_bad_sample_list.txt",
        "dev-other_bad_sample_list.txt"
    ]
    for file_list_file in ignore_files:
        with open("external_datasets/libritts-r/failed/" + file_list_file, "r") as f:
            for line in f:
                line = line.replace("./train-clean-100/", "external_datasets/libritts-r-clean-100/")
                line = line.replace("./train-clean-360/", "external_datasets/libritts-r-clean-360/")
                line = line.replace("./train-other-500/", "external_datasets/libritts-r-other-500/")
                line = line.replace("./test-clean/", "external_datasets/libritts-r/test-clean/")
                line = line.replace("./test-other/", "external_datasets/libritts-r/test-other/")
                line = line.replace("./dev-clean/", "external_datasets/libritts-r/dev-clean/")
                line = line.replace("./dev-other/", "external_datasets/libritts-r/dev-other/")
                ignored[line.strip()] = True

    # Load vctk corpus
    files_index = []
    for c in collections:
        files_index += glob.glob(f"external_datasets/{c}/*/*/*.wav")
    for file in files_index:
        p = Path(file)
        filename = p.name
        directory = p.parent
        speaker = "libritts_" + p.parents[1].name

        # Check if file is a valid audio file
        if file in ignored:
            # print("Ignored file: " + file)
            continue

         # Load text
        base = filename[:-4]
        text_file = os.path.join(directory, base + ".normalized.txt")
        if os.path.exists(text_file):
            with open(text_file, "r") as f:
                    text = f.read()
        else:
            print("Text not found: " + text_file)
            continue # not found

        # Extract speaker
        if speaker not in speakers:
            speakers[speaker] = len(speakers)
        speaker = speakers[speaker]

        # Append
        files.append((file, text, speaker, None))

    return { 'files': files, 'speakers': speakers, 'vad': False }

def load_common_voice_corpus(path):
    files = []
    speakers = {}

    # Load CSV
    with open(path + "/train.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # Skip header
        data = list(reader)

    # Extract files and speakers
    for row in data:
        file = path + "/clips/" + row[1]
        speaker = "cv_" + row[0]
        text = row[2]
        
        # Extract speaker
        if speaker not in speakers:
            speakers[speaker] = len(speakers)
        speaker = speakers[speaker]

        # Append
        files.append((file, text, speaker, None))

    return { 'files': files, 'speakers': speakers, 'vad': True }

def load_hifi_tts_corpus():
    files = []
    speakers = {}

    # Iterate all speakers
    for ff in Path("./external_datasets/hifi-tts").glob("*_train.json"):

        # Extract speaker
        speaker = "hifi_tts_" + ff.stem.split("_")[0]
        if speaker not in speakers:
            speakers[speaker] = len(speakers)
        speaker = speakers[speaker]

        # Read each line
        with open(ff, "r") as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                audio = "./external_datasets/hifi-tts/" + data["audio_filepath"]
                files.append((audio, text, speaker, None))

    return { 'files': files, 'speakers': speakers, 'vad': False }

def load_librilight_aligned(path):
    files = []
    speakers = {}

    # Load CSV
    ids = []
    with open(path + "files_valid.txt") as f:
        for line in f:
            ids.append(line.strip())
    
    # Process 
    for id in ids:
    
        # Speaker
        speaker = "librilight_" + id.split("/")[0]
        if speaker not in speakers:
            speakers[speaker] = len(speakers)
        speaker = speakers[speaker]

        # Audio
        audio = path + id + ".flac"

        # Extract text
        with open(path  + id + ".txt", "r") as t:
            text = t.read().strip()
                
        # Alignment
        alignment = path  + id + ".TextGrid"

        # Append
        files.append((audio, text, speaker, alignment))

    return { 'files': files, 'speakers': speakers, 'vad': False }

def execute_run():
    torch.multiprocessing.set_start_method('spawn')
    # dowload_enhancer()

    # Indexing files
    print("Build file index...")
    collections = {}
    collections['libritts'] = load_libritts_corpus(["libritts-r-clean-100", "libritts-r-clean-360"])
    collections['eval'] = load_libritts_corpus(["libritts-r/test-clean"])
    collections['vctk'] = load_vctk_corpus()
    collections['hifi-tts'] = load_hifi_tts_corpus()
    collections['librilight-small'] = load_librilight_aligned("./external_datasets/librilight-processed/")
    collections['librilight-medium'] = load_librilight_aligned("./external_datasets/librilight-medium-processed/")
    # collections['common-voice-en'] = load_common_voice_corpus("external_datasets/common-voice-16.0-en/en")
    # collections['common-voice-ru'] = load_common_voice_corpus("external_datasets/common-voice-16.0-ru/ru")
    # collections['common-voice-uk'] = load_common_voice_corpus("external_datasets/common-voice-16.0-uk/uk")

    # Process collections
    for collection in collections:
        print(f"Processing collection {collection} with {len(collections[collection]['files'])} files")
        name = collection
        files = collections[collection]['files']
        speakers = collections[collection]['speakers']
        vad = collections[collection]['vad']
        prepared_dir = "datasets/" + name + "-prepared/"
        aligned_dir = "datasets/" + name + "-aligned/"

        # Check if exists
        if Path(prepared_dir).exists():
            print(f"Collection {name} already prepared")
            continue

        # Creating directories
        for speaker in speakers:
            Path(prepared_dir + speaker_directory(speakers[speaker])).mkdir(parents=True, exist_ok=True)

        # Create alignment directories if needed
        if files[0][3] is not None:
            for speaker in speakers:
                Path(aligned_dir + speaker_directory(speakers[speaker])).mkdir(parents=True, exist_ok=True)

        # Process files
        with multiprocessing.Manager() as manager:
            files = manager.list(files)
            args_list = [(files, vad, prepared_dir, aligned_dir, i) for i in range(len(files))]
            with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
                for result in tqdm(pool.imap_unordered(execute_parallel, args_list, chunksize=32), total=len(files)):
                    pass

    # Indexing collections
    for collection in collections:
        print(f"Indexing collection {collection}...")
        ids = [str(file.with_suffix("").relative_to("./datasets/" + collection + "-prepared")) for file in Path("datasets/" + collection + "-prepared/").glob("*/*.wav")]
        ids.sort()
        with open("datasets/" + collection + ".txt", "w") as f:
            for id in ids:
                f.write(id + "\n")

    # End
    print("Done")

if __name__ == "__main__":
    execute_run()