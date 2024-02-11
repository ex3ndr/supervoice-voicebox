# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Imports
import os
import glob
import textgrid
from tqdm import tqdm
from pathlib import Path
import json

# Phonemes processor
def extract_phonemes(collection, path):
    phonemes = []

    # Load text
    with open('datasets/' + collection + "-prepared/" + path + ".txt", "r") as f:
        text = f.read()

    # Normalize
    text = text.lower().strip()

    # Load textgrid
    tg = textgrid.TextGrid.fromFile('datasets/' + collection + "-aligned/" + path + ".TextGrid")
    words = tg[0]
    phones = tg[1]
    assert words.name == "words"
    assert phones.name == "phones"

    # Process words
    output_words = []
    last_word_time = 0
    duration = words.maxTime
    time_offset = 0

    # Skip silence in the beginning
    i = 0
    while i < len(words) and words[i].mark == "":
        time_offset = -words[i].maxTime # Update offset
        last_word_time = words[i].maxTime
        i += 1

    # Process words
    for word in words:
        if word.mark == "": # Ignore empty words
            continue

        # Add silence between words
        if word.minTime != last_word_time:
            output_words.append({'t': [last_word_time + time_offset, word.minTime + time_offset], 'w': None})

        # Add word
        word_phonemes = []
        last_phone_time = 0
        for phone in phones:
            if phone.minTime != last_phone_time:
                word_phonemes.append({'t': [last_phone_time + time_offset, phone.minTime + time_offset], 'p': None})
            if phone.minTime >= word.minTime and phone.maxTime <= word.maxTime and phone.mark != "":
                word_phonemes.append({'t': [phone.minTime + time_offset, phone.maxTime + time_offset], 'p': phone.mark})
            last_phone_time = phone.maxTime

        # Processed word
        output_words.append({'t': [word.minTime + time_offset, word.maxTime + time_offset], 'w': word.mark, 'p': word_phonemes})
        last_word_time = word.maxTime

    return { 't': text, 'w': output_words, 'd': last_word_time + time_offset }

# Indexing files
print("Loading files...")
files = glob.glob("datasets/vctk-aligned/*/*.TextGrid") + glob.glob("datasets/libritts-aligned/*/*.TextGrid") + glob.glob("datasets/common-voice-en-aligned/*/*.TextGrid")

# Process files
print("Processing files...")
output = []
for file in tqdm(files):
    parts = Path(file).parts
    collection = parts[1][0:-(len('-aligned'))]
    file = parts[2] + "/" + parts[3].split(".")[0]
    output.append(extract_phonemes(collection, file))

# Sort
print("Sorting...")
output = sorted(output, key=lambda x: (-x['d'], x['t'])) # Descending by duration, deterministic by text

# Save
print("Saving...")
with open("datasets/phonemes.jsonl", "w") as f:
    for o in tqdm(output):
        json.dump(o, f)
        f.write('\n')