from supervoice.model import SuperVoice
from pathlib import Path
import torch

# We don't need a real model for this task
model = SuperVoice(gpt = None, vocoder = None)

# Find all wav files in the voices directory
wav_files = list(Path('voices').glob('*.wav'))
wav_files = [f.stem for f in wav_files]

# Generate voices
for id in wav_files:
    print(f"Processing {id}")
    created_voice = model.create_voice(audio = "./voices/" + id + ".wav", alignments = "./voices/" + id + ".TextGrid", text_file = "./voices/" + id + ".txt")
    torch.save(created_voice, f"./voices/{id}.pt")

# Generate index file
with open("supervoice/voices_gen.py", "w") as f:
    f.write(f"available_voices = {wav_files}")
        