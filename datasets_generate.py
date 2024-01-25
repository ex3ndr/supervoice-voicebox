import csv
import json
import torch
import torch.nn.functional as F
import math
from dvae.model import DiscreteVAE
from utils.audio import spectogram, load_mono_audio
from train_config import config
from tqdm import tqdm
from phonemizer import phonemize
from phonemizer.separator import Separator
from utils.alignment import init_alignment, alignment

# Config
token_split = ' \u266C ' # ♬
# token_audio_prefix = '\u2669' # ♩
# token_audio_suffix = '\u2669' # ♩
token_audio_prefix = ''
token_audio_suffix = ' '
output_file = 'gpt_dataset.jsonl'
def to_code(id):
    return f"{token_audio_prefix}{id}{token_audio_suffix}"

# Device
device = torch.device('cuda:0')
init_alignment("cuda:0")

# Model
print("Loading dVAE...")
dvae = DiscreteVAE(
    # Base mel spec parameters
    positional_dims=1,
    channels=config.audio.num_mels,

    # Number of possible tokens
    num_tokens=config.dvae.tokens,

    # Architecture
    codebook_dim=config.dvae.codebook_dim,
    hidden_dim=config.dvae.hidden_dim,
    num_resnet_blocks=config.dvae.num_resnet_blocks,
    kernel_size=config.dvae.kernel_size,
    num_layers=config.dvae.num_layers,
    use_transposed_convs=False,
).to(device)
checkpoint = "./checkpoints/dvae_dvae_3gelu_160.pt"
data = torch.load(checkpoint, map_location=device)
dvae.load_state_dict(data['dvae'])

# Loading files to process
print("Loading files...")
files = []
with open('./external_datasets/lj-speech-1.1/metadata.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='|')
    for row in spamreader:
        files.append(["./external_datasets/lj-speech-1.1/wavs/" + row[0] + ".wav", row[1]])

# Phonemes
# print("Phonemezation...")
# phonemes = []
# for file in files:
#     fname, text = file
#     phonemes.append(text)
# phonemes = phonemize(phonemes, separator=Separator(phone=None, word=' ', syllable='|'), strip=True, preserve_punctuation=True, njobs=16)

# Processing files
print("Quantization...")
quantized = []
quantized_full = []
for i in tqdm(range(len(files))):
    # phn = phonemes[i]
    fname, text = files[i]
    
    # Load audio
    waveform = load_mono_audio(fname, config.audio.sample_rate)
    spec = spectogram(waveform, config.audio.n_fft, config.audio.num_mels, config.audio.hop_size, config.audio.win_size, config.audio.sample_rate)
    codes = dvae.get_codebook_indices(spec.unsqueeze(0).to(device) / config.dvae.log_mel_multiplier).squeeze(0).cpu().tolist()
    codes = list(map(to_code, codes))

    # Append full text
    quantized_full.append([text, ''.join(codes)])

    # Append word spans
    # aligned = alignment(waveform, text, config.audio.sample_rate)
    # if aligned is not None:
    #     for j in range(len(aligned)):
    #         chars, start, end = aligned[j]

    #         # Convert to code indices
    #         start = math.floor(start * len(codes))
    #         end = min(math.ceil(end * len(codes)), len(codes))+1

    #         quantized.append([chars, ''.join(codes[start:end])])
    
# Synthesize dataset
print("Synthesizing dataset...")
with open(output_file, 'w') as f:
    for q in tqdm(quantized + quantized_full):
        text, codes = q
        f.write(json.dumps({"instruction": text, "output": codes}) + '\n')