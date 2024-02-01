# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
import textgrid
from tqdm import tqdm
import time
from contextlib import nullcontext
import shutil
import math
import random
from tqdm import tqdm

# ML
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import pandas
import wandb
from einops import rearrange, reduce, repeat

# Local
from train_config import config
from voicebox.model_audio import AudioPredictor
from voicebox.tokenizer import Tokenizer
from utils.tensors import count_parameters, probability_binary_mask

#
# Device and config
#

experiment = "audio_libritts"
project="audio_pre"
tags = ["audio", "vctk", "libritts"]
init_from = "scratch" # or "scratch" or "resume"
train_batch_size = 64
train_steps = 600000
loader_workers = 8
summary_interval = 100
save_interval = 10000
initial_lr = 1e-5
default_lr = 1e-4
warmup_steps = 5000
device = 'cuda:1'
device_type = 'cuda' if 'cuda' in device else 'cpu'
enable_autocast = False
enable_compile = False
enable_detect_anomaly = True

#
# Precision
# 

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # Using float32 since float16 sometimes not that stable
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
autocast = nullcontext() if device_type == 'cpu' or not enable_autocast else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.set_float32_matmul_precision('high')

#
# Resuming
#

checkpoint = None
if init_from == "resume":
    checkpoint = torch.load(f'./checkpoints/{experiment}.pt')

#
# Logger
#

wandb.init(project=project, config=config, tags=tags)

#
# Dataset
#

print("Loading dataset...")

# Load index

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

files = []
tg = []
for name in ["libritts", "vctk"]:
    t, f = load_dataset(name)
    files += f
    tg += t

# Sort two lists by length together
tg, files = zip(*sorted(zip(tg, files), key=lambda x: x[0].maxTime))

# Tokenizer
tokenizer = Tokenizer(config)

# Data extractor
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

        # Apply
        output_tokens.append(tok)
        output_durations.append(duration)

    # Outputs
    return output_tokens, output_durations

class TextGridDataset(torch.utils.data.Dataset):
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

        # Normalize audio
        audio = (audio - config.audio.norm_mean) / config.audio.norm_std

        # Phonemes
        phonemes = []
        for t in range(len(tokens)):
            tok = tokens[t]
            for i in range(durations[t]):
                phonemes.append(tok)
        phonemes = tokenizer(phonemes)

        # Cut Audio
        audio = audio[:len(phonemes)]

        # Mask
        if random.uniform(0, 1) < 0.3: # If need to mask

            # How much to mask
            mask_len = random.uniform(0.7, 1)

            # Where to mask
            mask_offset = random.uniform(0, 1 - mask_len)

            # Create mask
            mask = torch.zeros(len(phonemes))
            mask_start = math.floor(mask_offset * len(phonemes))
            mask_end = math.floor((mask_offset + mask_len) * len(phonemes))
            mask[mask_start : mask_end] = 1
            mask = mask.bool()
        else:
            mask = torch.ones(len(phonemes)).bool()

        # Outputs
        return phonemes, audio, mask

#
# Dataset
#

training_dataset = TextGridDataset(tg, files)

#
# Loader
#

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
                b[2][offset:offset + min_len],
            ))
        else:
            padded.append((
                b[0],
                b[1],
                b[2],
            ))
    return torch.stack([b[0] for b in padded]), torch.stack([b[1] for b in padded]), torch.stack([b[2] for b in padded])

train_loader = DataLoader(training_dataset, num_workers=loader_workers, shuffle=True, batch_size=train_batch_size, pin_memory=True, collate_fn=collate_to_shortest)

#
# Model
#

step = 0
base_model = AudioPredictor(config).to(device)
model = base_model
if enable_compile:
    model = torch.compile(base_model)

if enable_detect_anomaly:
    def nan_hook(self, inp, output):
        if not isinstance(output, tuple):
            outputs = [output]
        else:
            outputs = output

        for i, out in enumerate(outputs):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

    for name, submodule in model.named_modules():
        submodule.register_forward_hook(nan_hook)

#
# Optimizer
#

wd_params, no_wd_params = [], []
for param in base_model.parameters():
    param_list = no_wd_params if param.ndim < 2 else wd_params
    param_list.append(param)
optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], initial_lr, betas=[0.9, 0.99],weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)

#
# Save/Load
#

def save():
    torch.save({

        # Model
        'model': base_model.state_dict(), 

         # Optimizer
         'optimizer': optim.state_dict(), 
         'scheduler': scheduler.state_dict(),
         'step': step 

    },  f'./checkpoints/{experiment}.pt')
    shutil.copyfile(f'./checkpoints/{experiment}.pt', f'./checkpoints/{experiment}_step_{step}.pt')

if checkpoint is not None:

    # Model
    base_model.load_state_dict(checkpoint['model'])

    # Optimizer
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    step = checkpoint['step']

    print(f'Loaded at #{step}')

#
# Training
#

def cycle(dl):
    while True:
        for data in dl:
            yield data

loader_cycle = cycle(train_loader)

def train_step():
    global step

    # Update LR
    if step < warmup_steps:
        lr = initial_lr + ((default_lr - initial_lr) * step) / warmup_steps
        for param_group in optim.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

    # Load batch
    batch = next(loader_cycle)
    tokens, audio, mask = batch
    tokens = tokens.to(device, non_blocking=True)
    audio = audio.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    # Prepare CFM
    times = torch.rand((audio.shape[0],), dtype = audio.dtype, device = device)
    sigma = 0.0 # What to use here?
    t = rearrange(times, 'b -> b 1 1')
    noise = torch.randn_like(audio, device=device)
    audio_noizy = (1 - (1 - sigma) * t) * noise + t * audio
    flow = audio - (1 - sigma) * noise

    # Drop tokens and audio completely
    conditional_drop_mask = probability_binary_mask(shape = (audio.shape[0],), true_prob = 0.2, device = device)
    audio = torch.where(
        rearrange(conditional_drop_mask, '... -> ... 1 1'),
        torch.zeros_like(audio, dtype = tokens.dtype, device = device),
        audio
    )
    tokens = torch.where(
        rearrange(conditional_drop_mask, '... -> ... 1'),
        torch.full(tokens.shape, tokenizer.unknown_token_id, dtype = tokens.dtype, device = device),
        tokens
    )
    mask = torch.where(
        rearrange(conditional_drop_mask, '... -> ... 1'),
        torch.ones_like(mask), # Make mask all ones if we drop
        mask
    )

    # Train step
    with torch.autograd.detect_anomaly() if enable_detect_anomaly else nullcontext():

        # Forward
        with autocast:
            predicted, loss = model(
                tokens = tokens, 
                audio = audio, 
                audio_noizy = audio_noizy, 
                mask = mask, 
                times = times, 
                target = flow
            )

        # Backprop
        optim.zero_grad()   
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
        optim.step()

    # Advance
    step = step + 1

    # Summary
    if step % summary_interval == 0:
        wandb.log({
            "learning_rate": lr,
            "loss": loss,
            "predicted/mean": predicted.mean(),
            "predicted/max": predicted.max(),
            "predicted/min": predicted.min(),
            "target/mean": flow.mean(),
            "target/max": flow.max(),
            "target/min": flow.min(),
            "data/length": mask.shape[1],
        }, step=step)
        print(f'Step {step}: loss={loss}, lr={lr}')
        
    # Save
    if step % save_interval == 0:
        save()

#
# Start Training
#

print(f'Training {experiment} on {device} with {dtype} precision')
print(f'Parameters: {count_parameters(model)}')
while step < train_steps:
    train_step()