# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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

# ML
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import pandas
import wandb
from einops import rearrange, reduce, repeat

# Local
from train_config import config
from voicebox.model_duration import DurationPredictor
from voicebox.tokenizer import Tokenizer
from utils.tensors import count_parameters, probability_binary_mask

#
# Device and config
#

project="duration"
experiment = "duration_pre"
tags = ["duration", "vctk", "libritts"]
init_from = "resume" # or "scratch" or "resume"
train_batch_size = 64
train_steps = 600000
loader_workers = 4
summary_interval = 100
save_interval = 10000
initial_lr = 1e-6
default_lr = 1e-4
warmup_steps = 50000
device = 'cuda:0'
device_type = 'cuda' if 'cuda' in device else 'cpu'
enable_autocast = False
enable_compile = True
enable_detect_anomaly = False

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

# writer = SummaryWriter(f'runs/{experiment}')
wandb.init(project=project, config=config, tags=tags)

#
# Dataset
#

print("Loading dataset...")
files = glob("datasets/vctk-aligned/**/*.TextGrid") + glob("datasets/libritts-aligned/**/*.TextGrid")
# files = glob("datasets/vctk-aligned/**/*.TextGrid")[0:1000]
files = [textgrid.TextGrid.fromFile(f) for f in tqdm(files)]

# Tokenizer
tokenizer = Tokenizer(config)

# Data extractor
def extract_data(src):

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
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)        
    def __getitem__(self, index):
        tg = self.files[index]

        # Load tokens/durations
        tokens, durations = extract_data(tg)
        tokens = tokenizer(tokens)
        durations = torch.Tensor(durations)

        # Calculate mask
        if random.uniform(0, 1) < 0.2: # If need to mask

            # How much to mask
            mask_len = random.uniform(0.1, 1)

            # Where to mask
            mask_offset = random.uniform(0, 1 - mask_len)

            # Create mask
            mask = torch.zeros(len(durations))
            mask_start = math.floor(mask_offset * len(durations))
            mask_end = math.floor((mask_offset + mask_len) * len(durations))
            mask[mask_start : mask_end] = 1
            mask = mask.bool()
        else:
            mask = torch.ones(len(durations)).bool() # Mask everything

        # Result
        return tokens, durations, mask

training_dataset = TextGridDataset(files)

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

train_loader = DataLoader(training_dataset, num_workers=loader_workers, shuffle=False, batch_size=train_batch_size, pin_memory=True, collate_fn=collate_to_shortest)

#
# Model
#

step = 0
base_model = DurationPredictor(config).to(device)
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
    tokens, durations, mask = batch
    tokens = tokens.to(device, non_blocking=True)
    durations = durations.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    # Mask all for small sequences
    if tokens.shape[1] < 5:
        mask = torch.ones_like(mask).bool()
    else:
        # Mask everything completely sometimes
        conditional_drop_mask = probability_binary_mask(shape = (durations.shape[0],), true_prob = 0.3, device = device)
        durations = torch.where(
            rearrange(conditional_drop_mask, '... -> ... 1'),
            torch.zeros_like(durations),
            durations
        )
        mask = torch.where(
            rearrange(conditional_drop_mask, '... -> ... 1'),
            torch.zeros_like(mask),
            mask
        )

    # Train step
    with torch.autograd.detect_anomaly() if enable_detect_anomaly else nullcontext():

        # Forward
        with autocast:
            predicted, loss = model(
                tokens = tokens, 
                durations = durations, 
                mask = mask, 
                target = durations
            )
        predicted = predicted.float()
        durations = durations.float()

        # Check if loss is nan
        if torch.isnan(loss):
            raise RuntimeError("Loss is NaN")

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
            "target/mean": durations.mean(),
            "target/max": durations.max(),
            "target/min": durations.min(),
            "data/length": mask.shape[1],
        }, step=step)
        print(f'Step {step} | Loss: {loss} | LR: {lr}')
        
    # Save
    if step % save_interval == 0:
        save()

#
# Start Training
#

print(f'Training {experiment} on {device} with {dtype} precision')
while step < train_steps:
    train_step()