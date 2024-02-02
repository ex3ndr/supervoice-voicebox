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
from pathlib import Path
import math
import random
from tqdm import tqdm

# ML
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs

# Local
from train_config import config
from supervoice.model_audio import AudioPredictor
from supervoice.tokenizer import Tokenizer
from supervoice.tensors import count_parameters, probability_binary_mask, drop_using_mask, interval_mask
from utils.dataset import get_aligned_dataset_loader

# Train parameters
train_experiment = "audio_acc"
train_project="supervoice_audio"
train_auto_resume = True
train_batch_size = 8
train_grad_accum_every = 16
train_steps = 600000
train_loader_workers = 8
train_log_every = 1
train_save_every = 1000
train_evaluate_every = 200
train_evaluate_batches = 10
train_max_segment_size = 500
train_lr_start = 1e-7
train_lr_max = 1e-5
train_warmup_steps = 500

# Model parameters
enable_autocast = True
enable_compile = True
enable_detect_anomaly = False

# Train
def main():

    # Prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every)
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    accelerator.print("Loading dataset...")
    tokenizer = Tokenizer(config)
    train_loader = get_aligned_dataset_loader(names = ["libritts", "vctk"], max_length = train_max_segment_size, workers = train_loader_workers, batch_size = train_batch_size, tokenizer = tokenizer)

    # Prepare model
    accelerator.print("Loading model...")
    step = 0
    model = AudioPredictor(config)
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], train_lr_max, betas=[0.9, 0.99], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)

    # Accelerate
    model, optim, scheduler, train_loader = accelerator.prepare(model, optim, scheduler, train_loader)
    train_cycle = cycle(train_loader)
    hps = {
        "segment_size": train_max_segment_size, 
        "train_lr_start": train_lr_start, 
        "train_lr_max": train_lr_max, 
        "batch_size": train_batch_size, 
        "steps": train_steps, 
    }
    accelerator.init_trackers(train_project, config=hps)

    # Save
    def save():
        
        # Save step checkpoint
        fname = str(output_dir / f"{train_experiment}.pt")
        fname_step = str(output_dir / f"{train_experiment}.{step}.pt")
        torch.save({

            # Model
            'model': accelerator.get_state_dict(model), 

            # Optimizer
            'step': step,
            'optimizer': optim.state_dict(), 
            'scheduler': scheduler.state_dict(),

        },  f'./checkpoints/{experiment}.pt')

        # Overwrite main checkpoint
        shutil.copyfile(fname_step, fname)

    # Load
    if train_auto_resume and (output_dir / f"{train_experiment}.pt").exists():

        # Model
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model'])

        # Optimizer
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step = checkpoint['step']

        accelerator.print(f'Loaded at #{step}')
        

    # Train step
    def train_step():
        model.train()

        # Update LR
        if step < train_warmup_steps:
            lr = train_lr_start + ((train_lr_max - train_lr_start) * step) / train_warmup_steps
            for param_group in optim.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        # Load batch
        for _ in range(train_grad_accum_every):
            with accelerator.accumulate(model):
                batch = next(train_cycle)
                tokens, audio = batch
                batch_size = audio.shape[0]
                seq_len = audio.shape[1]

                # Normalize audio
                audio = (audio - config.audio.norm_mean) / config.audio.norm_std

                # Prepare Mask
                # 70% - 100% of sequence with a minimum length of 10
                # 30% rows of masking everything
                min_mask_length = min(max(10, math.floor(seq_len * 0.7)), seq_len)
                max_mask_length = seq_len
                mask = interval_mask(batch_size, seq_len, min_mask_length, max_mask_length, 0.3, device)

                # 0.2 probability of dropping everything
                conditional_drop_mask = probability_binary_mask(shape = (audio.shape[0],), true_prob = 0.2, device = device)
                audio = drop_using_mask(source = audio, replacement = 0, mask = conditional_drop_mask)
                tokens = drop_using_mask(source = tokens, replacement = tokenizer.unknown_token_id, mask = conditional_drop_mask)
                mask = drop_using_mask(source = mask, replacement = 1, mask = conditional_drop_mask)

                # Prepare CFM
                times = torch.rand((audio.shape[0],), dtype = audio.dtype, device = device)
                sigma = 0.0 # What to use here?
                t = rearrange(times, 'b -> b 1 1')
                noise = torch.randn_like(audio, device=device)
                audio_noizy = (1 - (1 - sigma) * t) * noise + t * audio
                flow = audio - (1 - sigma) * noise

                # Train step
                predicted, loss = model(
                    tokens = tokens, 
                    audio = audio, 
                    audio_noizy = audio_noizy, 
                    mask = mask, 
                    times = times, 
                    target = flow
                )

                # Check if loss is nan
                if torch.isnan(loss):
                    raise RuntimeError("Loss is NaN")

                # Backprop
                optim.zero_grad()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 0.2)
                optim.step()

        return loss, predicted, flow, mask, lr

    #
    # Start Training
    #

    accelerator.print("Training started at step", step)
    while step < train_steps:
        loss, predicted, flow, mask, lr = train_step()

        # Advance
        step = step + 1

        # Summary
        if step % train_log_every == 0:
            accelerator.log({
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
            accelerator.print(f'Step {step}: loss={loss}, lr={lr}')
        
        # Save
        if step % train_save_every == 0:
            save()

    # End training
    if accelerator.is_main_process:
        accelerator.print("Finishing training...")
        save()
    accelerator.end_training()
    accelerator.print('✨ Training complete!')

#
# Utility
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    

if __name__ == "__main__":
    main()