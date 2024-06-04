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
from typing import Optional
from dataclasses import asdict, dataclass, field
from omegaconf import OmegaConf
import argparse

# ML
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import wandb

# Local
from supervoice.config import config
from supervoice.model_audio import AudioPredictor
from supervoice.tokenizer import Tokenizer
from supervoice.tensors import count_parameters, probability_binary_mask, drop_using_mask, interval_mask, length_mask
from training.dataset import create_single_sampler, create_single_sampler_balanced, create_batch_sampler, create_async_loader

@dataclass(frozen=True)
class TrainingConfig:
    name: str = "ft-01"
    datasets: list[str] = field(default_factory=list)
    source: Optional[str] = None
    steps: int = 60000
    balanced: bool = False
    mask: bool = False

# Train
def main():

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, default=Path(__file__).parent / "config_stage1.yaml")
    args = parser.parse_args()

    # Train parameters
    train_experiment = "ft-02"
    train_project="supervoice"
    train_datasets = ["libritts", "vctk"]
    train_voices = None
    train_source_experiment = None
    train_auto_resume = True
    train_grad_accum_every = 2
    train_steps = 60000
    train_loader_workers = 8
    train_log_every = 1
    train_save_every = 1000
    train_watch_every = 1000
    train_evaluate_every = 1
    train_evaluate_batch_size = 10
    train_target_duration = 6000 # 60 seconds
    train_max_segment_size = 2000 # 20 seconds
    train_lr_start = 1e-7
    train_lr_max = 2e-5
    train_warmup_steps = 5000
    train_mixed_precision = "fp16" # "bf16" or "fp16" or None
    train_clip_grad_norm = 0.2
    train_sigma = 1e-5

    # Load config
    print('Loading config ' + str(args.yaml))
    train_config = TrainingConfig(**dict(OmegaConf.merge(TrainingConfig(), OmegaConf.load(args.yaml))))
    train_experiment = train_config.name
    train_datasets = train_config.datasets
    train_steps = train_config.steps
    train_source_experiment = train_config.source

    # Prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every, mixed_precision=train_mixed_precision)
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if train_mixed_precision == "fp16" else (torch.bfloat16 if train_mixed_precision == "bf16" else torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 
    # set_seed(42)
    lr_start = train_lr_start * accelerator.num_processes
    lr_max = train_lr_max * accelerator.num_processes

    # Prepare dataset
    accelerator.print("Loading dataset...")
    tokenizer = Tokenizer(config)
    if train_config.balanced:
        base_sampler = create_single_sampler_balanced(train_datasets)
    else:
        base_sampler = create_single_sampler(train_datasets)
    sampler = create_batch_sampler(base_sampler, tokenizer, frames = train_target_duration, max_single = train_max_segment_size, dtype = dtype)
    train_loader = create_async_loader(sampler, num_workers = train_loader_workers)
    # train_loader = get_aligned_dataset_loader(names = train_datasets, voices = train_voices, max_length = train_max_segment_size, workers = train_loader_workers, batch_size = train_batch_size, tokenizer = tokenizer, phoneme_duration = phoneme_duration, dtype = dtype)

    # Prepare model
    accelerator.print("Loading model...")
    step = 0
    flow_model = torch.hub.load(repo_or_dir='ex3ndr/supervoice-flow', model='flow')
    raw_model = AudioPredictor(flow_model, config)
    model = raw_model
    # model = torch.compile(model)
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], lr_max, betas=[0.9, 0.99], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)

    # Accelerate
    model, optim = accelerator.prepare(model, optim)
    train_cycle = cycle(train_loader)
    hps = {
        "segment_size": train_max_segment_size, 
        "train_lr_start": train_lr_start, 
        "train_lr_max": train_lr_max, 
        "grad_accum_every": train_grad_accum_every,
        "steps": train_steps, 
        "warmup_steps": train_warmup_steps,
        "mixed_precision": train_mixed_precision,
        "clip_grad_norm": train_clip_grad_norm,
    }
    accelerator.init_trackers(train_project, config=hps)
    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=train_watch_every * train_grad_accum_every)

    # Save
    def save():
        
        # Save step checkpoint
        fname = str(output_dir / f"{train_experiment}.pt")
        fname_step = str(output_dir / f"{train_experiment}.{step}.pt")
        torch.save({

            # Model
            'model': raw_model.state_dict(), 

            # Optimizer
            'step': step,
            'optimizer': optim.state_dict(), 
            'scheduler': scheduler.state_dict(),

        },  fname_step)

        # Overwrite main checkpoint
        shutil.copyfile(fname_step, fname)

    # Load
    source = None
    if (output_dir / f"{train_experiment}.pt").exists():
        source = train_experiment
    elif train_source_experiment and (output_dir / f"{train_source_experiment}.pt").exists():
        source = train_source_experiment

    if train_auto_resume and source is not None:
        accelerator.print("Resuming training...")
        checkpoint = torch.load(str(output_dir / f"{source}.pt"), map_location="cpu")

        # Model
        raw_model.load_state_dict(checkpoint['model'])

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
            lr = (lr_start + ((lr_max - lr_start) * step) / train_warmup_steps)
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            lr = lr / accelerator.num_processes
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0] / accelerator.num_processes

        # Load batch
        total = 0
        last_loss = 0
        for _ in range(train_grad_accum_every):
            with accelerator.accumulate(model):
                with accelerator.autocast():

                    # Load batch
                    batch = next(train_cycle)
                    tokens, style, audio, lengths = batch
                    tokens = tokens.squeeze(0)
                    style = style.squeeze(0)
                    audio = audio.squeeze(0)
                    batch_size = audio.shape[0]
                    max_seq_len = audio.shape[1]
                    total += sum(lengths).item()

                    # Normalize audio
                    audio = (audio - config.audio.norm_mean) / config.audio.norm_std

                    # Prepare CFM
                    times = torch.rand((audio.shape[0],), dtype = audio.dtype)
                    t = rearrange(times, 'b -> b 1 1')
                    noise = torch.randn_like(audio)
                    audio_noizy = (1 - (1 - train_sigma) * t) * noise + t * audio
                    flow = audio - (1 - train_sigma) * noise

                    #
                    # Calculating masks
                    # * 0.3 probability of dropping condition
                    # * 0.15 probability of masking whole sequence
                    #        otherwise mask 70% - 100% of sequence with a minimum length of 10
                    #
                    
                    loss_mask = torch.full((batch_size, max_seq_len), False, device = "cpu", dtype = torch.bool)
                    condition_mask = torch.full((batch_size, max_seq_len), False, device = "cpu", dtype = torch.bool)
                    for i in range(batch_size):

                        if random.random() <= 0.2:

                            # Unconditioned: mask all sequence and all condition
                            loss_mask[i, 0:lengths[i]] = True
                            condition_mask[i, 0:lengths[i]] = True
                        else:

                            # By default mask everything
                            mask_length = lengths[i]
                            mask_offset = 0

                            # Reduce mask if possible
                            if random.random() > 0.2:

                                # 70% - 100% of sequence
                                min_length = max(10, math.floor(lengths[i] * 0.7))
                                max_length = lengths[i]

                                # If difference is more than 10 then calculate random mask
                                if max_length - min_length > 10:
                                    mask_length = random.randint(min_length, max_length - 10) + 10

                            # Random offset
                            if mask_length < lengths[i]:
                                mask_offset = random.randint(0, lengths[i] - mask_length)

                            # Apply mask
                            loss_mask[i, mask_offset:mask_offset + mask_length] = True

                    # 
                    # Apply masks
                    # 
                    
                    # Zero out tokens if trained with masking tokens
                    if train_config.mask:
                        condition_mask = condition_mask | ~loss_mask

                    audio = drop_using_mask(source = audio, replacement = 0, mask = loss_mask)
                    tokens = drop_using_mask(source = tokens, replacement = 0, mask = condition_mask)
                    style = drop_using_mask(source = style, replacement = 0, mask = condition_mask)

                    # Train step
                    predicted, loss = model(

                        # Condition
                        tokens = tokens.to(device), 
                        tokens_style = style.to(device),
                        audio = audio.to(device), 

                        # Noise
                        audio_noizy = audio_noizy.to(device), 

                        # Time
                        times = times.to(device), 

                        # Loss
                        mask = loss_mask.to(device), 
                        target = flow.to(device)
                    )

                    # Backprop
                    optim.zero_grad()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_clip_grad_norm)
                    optim.step()

                    # Cleanup
                    del tokens
                    del style
                    del audio
                    del audio_noizy
                    del times
                    last_loss = loss.detach().cpu().item()
                    del loss


                    # Log skipping step
                    if optim.step_was_skipped:
                        accelerator.print("Step was skipped")

        return last_loss, total, lr

    #
    # Start Training
    #

    accelerator.print("Training started at step", step)
    while step < train_steps:
        start = time.time()
        loss, total, lr = train_step()
        total = total * accelerator.num_processes # Scale to all processes
        end = time.time()

        # Advance
        step = step + 1

        # Summary
        if step % train_log_every == 0 and accelerator.is_main_process:
            speed = total / (end - start)
            accelerator.log({
                "learning_rate": lr,
                "loss": loss,
                # "predicted/mean": predicted.mean(),
                # "predicted/max": predicted.max(),
                # "predicted/min": predicted.min(),
                # "target/mean": flow.mean(),
                # "target/max": flow.max(),
                # "target/min": flow.min(),
                "data/length": total,
                "speed": speed
            }, step=step)
            accelerator.print(f'Step {step}: loss={loss}, lr={lr}, time={end - start} sec, it/s={speed}')

        # del loss
        # del predicted
        # del flow
        # del total
        # del lr
        
        # Evaluate
        # if step % train_evaluate_every == 0:
        #     accelerator.print("Evaluating...")
        #     mos = train_eval()
        #     accelerator.print(f"Step {step}: MOS={mos}")
        #     accelerator.log({"eval/mos": mos}, step=step)
        
        # Save
        if step % train_save_every == 0 and accelerator.is_main_process:
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