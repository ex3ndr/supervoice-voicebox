# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
from tqdm import tqdm
import time
from contextlib import nullcontext
import shutil

# ML
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local
from train_config import config
from utils.dataset import SimpleAudioDataset
from utils.audio import spectogram
from dvae.model import DiscreteVAE

#
# Device and config
#

init_from = "scratch" # or "scratch" or "resume"
train_batch_size = 16
train_epochs = 3100
loader_workers = 16
summary_interval = 100
save_interval = 1
device = 'cuda:1'
device_type = 'cuda' if 'cuda' in device else 'cpu'
enable_autocast = False
enable_compile = False

#
# Precision
# 

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # Using float32 since float16 sometimes not that stable
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
autocast = nullcontext() if device_type == 'cpu' or not enable_autocast else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.set_float32_matmul_precision('high')

#
# Logger
#

writer = SummaryWriter(f'runs/dvae_{config.experiment}')

#
# Dataset
#

def transformer(audio):
    return spectogram(audio, config.audio.n_fft, config.audio.num_mels, config.audio.hop_size, config.audio.win_size, config.audio.sample_rate)
    
training_dataset = SimpleAudioDataset(glob("external_datasets/lj-speech-1.1/wavs/*.wav") + glob("external_datasets/vctk-corpus-0.92/**/*.flac"), config.audio.sample_rate, config.dvae.segment_size, transformer = transformer)

#
# Loader
#

train_loader = DataLoader(training_dataset, num_workers=loader_workers,  shuffle=False, batch_size=train_batch_size, pin_memory=True, drop_last=True)

#
# Model
#

epoch = -1
step = 0
base_dvae = DiscreteVAE(

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
dvae = base_dvae

#
# Optimizer
#

optim = torch.optim.AdamW(dvae.parameters(), config.dvae.learning_rate, betas=[config.dvae.adam_b1, config.dvae.adam_b2])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=config.dvae.lr_decay, last_epoch=epoch)

#
# Save/Load
#

def save():
    global epoch
    torch.save({

        # Model
        'dvae': base_dvae.state_dict(), 

         # Optimizer
         'optimizer': optim.state_dict(), 
         'scheduler': scheduler.state_dict(),
         'epoch': epoch, 
         'step': step 

    },  f'./checkpoints/dvae_{config.experiment}.pt')
    shutil.copyfile(f'./checkpoints/dvae_{config.experiment}.pt', f'./checkpoints/dvae_{config.experiment}_{epoch}.pt')

def load():
    global step
    global epoch

    # Load checkpoint
    checkpoint = torch.load(f'./checkpoints/dvae_{config.experiment}.pt')

    # Model
    base_dvae.load_state_dict(checkpoint['dvae'])

    # Optimizer
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    print(f'Loaded at #{epoch}/{step}')

# Do load if needed
if init_from == "resume":
    load()

#
# Training
#

def train_epoch():
    global step
    global epoch

    # Iterate each batch
    for i, x in enumerate(train_loader):
        
        # Load batch and move to GPU
        x = x.to(device, non_blocking=True)

        # Forward pass
        optim.zero_grad()
        recon_loss, commitment_loss, out = dvae.forward(x)
        # print(recon_loss.shape, commitment_loss.shape, out.shape)

        # Backward pass
        # loss = recon_loss + commitment_loss
        # loss = recon_loss.mean() + commitment_loss
        loss = recon_loss.mean()
        loss.backward()
        optim.step()

        # Advance
        step = step + 1

        # Summary
        if step % summary_interval == 0:
            writer.add_scalar("loss/recon", recon_loss.mean(), step)
            writer.add_scalar("loss/commitment", commitment_loss.mean(), step)

    # Advance
    epoch = epoch + 1
    scheduler.step()


#
# Start Training
#

print(f'Training {config.experiment} on {device} with {dtype} precision')
while epoch < train_epochs:

    # Train
    start = time.perf_counter()
    training_loss = train_epoch()
    duration = round((time.perf_counter() - start) * 1000)

    # Stats
    print(f'#{epoch} in {duration} ms')

    # Save
    if epoch % save_interval == 0:
        save()