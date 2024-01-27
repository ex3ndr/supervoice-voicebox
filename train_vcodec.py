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
import pandas

# Local
from train_config import config
from utils.dataset import SimpleAudioDataset, load_common_voice_files
from utils.audio import spectogram
from vcodec.model import VCodec

#
# Device and config
#

init_from = "resume" # or "scratch" or "resume"
train_batch_size = 12
train_epochs = 3100
loader_workers = 24
summary_interval = 100
save_interval = 10000
device = 'cuda:0'
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

writer = SummaryWriter(f'runs/vcodec_{config.experiment}')

#
# Dataset
#

print("Loading dataset index...")
train_files = pandas.read_pickle("datasets/cv_validated_train.pkl")[0].values.tolist()
test_files = pandas.read_pickle("datasets/cv_validated_test.pkl")[0].values.tolist()
# test_files = []
# for language in languages:
#     train_files = train_files + load_common_voice_files(language, "train")
#     test_files = test_files + load_common_voice_files(language, "test")

def transformer(audio):
    spec = spectogram(audio, config.audio.n_fft, config.audio.num_mels, config.audio.hop_size, config.audio.win_size, config.audio.sample_rate)
    return spec
    
train_dataset = SimpleAudioDataset(train_files, 
    config.audio.sample_rate, 
    config.vcodec.segment_size, 
    vad = True,
    transformer = transformer
)

test_dataset = SimpleAudioDataset(test_files, 
    config.audio.sample_rate, 
    config.vcodec.segment_size, 
    vad = True,
    transformer = transformer
)

#
# Loader
#

train_loader = DataLoader(train_dataset, num_workers=loader_workers, shuffle=True, batch_size=train_batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, num_workers=loader_workers, shuffle=True, batch_size=train_batch_size, drop_last=True)

#
# Model
#

epoch = -1
step = 0
base_model = VCodec(config).to(device)
model = base_model
if enable_compile:
    model = torch.compile(model)

#
# Optimizer
#
print(base_model)
print(list(base_model.residual_vq.parameters()))
optim = torch.optim.AdamW(base_model.parameters(), config.vcodec.learning_rate, betas=[config.vcodec.adam_b1, config.vcodec.adam_b2])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=config.vcodec.lr_decay, last_epoch=epoch)

#
# Save/Load
#

def save():
    global epoch
    torch.save({

        # Model
        'model': base_model.state_dict(), 

         # Optimizer
         'optimizer': optim.state_dict(), 
         'scheduler': scheduler.state_dict(),
         'epoch': epoch, 
         'step': step 

    },  f'./checkpoints/vcodec_{config.experiment}.pt')
    shutil.copyfile(f'./checkpoints/vcodec_{config.experiment}.pt', f'./checkpoints/vcodec_{config.experiment}_step_{step}.pt')

def load():
    global step
    global epoch

    # Load checkpoint
    checkpoint = torch.load(f'./checkpoints/vcodec_{config.experiment}.pt')

    # Model
    base_model.load_state_dict(checkpoint['model'])

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

        # Fit
        optim.zero_grad()
        reconstruction, loss, recon_loss, commitment_loss = model.forward(x)
        loss.backward()
        optim.step()

        # Advance
        step = step + 1

        # Summary
        if step % summary_interval == 0:
            writer.add_scalar("loss/total", loss, step)
            writer.add_scalar("loss/recon", recon_loss, step)
            writer.add_scalar("loss/commitment", commitment_loss, step)

        # Save
        if step % save_interval == 0:
            save()

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

    # Save on epoch end
    save()