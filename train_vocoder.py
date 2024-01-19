# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
from tqdm import tqdm
import time
from contextlib import nullcontext

# ML
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local
from vocoder.model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils.dataset import SimpleAudioDataset
from utils.audio import spectogram
from train_config import config

#
# Device and config
#

init_from = "scratch" # or "scratch" or "resume"
train_batch_size = 16
train_epochs = 180
loader_workers = 4
summary_interval = 100
save_interval = 1
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

writer = SummaryWriter(f'runs/vocoder_{config.experiment}')

#
# Dataset
#

def transformer(audio):
    mel = spectogram(audio, config.audio.n_fft, config.audio.num_mels, config.audio.hop_size, config.audio.win_size, config.audio.sample_rate)
    return mel, audio
    
training_dataset = SimpleAudioDataset(glob("external_datasets/lj-speech-1.1/wavs/*.wav"), config.audio.sample_rate, config.vocoder.segment_size, transformer = transformer)

#
# Loader
#

train_loader = DataLoader(training_dataset, num_workers=loader_workers,  shuffle=False, batch_size=train_batch_size, pin_memory=True, drop_last=True)

#
# Model
#

epoch = -1
step = 0
base_generator = Generator(config).to(device)
base_mpd = MultiPeriodDiscriminator().to(device)
base_msd = MultiScaleDiscriminator().to(device)
generator = base_generator
mpd = base_mpd
msd = base_msd
if enable_compile:
    generator = torch.compile(generator)
    mpd = torch.compile(mpd)
    msd = torch.compile(msd)

#
# Optimizer
#

optim_g = torch.optim.AdamW(generator.parameters(), config.vocoder.learning_rate, betas=[config.vocoder.adam_b1, config.vocoder.adam_b2])
optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), config.vocoder.learning_rate, betas=[config.vocoder.adam_b1, config.vocoder.adam_b2])
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.vocoder.lr_decay, last_epoch=epoch)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.vocoder.lr_decay, last_epoch=epoch)

#
# Save/Load
#

def save():
    torch.save({

        # Model
        'generator': base_generator.state_dict(), 
        "mpd": base_mpd.state_dict(),
        "msd": base_msd.state_dict(),

         # Optimizer
         'optimizer_g': optim_g.state_dict(), 
         'optimizer_d': optim_d.state_dict(), 
         'scheduler_g': scheduler_g.state_dict(),
         'scheduler_d': scheduler_d.state_dict(),
         'epoch': epoch, 
         'step': step 

    },  f'./checkpoints/vocoder_{experiment}.pt')

def load():
    global step
    global epoch

    # Load checkpoint
    checkpoint = torch.load(f'./checkpoints/vocoder_{experiment}.pt')

    # Model
    base_generator.load_state_dict(checkpoint['generator'])
    base_mpd.load_state_dict(checkpoint['mpd'])
    base_msd.load_state_dict(checkpoint['msd'])

    # Optimizer
    optim_g.load_state_dict(checkpoint['optimizer_g'])
    optim_d.load_state_dict(checkpoint['optimizer_d'])
    scheduler_g.load_state_dict(checkpoint['scheduler_g'])
    scheduler_d.load_state_dict(checkpoint['scheduler_d'])
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
    for i, batch in enumerate(train_loader):

        # Load batch and move to GPU
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y = y.unsqueeze(1) # Adding a channel dimension

        # Generate
        y_g_hat = generator(x)
        y_g_hat_mel = spectogram(y_g_hat.squeeze(1), config.audio.n_fft, config.audio.num_mels, config.audio.hop_size, config.audio.win_size, config.audio.sample_rate)

        #
        # Discriminator Optimisation
        #
    
        optim_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        loss_disc_all.backward()
        optim_d.step()

        #
        # Generator Optimization
        #

        optim_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(x, y_g_hat_mel) * 45

        # Discriminator-based losses
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        loss_gen_all.backward()
        optim_g.step()

        # Advance
        step = step + 1

        # Summary
        if step % summary_interval == 0:
            writer.add_scalar("loss/mpd", loss_disc_f, step)
            writer.add_scalar("loss/msd", loss_disc_s, step)
            writer.add_scalar("loss/generator_all", loss_gen_all, step)
            writer.add_scalar("loss/generator_mel", loss_mel, step)
        
    # Advance
    epoch = epoch + 1
    scheduler_g.step()
    scheduler_d.step()


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