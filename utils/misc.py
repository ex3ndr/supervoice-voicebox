import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np

#
# Plotting
#

def plot_waveform(waveform, sample_rate=16000, title="Waveform", xlim=(0,5)):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)

def plot_specgram(spectrogram, title="Spectrogram"):
    _, axis = plt.subplots(1, 1)
    axis.imshow(spectrogram, cmap="viridis", vmin=-10, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.tight_layout()

#
# Utilities
#

def dict_to_object(src):
    class DictToObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = DictToObject(value)
                self.__dict__[key] = value

        def __repr__(self):
            return f"{self.__dict__}"
    return DictToObject(src)

def exists(val):
    return val is not None