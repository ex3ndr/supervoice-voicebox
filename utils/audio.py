import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

#
# Cached Hann Window
#

hann_window_cache = {}
def hann_window(size, device):
    global hann_window_cache
    key = str(device) + "_" + str(size)
    if key in hann_window_cache:
        return hann_window_cache[key]
    else:
        res = torch.hann_window(size).to(device)
        hann_window_cache[key] = res
        return res

#
# Mel Log Bank
#

melscale_fbank_cache = {}
def melscale_fbanks(n_mels, n_fft, f_min, f_max, sample_rate, device):
    global melscale_fbank_cache
    key = str(n_mels) + "_" + str(n_fft) + "_" + str(f_min) + "_" + str(f_max) + "_" + str(sample_rate) + "_" + str(device)
    if key in melscale_fbank_cache:
        return melscale_fbank_cache[key]
    else:
        res = F.melscale_fbanks(
            n_freqs=int(n_fft // 2 + 1),
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm="slaney",
        ).transpose(-1, -2).to(device)
        melscale_fbank_cache[key] = res
        return res

#
# Spectogram caclulcation
#

def spectogram(audio, n_fft, n_mels, n_hop, n_window, sample_rate):

    # Hann Window
    window = hann_window(n_window, audio.device)

    # STFT
    stft = torch.stft(audio, 
        n_fft, 
        hop_length=n_hop, 
        window=window, 
        return_complex=False
    )

    # Compute magnitudes using squared value
    magnitudes = torch.sum((stft ** 2), dim=-1)[..., :-1]

    # Mel Log Bank
    mel_filters = melscale_fbanks(n_mels, n_fft, 0, sample_rate / 2, sample_rate, audio.device)
    mel_spec = (mel_filters @ magnitudes)

    # Log
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()

    return log_spec


#
# Load Mono Audio
#

def load_mono_audio(src, sample_rate):

    # Load audio
    audio, sr = torchaudio.load(src)

    # Resample
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
        sr = sample_rate

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Convert to single dimension
    audio = audio[0]

    return audio