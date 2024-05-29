import torch
import math
import os
import subprocess
from pathlib import Path
from resemble_enhance.enhancer.inference import enhance
from resemble_enhance.enhancer.download import download
from supervoice.audio import resampler

vad = None

def init_if_needed():
    global vad
    if vad is None:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        vad = (model, utils)
    else:
        model, utils = vad
        
    return vad

def trim_silence(audio, sample_rate, padding = 0.25):

    # Load VAD
    model, utils = init_if_needed()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Get speech timestamps
    padding_frames = math.floor(sample_rate * padding)
    source_audio = audio
    scale = 1.0
    if sample_rate != 16000:
        source_audio = resampler(sample_rate, 16000, device = audio.device)(source_audio)
        scale = sample_rate / 16000
    speech_timestamps = get_speech_timestamps(source_audio.unsqueeze(0), model.to(audio.device), sampling_rate=16000)

    # Trim silence
    if len(speech_timestamps) > 0:
        voice_start = round(speech_timestamps[0]['start'] * scale) - padding_frames
        voice_end = round(speech_timestamps[-1]['end'] * scale) + padding_frames
        voice_start = max(0, voice_start)
        voice_end = min(len(audio), voice_end)
        audio = audio[voice_start:voice_end]

    return audio

def run_command(command, msg=None, env={}):
    try:
        subprocess.run(command, check=True, env={**os.environ, **env})
    except subprocess.CalledProcessError as e:
        if msg is not None:
            raise RuntimeError(msg) from e
        raise e

def dowload_enhancer():

    # Create dir if not exists
    REPO_DIR = Path(__file__).parent.parent / ".enhancer"
    if not REPO_DIR.exists():
        REPO_DIR.mkdir()
        run_command(["git", "clone", "https://huggingface.co/ResembleAI/resemble-enhance", str(REPO_DIR)], "Failed to clone the repository, please try again.", {"GIT_LFS_SKIP_SMUDGE": "1"})
        run_command(["git", "-C", str(REPO_DIR), "lfs", "pull"], "Failed to pull latest changes, please try again.")

    return REPO_DIR / "enhancer_stage2"


def improve_audio(audio, sample_rate):    
    run_dir = dowload_enhancer()
    res, new_sr = enhance(audio.cpu(), sample_rate, audio.device, nfe=24, solver="midpoint", lambd=0.1, tau=0.5, run_dir = run_dir) # lambd = 0.9 if denoise and then enhance
    res = res.to(audio.device)
    if new_sr != sample_rate:
        resampler_ = resampler(new_sr, sample_rate, audio.device)
        res = resampler_(res.unsqueeze(0)).squeeze(0)
    return res