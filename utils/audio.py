import torch
import math

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
    speech_timestamps = get_speech_timestamps(audio.unsqueeze(0), model.to(audio.device), sampling_rate=sample_rate)    
    if len(speech_timestamps) > 0:
        voice_start = speech_timestamps[0]['start'] - padding_frames
        voice_end = speech_timestamps[-1]['end'] + padding_frames
        voice_start = max(0, voice_start)
        voice_end = min(len(audio), voice_end)
        audio = audio[voice_start:voice_end]

    return audio