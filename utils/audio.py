import torch

vad = None
def trim_silence(audio, sample_rate):

    # Load VAD
    global vad
    if vad is None:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        vad = (model, utils)
    else:
        model, utils = vad
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(audio.unsqueeze(0), model.to(audio.device), sampling_rate=sample_rate)    
    if len(speech_timestamps) > 0:
        voice_start = speech_timestamps[0]['start']
        voice_end = speech_timestamps[-1]['end']
        if voice_start > 160: # Add one frame of silence
            voice_start = voice_start - 160
        if voice_end < len(audio) - 160: # Add one frame of silence
            voice_end = voice_end + 160
        audio = audio[voice_start:voice_end]

    return audio