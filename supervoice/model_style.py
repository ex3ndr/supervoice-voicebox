import torch
import pyworld as pw

def export_style(config, waveform, spec):

    # Calculate f0
    f0, t = pw.dio(waveform.cpu().numpy().astype('double'), config.audio.sample_rate, frame_period=(1000 * config.audio.hop_size)/config.audio.sample_rate) # 1000ms * (hop/sample_rate)
    f0 = torch.tensor(f0)

    return f0

def resolve_style(config, src, durations):
    res = []
    offset = 0
    
    # Convert to log scale and normalize
    normalized_src = src
    normalized_src = torch.log(normalized_src + 1)
    normalized_src = torch.clamp(normalized_src, config.tokenizer_style.pitch_min, config.tokenizer_style.pitch_max)

    for i in range(len(durations)):

        # Calculate start and end
        start = round(offset)
        end = round(offset + durations[i])

        # Calculate pitch
        value = normalized_src[start:end].mean().item()
        value = ((value - config.tokenizer_style.pitch_min) / (config.tokenizer_style.pitch_max - config.tokenizer_style.pitch_min)) * config.tokenizer_style.tokens
        value = max(0, min(config.tokenizer_style.tokens - 1, int(value)))

        # Append
        res.append(value)

        # Update offset
        offset = end
    return res