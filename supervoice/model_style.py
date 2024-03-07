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

    # Make continuous and normalize
    src = _convert_to_continuous_f0(src)
    src = (src - torch.mean(src))
    src_std = torch.std(src)
    if src_std != 0: # If std is zero, we will get NaNs
        src = src / torch.std(src)
    
    # Convert to log scale and normalize
    src = torch.clamp(src, config.tokenizer_style.pitch_min, config.tokenizer_style.pitch_max)

    for i in range(len(durations)):

        # Calculate start and end
        start = round(offset)
        end = round(offset + durations[i])

        # Calculate pitch
        value = src[start:end].mean().item()
        value = ((value - config.tokenizer_style.pitch_min) / (config.tokenizer_style.pitch_max - config.tokenizer_style.pitch_min)) * config.tokenizer_style.tokens
        value = max(0, min(config.tokenizer_style.tokens - 1, int(value)))

        # Append
        res.append(value)

        # Update offset
        offset = end
    return res


def _convert_to_continuous_f0(f0):
    nonzero_f0 = f0[f0 != 0]
    if len(nonzero_f0) > 0:  # Check if there are any non-zero values
        start_f0 = nonzero_f0[0]
        end_f0 = nonzero_f0[-1]
        start_idx = (f0 == start_f0).nonzero(as_tuple=True)[0][0]
        end_idx = (f0 == end_f0).nonzero(as_tuple=True)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx + 1:] = end_f0  # Adding 1 to end_idx to include the end value itself

        # get non-zero frame index and corresponding f0 values
        nonzero_idxs = (f0 != 0).nonzero(as_tuple=True)[0]
        nonzero_f0_values = f0[nonzero_idxs]

        # perform linear interpolation
        interp_f0 = torch.zeros_like(f0)
        for i in range(len(nonzero_idxs) - 1):
            start_idx, end_idx = nonzero_idxs[i], nonzero_idxs[i + 1]
            interp_values = torch.linspace(nonzero_f0_values[i], nonzero_f0_values[i + 1], steps=end_idx - start_idx + 1)
            interp_f0[start_idx:end_idx + 1] = interp_values

        return interp_f0
    else:
        return f0