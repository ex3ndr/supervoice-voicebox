from .model_style import resolve_style

def normalize_continious_phonemes(src):
    """
    Normalizing and fixing holes in phonemes
    """
    res = []
    time = 0
    for t in src:
        tok = t[0]
        start = t[1]
        end = t[2]
        if start != time:
            res.append(('<SIL>', time, start))
        res.append(t)
        time = end
    return res


def extract_textgrid_alignments(tg):
    """
    Converts a TextGrid object to a list of tuples (phoneme, start, end)
    """
    output = []
    for t in tg[1]:
        ends = t.maxTime
        tok = t.mark
        if tok == '': # Ignore spaces
            continue
        if tok == 'spn':
            tok = '<UNK>'
        output.append((tok, t.minTime, t.maxTime))
    return output

def quantisize_phoneme_positions(src, phoneme_duration):
    """
    Quantisize phoneme positions, according to the single token duration
    """
    res = []
    for t in src:
        tok = t[0]
        # NOTE: We are expecting src to be normalized and start and end to match in adjacent tokens
        start = int(t[1] // phoneme_duration)
        end = int(t[2] // phoneme_duration)
        res.append((tok, start, end))
    return res


def continious_phonemes_to_discreete(raw_phonemes, phoneme_duration):
    """
    Convert continious phonemes to a list of integer intervals
    """

    # Normalize: add silence between intervals,
    #            ensure that start of any token is equal to end of a previous,
    #            ensure that first token is zero
    raw_phonemes = normalize_continious_phonemes(raw_phonemes)

    # Quantisize offsets: convert from real one to a discreete one
    quantisized = quantisize_phoneme_positions(raw_phonemes, phoneme_duration)

    # Convert to intervals
    intervals = [(i[0], i[2] - i[1]) for i in quantisized]

    return intervals


def compute_alignments(config, tg, style, total_duration, adjust_style=True):
    """
    Compute alignments from TextGrid object and style tensor
    """

    phoneme_duration = config.audio.hop_size / config.audio.sample_rate

    # Extract alignments
    x = extract_textgrid_alignments(tg)

    # Convert to discreete
    x = continious_phonemes_to_discreete(x, phoneme_duration)

    # Trim empty
    x = [i for i in x if i[1] > 0]

    # Pad with silence
    total_length = sum([i[1] for i in x])
    assert total_length <= total_duration # We don't have reversed case in our datasets
    if total_length < total_duration: # Pad with silence because textgrid is usually shorter
        x += [(config.tokenizer.silence_token, total_duration - total_length)]

    # Style tokens
    y = resolve_style(config, style, [i[1] for i in x])
    x = [(xi[0], xi[1], yi + 1 if adjust_style else yi) for xi, yi in zip(x, y)]

    return x