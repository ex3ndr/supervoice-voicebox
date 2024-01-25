import torch
import torchaudio
from uroman import uroman
import re
from torchaudio.pipelines import MMS_FA as bundle

# Init
model = None
device = None
def init_alignment(d):
    global model
    global device
    device = d
    model = bundle.get_model().to(d)

# Normalize text
def prepare_text(text):
    text = uroman(text)
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def alignment(waveform, text, sample_rate):

    # Prepare
    text = prepare_text(text).split()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    # Check
    if len(text) == 0:
        return None

    # Align
    with torch.inference_mode():
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, bundle.sample_rate)(waveform)
        emission, _ = model(waveform.unsqueeze(0).to(device))
        token_spans = aligner(emission[0], tokenizer(text))
        ratio = 1 / emission.shape[1]
    
    # Convert to word spans
    output = []
    for t_spans, chars in zip(token_spans, text):
        output.append((chars, t_spans[0].start * ratio, t_spans[-1].end * ratio))

    return output