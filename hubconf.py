dependencies = ['torch', 'torchaudio']

def phonemizer(gpt = None, vocoder = None):

    # Imports
    import torch
    import os
    from supervoice.model import SuperVoice

    # Load model
    checkpoint = torch.hub.load_state_dict_from_url("https://github.com/ex3ndr/supervoice/releases/download/v0.0.1/supervoice_pitch_400000.pt", map_location="cpu")
    model = SuperVoice(gpt, vocoder)
    model.audio_model.load_state_dict(checkpoint['model'])
    model.eval()

    return model
            
