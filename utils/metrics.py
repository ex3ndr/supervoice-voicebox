
mos_predictor = None

def evaluate_mos(audio, sr):

    # Load
    global mos_predictor
    if mos_predictor is None:
        mos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    # Evaluate
    mos_predictor.to(audio.device)
    return mos_predictor(audio, sr)
