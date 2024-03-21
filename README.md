# ✨ SuperVoice [BETA]
An independent VoiceBox implementation for voice synthesis. Currently in BETA.

## Features

* ⚡️ Narural sounding
* 🎤 High quality - 24khz audio
* 🤹‍♂️ Versatile - synthesiszed voice has high variability
* 📕 Currently only English language is supported, but nothing stops us from adding more languages.

## How to use

Supervoice consists of three networks: `gpt` for phoneme and prosogy generation, `audio model` for audio synthesis and `vocoder` for audio generation. Supervoice is published using Torch Hub, so you can use it as follows:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocoder
vocoder = torch.hub.load(repo_or_dir='ex3ndr/supervoice-vocoder', model='bigvsan')
vocoder.to(device)
vocoder.eval()

# GPT Model
gpt = torch.hub.load(repo_or_dir='ex3ndr/supervoice-gpt', model='phonemizer')
gpt.to(device)
gpt.eval()

# Main Model
model = torch.hub.load(repo_or_dir='ex3ndr/supervoice', model='phonemizer', gpt=gpt, vocoder=vocoder)
model.to(device)
model.eval()

# Generate audio
# Supervoice has three example voices: "voice_1", "voice_2" (my favorite), "voice_3"
# You can also remove the voice parameter to use the random one, or provide your own, but you need a TextGrid alignment for that.
# Steps means quality of the audio, recommended value is 4, 8 or 32.
# Alpha is a parameter of randomness, it should be less than 1.0, stable synthesis with small variaons is 0.1, 0.3 is a good value for more expressive synthesis, 0.5 is a maximum recommended value.
output = model.synthesize("What time is it, Steve?", voice = "voice_1", steps = 8, alpha = 0.1)

# Output of melspec
melspec = output['melspec']

# Output 1D tensor of 24000khz audio (missing if vocoder is not provided)
waveform = output['wav']

# Play audio in notebook
display(Audio(data=waveform, rate=24000))

```

## License

MIT