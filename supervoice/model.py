import torch
import textgrid
from .model_audio import AudioPredictor
from .model_style import export_style, resolve_style
from .audio import resampler, spectogram, load_mono_audio
from .tokenizer import Tokenizer
from .tensors import drop_using_mask
from .alignment import compute_alignments

class SuperVoice(torch.nn.Module):

    def __init__(self, config, audio_predictor, vocoder):
        super(SuperVoice, self).__init__()
        self.config = config
        self.tokenizer = Tokenizer(config)
        self.audio_model = audio_predictor
        self.vocoder = vocoder

    def load_prompt(self, audio, alignments = None):
        """
        Load a prompt from audio and textgrid alignments
        """
        
        # Load arguments
        if type(audio) is str:
            audio = load_mono_audio(audio, sample_rate = self.config.audio.sample_rate)
            audio = audio.to(self._device())
        if type(alignments) is str:
            alignments = textgrid.TextGrid.fromFile(alignments)

        # Create spectogram
        audio = audio.to(self._device())
        spec = self._do_spectogram(audio)

        # Create style
        style = export_style(self.config, audio, spec)

        # Calculate alignments
        if alignments is not None:
            alignments = compute_alignments(self.config, alignments, style, spec.shape[0])

            # Load phonemes and styles
            phonemes = []
            styles = []
            for t in alignments:
                for i in range(t[1]):
                    phonemes.append(t[0])
                    styles.append(t[2])
            tokens = self.tokenizer(phonemes).long().to(self._device())
            styles = torch.tensor(styles, device = self._device()).long()
        else:
            tokens = None
            styles = None

        # Create text prompt
        text_prompt = self.create_text_prompt(tokens, styles, spec.shape[0])

        # Create audio prompt
        audio_prompt = self._audio_normalize(spec)

        # Return prompt
        return (audio_prompt, text_prompt)

    
    def create_text_prompt(self, tokens = None, token_styles = None, count = None):
        """
        Create a text prompt from token values and styles
        """

        # Resolve count
        C = None
        if tokens is not None:
            C = tokens.shape[0]
        if token_styles is not None:
            if C is not None and C != token_styles.shape[0]:
                raise ValueError("All inputs must have the same length")
            C = token_styles.shape[0]
        if count is not None:
            if C is not None and C != count:
                raise ValueError("All inputs must have the same length")
            C = count
        
        # Create tokens if is not provided
        if tokens is None:
            tokens = torch.zeros(C, device = self._device()).long()
        
        # Create token styles if is not provided
        if token_styles is None:
            token_styles = torch.zeros(C, device = self._device()).long()

        # Return prompt
        return (tokens, token_styles)

    def create_audio_prompt(self, waveform, sample_rate = None, tokens = None, token_styles = None):
        """
        Create a prompt from raw waveform, optionally providing tokens and token styles.
        """
        device = self._device()
        waveform = waveform.to(device)

        # Resample if needed
        if sample_rate is not None and sample_rate != self.config.audio.sample_rate:
            waveform = resampler(sample_rate, self.config.audio.sample_rate, device)(waveform)

        # Create spectogram
        spec = self._do_spectogram(waveform)
        C = spec.shape[0]

        # Create text prompt
        text_prompt = create_text_prompt(tokens, token_styles, C)

        # Create audio prompt
        audio_prompt = self._audio_normalize(spec)

        # Return prompt
        return (audio_prompt, text_prompt)

    def load_gpt_prompt(self, prompt):
        """
        Loads a GPT prompt
        """

        tokens = []
        token_styles = []
        for ph, dur, st in prompt:
            tok = self.tokenizer([ph])
            for i in range(dur):
                tokens.append(tok[0])
                token_styles.append(st)
        tokens = torch.tensor(tokens, device = self._device()).long()
        token_styles = torch.tensor(token_styles, device = self._device()).long()
        return (tokens, token_styles)


    def restore_segment(self, prompt, interval, steps = 4, alpha = None):
        """
        Restore segment of a source audio prompt
        """

        # Unpack prompt
        (audio, (tokens, token_styles)) = prompt
        seq_len = audio.shape[0]
        device = self._device()
        audio = audio.to(device)
        tokens = tokens.to(device)
        token_styles = token_styles.to(device)

        # Normalize interval
        phoneme_duration = self.config.audio.hop_size / self.config.audio.sample_rate
        start = round(interval[0] // phoneme_duration)
        end = round(interval[1] // phoneme_duration)
        start = min(max(0, start), seq_len - 1)
        end = min(max(0, end), seq_len - 1)

        # Create mask
        mask = torch.zeros((seq_len)).bool().to(device)
        mask[start:end] = True

        # Drop audio that need to be restored
        audio = drop_using_mask(audio, 0, mask)

        # Restore audio
        with torch.no_grad():
            restored, _ = self.audio_model.sample(tokens = tokens, tokens_style = token_styles, audio = audio, mask = mask, steps = steps, alpha = alpha)
        restored = self._audio_denormalize(restored)

        # Vocoder
        with torch.no_grad():
            waveform = self.vocoder.generate(restored.transpose(1, 0)).squeeze(0)

        # Return restored audio
        return waveform

    def synthesize(self, prompt, condition = None, steps = 4, alpha = 0.5):
        
        # Unpack prompt
        (tokens, token_styles) = prompt
        (tokens, token_styles) = self.create_text_prompt(tokens, token_styles) # Normalize inputs
        device = self._device()
        tokens = tokens.to(device)
        token_styles = token_styles.to(device)

        # Handle conditioning
        target_pad_begin = 1
        target_pad_end = 1
        if condition is not None:

            # If tensor
            if type(condition) is torch.Tensor:
                cond_audio = condition
                cond_tokens = None
                cond_token_styles = None
            else:
                (cond_audio, (cond_tokens, cond_token_styles)) = condition

            # Unpack condition
            (cond_tokens, cond_token_styles) = self.create_text_prompt(cond_tokens, cond_token_styles, cond_audio.shape[0]) # Normalize inputs
            target_pad_begin = cond_audio.shape[0]

            # Prepare audio
            audio = torch.cat([cond_audio, torch.zeros(tokens.shape[0] + 1, self.config.audio.n_mels, device=device)])

            # Prepare tokens
            tokens = torch.cat([cond_tokens, tokens, torch.tensor([self.tokenizer.end_token_id], device = device)])
            tokens[0] = self.tokenizer.begin_token_id

            # Prepare token styles
            token_styles = torch.cat([cond_token_styles, token_styles, torch.tensor([0], device = device)])
            token_styles[0] = 0

            # Prepare mask
            mask = torch.zeros((tokens.shape[0])).bool().to(device)
            mask[target_pad_begin:-target_pad_end] = True

        else:
            # Add begin/end tokens
            tokens = torch.cat([torch.tensor([self.tokenizer.begin_token_id], device = device), tokens, torch.tensor([self.tokenizer.end_token_id], device = device)])
            token_styles = torch.cat([torch.tensor([0], device = device), token_styles, torch.tensor([0], device = device)])

            # Create empty audio and mask
            audio = torch.zeros((tokens.shape[0], self.config.audio.n_mels)).to(device)
            mask = torch.ones((tokens.shape[0])).bool().to(device)

        # Restore audio
        with torch.no_grad():
            restored, _ = self.audio_model.sample(tokens = tokens, tokens_style = token_styles, audio = audio, mask = mask, steps = steps, alpha = alpha)
        restored = self._audio_denormalize(restored)

         # Vocoder
        with torch.no_grad():
            waveform = self.vocoder.generate(restored[target_pad_begin:-target_pad_end].transpose(1, 0)).squeeze(0)

        # Return synthesized audio
        return waveform

    def _do_spectogram(self, waveform):
        return spectogram(waveform, self.config.audio.n_fft, self.config.audio.n_mels, self.config.audio.hop_size, self.config.audio.win_size, self.config.audio.mel_norm, self.config.audio.mel_scale, self.config.audio.sample_rate).transpose(1, 0)
    
    def _audio_normalize(self, src):
        return (src - self.config.audio.norm_mean) / self.config.audio.norm_std

    def _audio_denormalize(self, src):
        return (src * self.config.audio.norm_std) + self.config.audio.norm_mean

    def _device(self):
        return next(self.parameters()).device

    def _load_style(self, wav, spec, durations):
        style = export_style(self.config, wav, spec)
        style = resolve_style(self.config, style, durations)
        return style