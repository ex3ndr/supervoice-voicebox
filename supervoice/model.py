import torch
import textgrid
from .model_audio import AudioPredictor
from .model_style import export_style, resolve_style
from .audio import resampler, spectogram, load_mono_audio
from .tokenizer import Tokenizer
from .tensors import drop_using_mask
from .alignment import compute_alignments
from .config import config
from .voices_gen import available_voices
import time
import os

class SuperVoice(torch.nn.Module):

    def __init__(self, gpt, vocoder):
        super(SuperVoice, self).__init__()
        self.gpt = gpt
        self.tokenizer = Tokenizer(config)
        self.audio_model = AudioPredictor(config)
        self.vocoder = vocoder

    def create_voice(self, audio, alignments, text = None, text_file = None):

        # Load text
        assert text is not None or text_file is not None, "Either text or text_file must be provided"
        assert text is None or text_file is None, "Either text or text_file must be provided, but not both"
        if text_file is not None:
            with open(text_file, 'r') as f:
                text = f.read().strip()

        # Load audio
        if type(audio) is str:
            audio = load_mono_audio(audio, sample_rate = config.audio.sample_rate)
        audio = audio.to(self._device())

        # Load alignments
        if type(alignments) is str:
            alignments = textgrid.TextGrid.fromFile(alignments)

        # Load basic prompt
        (audio_prompt, (tokens, token_styles)) = self.load_prompt(audio, alignments)

        # Calculate style
        spec = self._do_spectogram(audio)
        style = export_style(config, audio, spec)

        # Calculate alignments
        alignments = compute_alignments(config, alignments, style, spec.shape[0], adjust_style = False) # GPT expects unadjusted style tokens

        return {
            "audio": audio_prompt.cpu(),
            "audio_tokens": tokens.cpu(),
            "audio_token_styles": token_styles.cpu(),
            "text": text,
            "text_alignment": alignments
        }

    def load_prompt(self, audio, alignments = None):
        """
        Load a prompt from audio and textgrid alignments
        """
        
        # Load arguments
        if type(audio) is str:
            audio = load_mono_audio(audio, sample_rate = config.audio.sample_rate)
            audio = audio.to(self._device())
        if type(alignments) is str:
            alignments = textgrid.TextGrid.fromFile(alignments)

        # Create spectogram
        audio = audio.to(self._device())
        spec = self._do_spectogram(audio)

        # Create style
        style = export_style(config, audio, spec)

        # Calculate alignments
        if alignments is not None:
            alignments = compute_alignments(config, alignments, style, spec.shape[0])

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
        if sample_rate is not None and sample_rate != config.audio.sample_rate:
            waveform = resampler(sample_rate, config.audio.sample_rate, device)(waveform)

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
        phoneme_duration = config.audio.hop_size / config.audio.sample_rate
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

    @torch.no_grad()
    def synthesize(self, prompt, voice = None, steps = 4, alpha = 0.5, max_tokens = 256, top_k = None, pad_begin = 4, pad_end = 4):
        output = {}
        stats = {}
        output['stats'] = stats

        # Load voice if provided
        cond_audio = None
        cond_tokens = None
        cond_token_styles = None
        cond_text = None
        cond_alignments = None
        if type(voice) is torch.Tensor:
            cond_audio = voice
            cond_tokens = None
            cond_token_styles = None
        elif isinstance(voice, dict):
            cond_audio = voice['audio']
            cond_tokens = voice['audio_tokens']
            cond_token_styles = voice['audio_token_styles']
            cond_text = voice['text']
            cond_alignments = voice['text_alignment']
        elif isinstance(voice, tuple):
            (cond_audio, (cond_tokens, cond_token_styles)) = voice
        elif isinstance(voice, str):
            # Check if voice is available
            if voice not in available_voices:
                raise ValueError(f"Voice {voice} is not available")

            # Get the current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Load the voice if provided
            voice_file = os.path.join(current_dir, "..", "voices", voice + ".pt")
            voice = torch.load(voice_file, map_location = "cpu")
            cond_audio = voice['audio']
            cond_tokens = voice['audio_tokens']
            cond_token_styles = voice['audio_token_styles']
            cond_text = voice['text']
            cond_alignments = voice['text_alignment']

        # Run GPT model if string is provided
        if type(prompt) is str:

            # Run GPT
            start_time = time.time()
            gpt_conditioning = None
            if cond_text is not None and cond_alignments is not None:
                gpt_conditioning = (cond_text, cond_alignments)
            gpt_output = self.gpt.generate(prompt, conditioning = gpt_conditioning, top_k = top_k, max_new_tokens = max_tokens)
            gpt_output = gpt_output['output']
            end_time = time.time()
            execution_time = end_time - start_time
            stats['gpt_execute'] = execution_time

            # Post-process GPT output
            if pad_begin > 0:
                gpt_output = [('<SIL>', pad_begin, 0)] + gpt_output
            if pad_end > 0:
                gpt_output = gpt_output + [('<SIL>', pad_end, 0)]
            output['gpt_output'] = gpt_output
            prompt = self.load_gpt_prompt(gpt_output)
        
        # Unpack prompt
        (tokens, token_styles) = prompt
        (tokens, token_styles) = self.create_text_prompt(tokens, token_styles) # Normalize inputs
        device = self._device()
        tokens = tokens.to(device)
        token_styles = token_styles.to(device)
        if cond_audio is not None:
            cond_audio = cond_audio.to(device)
        if cond_tokens is not None:
            cond_tokens = cond_tokens.to(device)
        if cond_token_styles is not None:
            cond_token_styles = cond_token_styles.to(device)

        # Handle audio conditioning
        target_pad_begin = 1
        target_pad_end = 1
        if cond_audio is not None:

            # Unpack condition
            (cond_tokens, cond_token_styles) = self.create_text_prompt(cond_tokens, cond_token_styles, cond_audio.shape[0]) # Normalize inputs
            target_pad_begin = cond_audio.shape[0]

            # Prepare audio
            audio = torch.cat([cond_audio, torch.zeros(tokens.shape[0] + 1, config.audio.n_mels, device=device)])

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
            audio = torch.zeros((tokens.shape[0], config.audio.n_mels)).to(device)
            mask = torch.ones((tokens.shape[0])).bool().to(device)

        # Restore audio
        start_time = time.time()
        restored, _ = self.audio_model.sample(tokens = tokens, tokens_style = token_styles, audio = audio, mask = mask, steps = steps, alpha = alpha)
        restored = self._audio_denormalize(restored)
        output['melspec'] = restored
        end_time = time.time()
        execution_time = end_time - start_time
        stats['audio_model_execute'] = execution_time

        # If no vocoder: return mel-spectogram
        if self.vocoder is None:
            return output

        # Vocoder
        start_time = time.time()
        waveform = self.vocoder.generate(restored[target_pad_begin:-target_pad_end].transpose(1, 0)).squeeze(0)
        output['wav'] = waveform
        end_time = time.time()
        execution_time = end_time - start_time
        stats['vocoder_execute'] = execution_time

        # Return output
        return output

    def _do_spectogram(self, waveform):
        return spectogram(waveform, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.mel_norm, config.audio.mel_scale, config.audio.sample_rate).transpose(1, 0)
    
    def _audio_normalize(self, src):
        return (src - config.audio.norm_mean) / config.audio.norm_std

    def _audio_denormalize(self, src):
        return (src * config.audio.norm_std) + config.audio.norm_mean

    def _device(self):
        return next(self.parameters()).device

    def _load_style(self, wav, spec, durations):
        style = export_style(config, wav, spec)
        style = resolve_style(config, style, durations)
        return style