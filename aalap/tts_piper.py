import numpy as np
from piper.config import SynthesisConfig
from piper.voice import PiperVoice


class PiperTTS:
    def __init__(self, model_path: str, length=1.0, noise=0.667, noise_w=0.8):
        self.model_path = model_path
        self.length = length
        self.noise = noise
        self.noise_w = noise_w

        cfg = self.model_path + ".json" if self.model_path.endswith(".onnx") else None
        self.voice = PiperVoice.load(self.model_path, config_path=cfg)

    def linear_resample_int16(self, x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        """Simple linear resampler to avoid extra deps. Mono int16 in -> int16 out."""
        if sr_in == sr_out:
            return x
        ratio = sr_out / sr_in
        n_out = int(round(len(x) * ratio))
        t_out = np.linspace(0, len(x) - 1, n_out)
        t_in = np.arange(len(x))
        y = np.interp(t_out, t_in, x.astype(np.float32))
        y = np.clip(y, -32768, 32767).astype(np.int16)
        return y

    def synth(self, text: str) -> np.ndarray:
        syn = SynthesisConfig(
            length_scale=self.length,
            noise_scale=self.noise,
            noise_w_scale=self.noise_w,
            normalize_audio=True,
            volume=1.0,
        )
        chunks = []
        sr = None
        for ch in self.voice.synthesize(text, syn_config=syn):
            chunks.append(ch.audio_int16_array)
            sr = ch.sample_rate
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
        if sr and sr != 16000:
            pcm = self.linear_resample_int16(pcm, sr, 16000)
        return pcm
