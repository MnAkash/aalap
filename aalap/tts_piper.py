import os
import hashlib
import urllib.request
from tqdm import tqdm
from pathlib import Path
import numpy as np
from piper.config import SynthesisConfig
from piper.voice import PiperVoice

PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="iB", unit_scale=True, desc=dest.name) if total else None
        while True:
            chunk = r.read(8192)
            if not chunk:
                break
            f.write(chunk)
            if bar:
                bar.update(len(chunk))
        if bar:
            bar.close()
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_piper_voice(language: str, voice: str, quality: str, cache_dir: Path | str | None = None,
                         expected_sha256: str | None = None) -> tuple[Path, Path]:
    """
    Download a Piper voice model (.onnx) and its config (.onnx.json) to a cache.
    Returns (model_path, config_path).
    """
    cache = Path(cache_dir) if cache_dir is not None else Path(os.path.expanduser("~/.cache/aalap/piper"))
    model_name = f"{language}-{voice}-{quality}"
    base = f"{PIPER_BASE_URL}/{language.split('_')[0]}/{language}/{voice}/{quality}/{model_name}"
    model_url = f"{base}.onnx"
    config_url = f"{base}.onnx.json"

    model_path = cache / f"{model_name}.onnx"
    config_path = cache / f"{model_name}.onnx.json"

    if not model_path.exists():
        _download_file(model_url, model_path)
    if expected_sha256:
        digest = _sha256(model_path)
        if digest.lower() != expected_sha256.lower():
            raise ValueError(f"SHA256 mismatch for {model_path}: {digest} != {expected_sha256}")
    if not config_path.exists():
        _download_file(config_url, config_path)
    return model_path, config_path


class PiperTTS:
    def __init__(
        self,
        language: str = "en_US",
        voice: str = "amy",
        quality: str = "medium",
        length: float = 1.0,
        noise: float = 0.667,
        noise_w: float = 0.8,
        cache_dir: Path | str | None = None,
    ):
        self.language = language
        self.voice_name = voice
        self.quality = quality
        self.length = length
        self.noise = noise
        self.noise_w = noise_w
        self.cache_dir = cache_dir

        model_path, cfg_path = download_piper_voice(language, voice, quality, cache_dir=cache_dir)
        self.model_path = str(model_path)
        self.config_path = str(cfg_path)
        self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)

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
