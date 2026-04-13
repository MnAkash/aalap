from collections import deque

import numpy as np

DEFAULT_SAMPLE_RATE = 16000


def to_float32(pcm16: np.ndarray) -> np.ndarray:
    return pcm16.astype(np.float32) / 32768.0


class VAD:
    "Voice Activity Detection with Silero."

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        silero_threshold: float = 0.5,
        silero_window_ms: int = 320,
        silero_min_speech_ms: int = 60,
        silero_min_silence_ms: int = 0,
    ):
        """
        Args:
            sample_rate: Input sample rate in Hz. Silero expects 16 kHz mono PCM16.
            silero_threshold: Silero speech probability threshold (0..1). Lower = more sensitive.
            silero_window_ms: Rolling window size Silero sees for a decision (ms). Longer can improve stability.
            silero_min_speech_ms: Minimum speech duration for Silero to count a segment (ms).
            silero_min_silence_ms: Minimum silence duration between segments for Silero (ms).
        """
        self.sample_rate = sample_rate
        self.silero_threshold = silero_threshold
        self.silero_window_samples = max(1, (sample_rate * silero_window_ms) // 1000)
        self.silero_min_samples = max(1, (sample_rate * silero_min_speech_ms) // 1000)
        self.silero_min_speech_ms = silero_min_speech_ms
        self.silero_min_silence_ms = silero_min_silence_ms

        self._torch = None
        self._silero_model = None
        self._silero_utils = None
        self._silero_buf: deque[np.ndarray] = deque()
        self._silero_buf_samples = 0
        self._device = "cpu"

        self._init_silero()
        if self._silero_model is None:
            raise RuntimeError(
                "Silero VAD could not be initialized. "
                "Check your torch/network environment and try again."
            )

    def _init_silero(self):
        try:
            import torch

            self._torch = torch
            # print("[VAD] Loading Silero VAD model...")
            self._silero_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
            (self._silero_get_speech_timestamps, _, _, _, _) = utils
            self._silero_model.to(self._device)
            self._silero_model.eval()
        except Exception as e:
            print(f"[VAD] Silero load failed ({e}).")
            self._silero_model = None

    def _push_silero_frame(self, pcm: np.ndarray):
        self._silero_buf.append(pcm.copy())
        self._silero_buf_samples += len(pcm)
        while self._silero_buf and self._silero_buf_samples > self.silero_window_samples:
            dropped = self._silero_buf.popleft()
            self._silero_buf_samples -= len(dropped)

    def _is_speech_silero(self, pcm: np.ndarray) -> bool:
        if self._silero_model is None or self._torch is None:
            return False

        self._push_silero_frame(pcm)
        if self._silero_buf_samples < self.silero_min_samples:
            return False

        wav = np.concatenate(list(self._silero_buf)) if self._silero_buf else np.zeros(0, dtype=np.int16)
        if wav.size == 0:
            return False

        try:
            wav_f = to_float32(wav)
            wav_t = self._torch.from_numpy(wav_f).to(self._device)
            with self._torch.no_grad():
                ts = self._silero_get_speech_timestamps(
                    wav_t,
                    self._silero_model,
                    sampling_rate=self.sample_rate,
                    threshold=self.silero_threshold,
                    min_speech_duration_ms=self.silero_min_speech_ms,
                    min_silence_duration_ms=self.silero_min_silence_ms,
                )
            return len(ts) > 0
        except Exception as e:
            raise RuntimeError(f"Silero VAD failed during inference: {e}") from e

    def is_speech(self, pcm: np.ndarray) -> bool:
        return self._is_speech_silero(pcm)

    def reset(self):
        """Clear internal buffers/state so old audio does not leak into the next decision."""
        self._silero_buf.clear()
        self._silero_buf_samples = 0
