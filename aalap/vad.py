from collections import deque

import numpy as np
import webrtcvad

DEFAULT_SAMPLE_RATE = 16000


def to_float32(pcm16: np.ndarray) -> np.ndarray:
    return pcm16.astype(np.float32) / 32768.0


def to_bytes(pcm16: np.ndarray) -> bytes:
    return pcm16.astype(np.int16).tobytes()


class VAD:
    "Voice Activity Detection with Silero (default) or WebRTC fallback + energy gating."

    def __init__(
        self,
        backend: str = "silero",
        webrtc_aggressiveness=1,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        silero_threshold: float = 0.5,
        silero_window_ms: int = 320,
        silero_min_speech_ms: int = 60,
        silero_min_silence_ms: int = 0,
    ):
        """
        Args:
            backend: Which VAD to use. "silero" (default) or "webrtc". Falls back to WebRTC if Silero load fails.
            webrtc_aggressiveness: WebRTC aggressiveness 0..3. Higher = stricter (more likely to reject speech).
            sample_rate: Input sample rate in Hz. Both VADs expect 16 kHz mono PCM16.
            silero_threshold: Silero speech probability threshold (0..1). Lower = more sensitive.
            silero_window_ms: Rolling window size Silero sees for a decision (ms). Longer can improve stability.
            silero_min_speech_ms: Minimum speech duration for Silero to count a segment (ms).
            silero_min_silence_ms: Minimum silence duration between segments for Silero (ms).
        """
        self.sample_rate = sample_rate
        self.backend = backend.lower().strip()
        self.silero_threshold = silero_threshold
        self.silero_window_samples = max(1, (sample_rate * silero_window_ms) // 1000)
        self.silero_min_samples = max(1, (sample_rate * silero_min_speech_ms) // 1000)
        self.silero_min_speech_ms = silero_min_speech_ms
        self.silero_min_silence_ms = silero_min_silence_ms
        self.webrtc_aggressiveness = webrtc_aggressiveness

        self._torch = None
        self._silero_model = None
        self._silero_utils = None
        self._silero_buf: deque[np.ndarray] = deque()
        self._silero_buf_samples = 0
        self._device = "cpu"

        self._init_silero()
        if self._silero_model is None:
            self.backend = "webrtc"
            self._init_webrtc(self.webrtc_aggressiveness)

    def _init_silero(self):
        if self.backend != "silero":
            return
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
            print(f"[VAD] Silero load failed ({e}). Falling back to WebRTC.")
            self._silero_model = None

    def _init_webrtc(self, aggressiveness: int):
        self.v = webrtcvad.Vad(aggressiveness)
        # Noise floor for VAD gating
        self.noise_rms = 0.01           # initial guess
        self.noise_alpha = 0.10         # how fast we track noise (0..1)
        self.speech_snr = 4.0    # speech must be > SNR * noise 

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
            print(f"[VAD] Silero error ({e}); switching to WebRTC.")
            self.backend = "webrtc"
            self._init_webrtc(self.webrtc_aggressiveness)
            return self._is_speech_webrtc(pcm)

    def _is_speech_webrtc(self, pcm: np.ndarray) -> bool:
        pcm16_bytes = to_float32(pcm)
        rms = float(np.sqrt(np.mean(pcm16_bytes * pcm16_bytes) + 1e-12))

        clean_b = to_bytes(pcm)
        vad_flag = self.v.is_speech(clean_b, self.sample_rate)

        if not vad_flag:
            self.noise_rms = (1.0 - self.noise_alpha) * self.noise_rms + self.noise_alpha * rms

        is_speech = vad_flag and (rms > self.speech_snr * max(self.noise_rms, 1e-6))

        return is_speech

    def is_speech(self, pcm: np.ndarray) -> bool:
        if self.backend == "silero" and self._silero_model is not None:
            return self._is_speech_silero(pcm)
        return self._is_speech_webrtc(pcm)

    def reset(self):
        """Clear internal buffers/state so old audio does not leak into the next decision."""
        self._silero_buf.clear()
        self._silero_buf_samples = 0
        # Reset noise tracking for WebRTC gating
        self.noise_rms = 0.01
