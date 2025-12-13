import os
from collections import deque
from typing import Optional, List, Union

import numpy as np

try:
    from openwakeword.model import Model as OWWModel
    OWW_AVAILABLE = True
except Exception:
    OWW_AVAILABLE = False


class WakeWord:
    """
    openWakeWord with EMA+hysteresis+latch.
    - Fires on rising edge: EMA crosses arm_thresh.
    - Re-arms only after EMA crosses back below disarm_thresh.
    - Optional non-overlapping evaluation to avoid double-firing on the same window.
    """
    def __init__(
        self,
        wakeword_keywords: Union[str, List[str]] = "hey_jarvis",
        wakeword_model_paths: Union[str, List[str]] = None,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        window_ms: int = 500,
        ema_alpha: float = 0.30,
        arm_thresh: float = 0.001,
        disarm_thresh: float = 0.01,
        non_overlap: bool = True,
    ):
        if isinstance(wakeword_keywords, str):
            self.labels = [wakeword_keywords]
        else:
            try:
                self.labels = list(wakeword_keywords) if wakeword_keywords is not None else []
            except TypeError:
                self.labels = []

        self.enabled = OWW_AVAILABLE and len(self.labels) > 0
        self.window_frames = max(1, window_ms // frame_ms)
        frame_samples = sample_rate * frame_ms // 1000
        self.window_samples = self.window_frames * frame_samples
        self._buf = deque(maxlen=self.window_samples)
        self._printed_labels = False

        # edge trigger state (no timers)
        self.ema = 0.0
        self.armed = True

        # for non-overlap evaluation
        self._stride_count = 0

        self.ema_alpha = ema_alpha
        self.arm_thresh = arm_thresh
        self.disarm_thresh = disarm_thresh
        self.non_overlap = non_overlap

        if self.enabled:
            if isinstance(wakeword_model_paths, str):
                wakeword_model_paths = [wakeword_model_paths]
            else:
                try:
                    wakeword_model_paths = list(wakeword_model_paths) if wakeword_model_paths is not None else []
                except TypeError:
                    wakeword_model_paths = []
            self.model = OWWModel(wakeword_model_paths=wakeword_model_paths)

    def _predict_score(self, window_i16: np.ndarray) -> float:
        preds = self.model.predict(window_i16) or {}
        if not self._printed_labels:
            missing = [k for k in self.labels if k not in preds]
            if missing:
                print(f"[WakeWord] Missing labels (0'd): {missing}")
            self._printed_labels = True
        return max((preds.get(lbl, 0.0) for lbl in self.labels), default=0.0)

    def step(self, mic_frame_f32: np.ndarray) -> tuple[float, bool]:
        """
        Feed one 20ms frame (float32) -> (ema, fired).
        """
        if not self.enabled:
            return 0.0, False

        # accumulate mic
        mic_i16 = (np.clip(mic_frame_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        self._buf.extend(mic_i16.tolist())

        # not enough samples yet
        if len(self._buf) < self.window_samples:
            return self.ema, False

        # optional non-overlap gating: evaluate once per full window
        self._stride_count += 1
        if self.non_overlap and (self._stride_count % self.window_frames != 0):
            # maintain EMA slowly drifting to zero when not evaluating
            self.ema = (1.0 - self.ema_alpha) * self.ema
            return self.ema, False

        # run model on the last full window
        window = np.asarray(self._buf, dtype=np.int16)
        raw = self._predict_score(window)

        # EMA smoothing
        self.ema = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self.ema

        # hysteresis + latch
        fired = False
        if self.armed and (self.ema >= self.arm_thresh):
            fired = True
            self.armed = False
            # clear buffer to avoid seeing exact same window again
            self._buf.clear()
            self._stride_count = 0
            self.ema = 0.0  # optional: drop EMA so we don't stay above ARM
        elif (not self.armed) and (self.ema <= self.disarm_thresh):
            # re-arm only after score truly falls low again
            self.armed = True

        return self.ema, fired
