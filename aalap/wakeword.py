import os
from collections import deque
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import requests
from tqdm import tqdm

try:
    from openwakeword.model import Model as OWWModel
    OWW_AVAILABLE = True
except Exception:
    OWW_AVAILABLE = False

FEATURE_MODELS = {
    "embedding": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/embedding_model.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
    },
    "melspectrogram": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/melspectrogram.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
    }
}

VAD_MODELS = {
    "silero_vad": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/silero_vad.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx"
    }
}

MODELS = {
    "hey_jarvis": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_jarvis_v0.1.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx"
    },
}


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
        arm_thresh: float = 0.05,
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
            models_dir = Path(os.path.expanduser("~/.cache/aalap"))
            models_dir.mkdir(parents=True, exist_ok=True)
            self.download_models(target_directory=models_dir)

            melspec_model_path   = os.path.join(models_dir, "melspectrogram.onnx")
            embedding_model_path = os.path.join(models_dir, "embedding_model.onnx")
            jarvis_model_path = os.path.join(models_dir, "hey_jarvis.onnx")

            if isinstance(wakeword_model_paths, str):
                wakeword_model_paths = [wakeword_model_paths]
            else:
                try:
                    wakeword_model_paths = list(wakeword_model_paths) if wakeword_model_paths is not None else []
                except TypeError:
                    wakeword_model_paths = []

            # Default behavior for hey_jarvis: download if missing
            if len(self.labels) == 1 and "hey_jarvis" in self.labels[0]:
                wakeword_model_paths = [jarvis_model_path]
            elif not wakeword_model_paths:
                raise ValueError("Custom wakeword specified but no model path provided. Please supply wakeword_model_paths.")

            print("Using wakeowrd path: ", wakeword_model_paths)
            self.model = OWWModel(wakeword_models=wakeword_model_paths,
                                  inference_framework='onnx',
                                  melspec_model_path = melspec_model_path,
                                  embedding_model_path = embedding_model_path
                                  )

    def _predict_score(self, window_i16: np.ndarray) -> float:
        preds = self.model.predict(window_i16) or {}
        if not self._printed_labels:
            missing = [k for k in self.labels if k not in preds]
            if missing:
                print(f"[WakeWord] Missing labels (0'd): {missing}")
                print(f"Make sure your wakeword and model file has same name")
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

    # Function to download files from a URL with a progress bar
    def download_file(self, url, target_directory, local_filename=None, file_size=None):
        """A simple function to download a file from a URL with a progress bar using only the requests library"""
        if local_filename == None:
            local_filename = url.split('/')[-1]

        with requests.get(url, stream=True) as r:
            if file_size is not None:
                progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f"{local_filename}")
            else:
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{local_filename}")

            with open(os.path.join(target_directory, local_filename), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()
    
    def download_models(
        self,
        model_names: List[str] = None,
        target_directory: Union[str, Path] = Path(os.path.expanduser("~/.cache/aalap")),
    ):
        """
        Download wake word models and dependencies into target_directory.
        - FEATURE_MODELS and VAD_MODELS keep their original filenames.
        - Wakeword models are saved as <model_name>.onnx (using MODELS keys).
        """
        if model_names is None:
            model_names = list(MODELS.keys())

        target_directory = Path(target_directory)
        target_directory.mkdir(parents=True, exist_ok=True)

        # Feature models (as-is)
        for feat in FEATURE_MODELS.values():
            fname = Path(feat["download_url"]).name
            target = target_directory / fname
            if not target.exists():
                self.download_file(feat["download_url"], str(target_directory))

        # VAD models (as-is)
        for vad in VAD_MODELS.values():
            fname = Path(vad["download_url"]).name
            target = target_directory / fname
            if not target.exists():
                self.download_file(vad["download_url"], str(target_directory))

        # Wakeword models: save as <model_name>.onnx
        for name in model_names:
            info = MODELS.get(name)
            if not info:
                continue
            url = info["download_url"]
            # ensure we grab the ONNX variant and name it <key>.onnx
            if url.endswith(".tflite"):
                url = url.replace(".tflite", ".onnx")
            dest_name = f"{name}.onnx"
            dest_path = target_directory / dest_name
            if dest_path.exists():
                continue
            self.download_file(url, str(target_directory), local_filename= str(dest_name))
