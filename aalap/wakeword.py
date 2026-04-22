import logging
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import requests
from tqdm import tqdm

try:
    from openwakeword.model import Model as OWWModel

    OWW_AVAILABLE = True
except Exception:
    OWW_AVAILABLE = False

logger = logging.getLogger(__name__)

FEATURE_MODELS = {
    "embedding": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/embedding_model.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    },
    "melspectrogram": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/melspectrogram.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    },
}

VAD_MODELS = {
    "silero_vad": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/silero_vad.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx",
    }
}

MODELS = {
    "hey_jarvis": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_jarvis_v0.1.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
    },
}


@dataclass
class WakeWordEvent:
    kind: str
    label: str
    score: float
    threshold: float
    consecutive_hits: int
    patience_frames: int


class WakeWord:
    """
    openWakeWord streaming wrapper with explicit score threshold, patience, and debounce.
    """

    def __init__(
        self,
        wakeword_keywords: Union[str, List[str]] = "hey_jarvis",
        wakeword_model_paths: Union[str, List[str]] = None,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        vad_threshold: float = 0.5,
        score_thresh: float = 0.45,
        patience_frames: int = 2,
        debounce_ms: int = 900,
        _model_factory: Optional[Callable[..., object]] = None,
        _time_fn: Optional[Callable[[], float]] = None,
    ):
        if isinstance(wakeword_keywords, str):
            self.labels = [wakeword_keywords]
        else:
            try:
                self.labels = list(wakeword_keywords) if wakeword_keywords is not None else []
            except TypeError:
                self.labels = []

        self.enabled = OWW_AVAILABLE and len(self.labels) > 0
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        self.chunk_samples = sample_rate * 80 // 1000
        self._chunk_buf: deque[int] = deque()
        self._printed_labels = False

        self.vad_threshold = vad_threshold
        self.score_thresh = score_thresh
        self.patience_frames = max(1, patience_frames)
        self.debounce_ms = max(0, debounce_ms)
        self.near_trigger_ratio = 0.8

        self._consecutive_hits = 0
        self._debounce_until_ms = 0.0
        self._near_trigger_active = False
        self._last_score = 0.0
        self._last_label = self.labels[0] if self.labels else ""
        self._time_fn = _time_fn or (lambda: time.time() * 1000.0)

        model_factory = _model_factory or OWWModel

        if self.enabled:
            models_dir = Path(os.path.expanduser("~/.cache/aalap"))
            models_dir.mkdir(parents=True, exist_ok=True)
            self.download_models(model_names=[], target_directory=models_dir)
            self._ensure_openwakeword_vad_model(models_dir)

            melspec_model_path = os.path.join(models_dir, "melspectrogram.onnx")
            embedding_model_path = os.path.join(models_dir, "embedding_model.onnx")
            jarvis_model_path = os.path.join(models_dir, "hey_jarvis.onnx")

            if isinstance(wakeword_model_paths, str):
                wakeword_model_paths = [wakeword_model_paths]
            else:
                try:
                    wakeword_model_paths = list(wakeword_model_paths) if wakeword_model_paths is not None else []
                except TypeError:
                    wakeword_model_paths = []

            if len(self.labels) == 1 and "hey_jarvis" in self.labels[0]:
                self.download_models(model_names=["hey_jarvis"], target_directory=models_dir)
                wakeword_model_paths = [jarvis_model_path]
            elif not wakeword_model_paths:
                raise ValueError("Custom wakeword specified but no model path provided. Please supply wakeword_model_paths.")

            self.model = model_factory(
                wakeword_models=wakeword_model_paths,
                inference_framework="onnx",
                vad_threshold=self.vad_threshold,
                melspec_model_path=melspec_model_path,
                embedding_model_path=embedding_model_path,
            )

    def _ensure_openwakeword_vad_model(self, models_dir: Path) -> None:
        """
        openWakeWord's internal VAD loader expects a packaged resource path.
        Reuse Aalap's cached silero_vad.onnx by linking or copying it there.
        """
        if self.vad_threshold <= 0:
            return

        cache_vad_path = models_dir / "silero_vad.onnx"
        if not cache_vad_path.exists():
            return

        try:
            import openwakeword
        except Exception:
            return

        package_vad_path = Path(openwakeword.__file__).resolve().parent / "resources" / "models" / "silero_vad.onnx"
        if package_vad_path.exists():
            return

        package_vad_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            package_vad_path.symlink_to(cache_vad_path)
        except OSError:
            shutil.copy2(cache_vad_path, package_vad_path)

    def _predict_score(self, chunk_i16: np.ndarray) -> tuple[str, float]:
        preds = self.model.predict(chunk_i16) or {}
        if not self._printed_labels:
            missing = [k for k in self.labels if k not in preds]
            if missing:
                logger.warning("[WakeWord] Missing labels (0'd): %s. Make sure your wakeword and model file have the same name.", missing)
            self._printed_labels = True

        label = ""
        score = 0.0
        for candidate in self.labels:
            candidate_score = float(preds.get(candidate, 0.0))
            if candidate_score >= score:
                score = candidate_score
                label = candidate
        return label, score

    def _make_event(self, kind: str, label: str, score: float) -> WakeWordEvent:
        return WakeWordEvent(
            kind=kind,
            label=label or self._last_label,
            score=score,
            threshold=self.score_thresh,
            consecutive_hits=self._consecutive_hits,
            patience_frames=self.patience_frames,
        )

    def _process_chunk(self, chunk_i16: np.ndarray, now_ms: float) -> tuple[float, bool, Optional[WakeWordEvent]]:
        label, raw_score = self._predict_score(chunk_i16)
        self._last_score = raw_score
        if label:
            self._last_label = label

        near_threshold = self.score_thresh * self.near_trigger_ratio

        if now_ms < self._debounce_until_ms:
            self._consecutive_hits = 0
            if raw_score < near_threshold:
                self._near_trigger_active = False
            return raw_score, False, None

        event: Optional[WakeWordEvent] = None
        fired = False

        if raw_score >= self.score_thresh:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        if raw_score >= near_threshold and self._consecutive_hits < self.patience_frames:
            if not self._near_trigger_active:
                event = self._make_event("near_trigger", label, raw_score)
                self._near_trigger_active = True
        else:
            self._near_trigger_active = False

        if self._consecutive_hits >= self.patience_frames:
            fired = True
            event = self._make_event("trigger", label, raw_score)
            self._debounce_until_ms = now_ms + self.debounce_ms
            self._consecutive_hits = 0
            self._near_trigger_active = False

        if raw_score < near_threshold:
            self._near_trigger_active = False

        return raw_score, fired, event

    def step(self, mic_frame_f32: np.ndarray, now_ms: Optional[float] = None) -> tuple[float, bool, Optional[WakeWordEvent]]:
        """
        Feed one 20 ms frame (float32) and return (raw_score, fired, optional_event).
        """
        if not self.enabled:
            return 0.0, False, None

        mic_i16 = (np.clip(mic_frame_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        self._chunk_buf.extend(mic_i16.tolist())

        current_time_ms = now_ms if now_ms is not None else self._time_fn()
        latest_score = self._last_score
        fired = False
        event: Optional[WakeWordEvent] = None

        while len(self._chunk_buf) >= self.chunk_samples:
            chunk = np.fromiter((self._chunk_buf.popleft() for _ in range(self.chunk_samples)), dtype=np.int16, count=self.chunk_samples)
            latest_score, fired, current_event = self._process_chunk(chunk, current_time_ms)
            if current_event is not None:
                event = current_event
            if fired:
                break

        return latest_score, fired, event

    def reset(self) -> None:
        self._chunk_buf.clear()
        self._consecutive_hits = 0
        self._debounce_until_ms = 0.0
        self._near_trigger_active = False
        self._last_score = 0.0
        if getattr(self, "model", None) is not None and hasattr(self.model, "reset"):
            self.model.reset()

    def download_file(self, url, target_directory, local_filename=None, file_size=None):
        """Download a file with a progress bar using requests."""
        if local_filename is None:
            local_filename = url.split("/")[-1]

        with requests.get(url, stream=True) as r:
            if file_size is not None:
                progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True, desc=f"{local_filename}")
            else:
                total_size = int(r.headers.get("content-length", 0))
                progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"{local_filename}")

            with open(os.path.join(target_directory, local_filename), "wb") as f:
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
        Download wake-word models and dependencies into target_directory.
        - FEATURE_MODELS and VAD_MODELS keep their original filenames.
        - Wakeword models are saved as <model_name>.onnx (using MODELS keys).
        """
        if model_names is None:
            model_names = list(MODELS.keys())

        target_directory = Path(target_directory)
        target_directory.mkdir(parents=True, exist_ok=True)

        for feat in FEATURE_MODELS.values():
            fname = Path(feat["download_url"]).name
            target = target_directory / fname
            if not target.exists():
                self.download_file(feat["download_url"], str(target_directory))

        for vad in VAD_MODELS.values():
            fname = Path(vad["download_url"]).name
            target = target_directory / fname
            if not target.exists():
                self.download_file(vad["download_url"], str(target_directory))

        for name in model_names:
            info = MODELS.get(name)
            if not info:
                continue
            url = info["download_url"]
            if url.endswith(".tflite"):
                url = url.replace(".tflite", ".onnx")
            dest_name = f"{name}.onnx"
            dest_path = target_directory / dest_name
            if dest_path.exists():
                continue
            self.download_file(url, str(target_directory), local_filename=str(dest_name))
