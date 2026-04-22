import unittest
from unittest import mock
import importlib

import numpy as np

cli = importlib.import_module("aalap.cli")
dm = importlib.import_module("aalap.dialogue_manager")
from aalap.wakeword import WakeWord


def feed_chunk(ww: WakeWord, *, now_ms: int) -> tuple[float, bool, object]:
    result = (0.0, False, None)
    frame = np.zeros(320, dtype=np.float32)
    for _ in range(4):
        result = ww.step(frame, now_ms=now_ms)
    return result


class FakeOWWModel:
    def __init__(self, scores, **kwargs):
        self.scores = list(scores)
        self.calls = 0
        self.reset_calls = 0

    def predict(self, x):
        idx = min(self.calls, len(self.scores) - 1)
        self.calls += 1
        return {"custom": self.scores[idx]}

    def reset(self):
        self.reset_calls += 1
        self.calls = 0


class DummyASR:
    def __init__(self, *args, **kwargs):
        pass


class DummyTTS:
    def __init__(self, *args, **kwargs):
        pass


class DummyVAD:
    def __init__(self, *args, **kwargs):
        pass


class DummyWakeWord:
    def __init__(self, *args, **kwargs):
        self.labels = ["custom"]


class DummyAudioCapture:
    def __init__(self, *args, **kwargs):
        pass


class DummyTTSPlayer:
    def __init__(self, *args, **kwargs):
        pass


class WakeWordTests(unittest.TestCase):
    def make_wakeword(self, scores, **kwargs) -> WakeWord:
        with mock.patch.object(WakeWord, "download_models", autospec=True, side_effect=lambda *args, **kw: None):
            return WakeWord(
                wakeword_keywords="custom",
                wakeword_model_paths=["custom.onnx"],
                vad_threshold=0.0,
                _model_factory=lambda **factory_kwargs: FakeOWWModel(scores, **factory_kwargs),
                **kwargs,
            )

    def test_triggers_only_after_patience_frames(self):
        ww = self.make_wakeword([0.52, 0.61], score_thresh=0.45, patience_frames=2, debounce_ms=900)

        score, fired, event = feed_chunk(ww, now_ms=0)
        self.assertAlmostEqual(score, 0.52)
        self.assertFalse(fired)
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "near_trigger")

        score, fired, event = feed_chunk(ww, now_ms=120)
        self.assertAlmostEqual(score, 0.61)
        self.assertTrue(fired)
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "trigger")

    def test_debounce_blocks_repeated_triggers(self):
        ww = self.make_wakeword([0.7, 0.75, 0.8], score_thresh=0.45, patience_frames=1, debounce_ms=900)

        _, fired, event = feed_chunk(ww, now_ms=0)
        self.assertTrue(fired)
        self.assertEqual(event.kind, "trigger")

        _, fired, event = feed_chunk(ww, now_ms=300)
        self.assertFalse(fired)
        self.assertIsNone(event)

        _, fired, event = feed_chunk(ww, now_ms=1000)
        self.assertTrue(fired)
        self.assertEqual(event.kind, "trigger")

    def test_near_trigger_event_without_firing(self):
        ww = self.make_wakeword([0.40, 0.20], score_thresh=0.45, patience_frames=2, debounce_ms=900)

        score, fired, event = feed_chunk(ww, now_ms=0)
        self.assertAlmostEqual(score, 0.40)
        self.assertFalse(fired)
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "near_trigger")

        _, fired, event = feed_chunk(ww, now_ms=120)
        self.assertFalse(fired)
        self.assertIsNone(event)

    def test_reset_clears_model_and_internal_state(self):
        ww = self.make_wakeword([0.7], score_thresh=0.45, patience_frames=1, debounce_ms=900)

        _, fired, event = feed_chunk(ww, now_ms=0)
        self.assertTrue(fired)
        self.assertEqual(event.kind, "trigger")

        ww.reset()

        self.assertEqual(ww.model.reset_calls, 1)
        self.assertEqual(ww._consecutive_hits, 0)
        self.assertEqual(ww._debounce_until_ms, 0.0)
        self.assertEqual(len(ww._chunk_buf), 0)


class CliConfigTests(unittest.TestCase):
    def test_cli_maps_new_wakeword_arguments(self):
        parser = cli._build_arg_parser()
        args = parser.parse_args(
            [
                "--wakeword-score-thresh",
                "0.5",
                "--wakeword-patience-frames",
                "3",
                "--wakeword-debounce-ms",
                "1200",
                "--wakeword-debug",
                "--save-wakeword-debug-audio",
                "--wakeword-debug-audio-dir",
                "clips",
            ]
        )
        config = cli._build_config(args)

        self.assertEqual(config["wakeword_score_thresh"], 0.5)
        self.assertEqual(config["wakeword_patience_frames"], 3)
        self.assertEqual(config["wakeword_debounce_ms"], 1200)
        self.assertTrue(config["wakeword_debug"])
        self.assertTrue(config["save_wakeword_debug_audio"])
        self.assertEqual(config["wakeword_debug_audio_dir"], "clips")


if __name__ == "__main__":
    unittest.main()
