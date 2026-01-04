# Aalap

<!-- [![Downloads](https://img.shields.io/github/downloads/MnAkash/aalap/total)](https://github.com/MnAkash/aalap/releases) -->
[![Commits](https://img.shields.io/github/commit-activity/t/MnAkash/aalap)](https://github.com/MnAkash/aalap/commits/main)
[![Stars](https://img.shields.io/github/stars/MnAkash/aalap)](https://github.com/MnAkash/aalap/stargazers)
[![Forks](https://img.shields.io/github/forks/MnAkash/aalap)](https://github.com/MnAkash/aalap/network/members)

Aalap is a Python voice-assistant dialogue manager that combines wake word detection, VAD, streaming ASR, and TTS playback in a single loop. It is built around a threaded state machine and is usable both as a CLI tool and a library component.

## Project status

- Maturity: early-stage (v0.1.0); APIs may change before 1.0.
- Maintenance: active development.
- Supported Python: 3.9+.
- Platforms: tested on Ubuntu; should work on macOS and Windows with PortAudio, but not yet validated.

## Features

- Wake word detection via [openWakeWord](https://github.com/dscripka/openWakeWord) with smoothing, hysteresis, and custom model support.
- Voice activity detection using [Silero VAD](https://github.com/snakers4/silero-vad) (default) with [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) fallback.
- Streaming ASR in a worker process using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- Offline TTS with [Piper](https://github.com/rhasspy/piper) or online TTS via [gTTS](https://github.com/pndurette/gTTS).
- Shared input/output audio backend with [sounddevice](https://python-sounddevice.readthedocs.io/) and barge-in handling.
- Optional transcript audio capture with `ffmpeg`.
- Programmatic triggers and status/transcript callbacks.

## Installation

### Requirements

- Python 3.9+
- PortAudio (required by [sounddevice](https://python-sounddevice.readthedocs.io/))
- `ffmpeg` is recommended for transcript audio saving and for MP3 decoding via [pydub](https://github.com/jiaaro/pydub)

### System packages

Install PortAudio and `ffmpeg` for your OS:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y libportaudio2 portaudio19-dev ffmpeg
```

```bash
# macOS (Homebrew)
brew install portaudio ffmpeg
```

```bash
# Windows (Chocolatey)
choco install portaudio ffmpeg
```

### Install with pip (no clone)

```bash
python3 -m pip install "git+https://github.com/MnAkash/aalap.git"
```

Dependencies are listed in [requirements.txt](requirements.txt).

### Install from source

```bash
git clone https://github.com/MnAkash/aalap.git
cd aalap
python -m pip install -e .
```

## Quickstart (CLI)

After installation, run:

```bash
aalap
```

This uses the defaults defined in [aalap/dialogue_manager.py](aalap/dialogue_manager.py).

## Quickstart (Python)

```python
import time
import queue
from aalap.dialogue_manager import DialogManager

transcript_q: queue.Queue[str] = queue.Queue()
status_q: queue.Queue[str] = queue.Queue()

def on_transcript(text: str) -> None:
    transcript_q.put(text)

def on_status(status: str) -> None:
    status_q.put(status)

def my_policy(user_text: str) -> str:
    # Replace with your LLM or rules. Return a reply string.
    return f"You said: {user_text}"

manager = DialogManager(
    model="base.en",
    device="auto",
    tts_backend="piper",
    wakeword_keywords="hey_jarvis",
    wakeword_model_paths=None,
    on_transcript=on_transcript,
    on_status=on_status,
    external_policy=my_policy,
)
manager.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    manager.stop()
```

A fuller example is in [examples/simple_dialogue.py](examples/simple_dialogue.py).

## Runtime control

The `DialogManager` exposes a few useful control methods in [aalap/dialogue_manager.py](aalap/dialogue_manager.py):

- `trigger_wakeword()` to start listening programmatically
- `deactivate_wakeword_session()` to force the session back to IDLE
- `speak(text)` to enqueue TTS output directly

## Status callback

When you pass `on_status`, the callback receives the dialog state string emitted by the state machine in [aalap/dialogue_manager.py](aalap/dialogue_manager.py):

- `IDLE`: waiting for wake word or programmatic trigger
- `LISTENING`: waiting for user speech to start recording
- `RECORDING`: capturing user speech
- `TRANSCRIBING`: running ASR on the captured audio
- `THINKING`: waiting on the external policy to return a reply
- `SPEAKING`: playing back TTS audio
- `WAKEWORD_TRIGGER`: wake word fired and session is activating
- `SYSTEM_TRIGGER`: programmatic trigger fired and session is activating

## Configuration highlights

Most knobs are in [aalap/dialogue_manager.py](aalap/dialogue_manager.py) and exposed through the `DialogManager` constructor.

- Wake word: `wakeword_keywords`, `wakeword_model_paths` (see [aalap/wakeword.py](aalap/wakeword.py))
- VAD: `vad_backend`, `vad_silero_threshold`, `vad_silero_window_ms`, `vad_silero_min_speech_ms`
- ASR: `model`, `device` (uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- TTS: `tts_backend`, `piper_language`, `piper_voice`, `piper_quality`
- Timing: `silence_ms_after_speech`, `no_speech_timeout`, `post_tts_mute`
- Debug audio capture: `save_transcript_audio`, `transcript_audio_dir`

## Wake word models

By default, the built-in `"hey_jarvis"` model is downloaded automatically. If you provide custom wake words, you must supply matching model paths and name them `<wakeword>.onnx`.

Model downloads are cached under `~/.cache/aalap` (see [aalap/wakeword.py](aalap/wakeword.py)).

## Piper voice models

Piper voices are fetched from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) and cached under `~/.cache/aalap/piper` (see [aalap/tts_piper.py](aalap/tts_piper.py)).

## Audio device selection

List available devices with:

```bash
python -m aalap.list_soundDevices
```

See [aalap/list_soundDevices.py](aalap/list_soundDevices.py).

## Notes

- `gTTS` requires network access and depends on MP3 decoding via [pydub](https://github.com/jiaaro/pydub).
- Transcript audio saving uses `ffmpeg` (see `_save_audio_debug` in [aalap/dialogue_manager.py](aalap/dialogue_manager.py)).
- If [openWakeWord](https://github.com/dscripka/openWakeWord) is not installed or fails to load, wake word detection is disabled and only programmatic triggers are available.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) downloads ASR models from the [Hugging Face Hub](https://huggingface.co/models).

## License

MIT. See [LICENSE](LICENSE).

## Collaboration

This is an open-source project and contributions are welcome via pull requests. Please open an issue first for major changes so we can align on scope and approach.
