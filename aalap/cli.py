import argparse
import multiprocessing as mp
import time
from typing import Any, Optional

from .dialogue_manager import (
    ASR_TIMEOUT,
    DialogManager,
    LISTEN_NO_SPEECH_TIMEOUT_MS,
    PIPER_LANGUAGE,
    PIPER_QUALITY,
    PIPER_VOICE,
    POST_TTS_MUTE_MS,
    SAVE_TRANSCRIPT_AUDIO,
    SILENCE_MS_AFTER_SPEECH,
    TRANSCRIPT_AUDIO_DIR,
    TTS_BACKEND,
    VAD_SILERO_MIN_SILENCE_MS,
    VAD_SILERO_MIN_SPEECH_MS,
    VAD_SILERO_THRESHOLD,
    VAD_SILERO_WINDOW_MS,
    WAKEWORD_ARM_THRESH,
    WAKEWORD_DISARM_THRESH,
    WAKEWORD_EMA_ALPHA,
    WAKEWORD_KEYWORDS,
    WAKEWORD_WINDOW_MS,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)


CLI_DEFAULTS: dict[str, Any] = {
    "model": WHISPER_MODEL,
    "device": WHISPER_DEVICE,
    "mic_index": None,
    "speaker_index": None,
    "asr_timeout": ASR_TIMEOUT,
    "silence_ms_after_speech": SILENCE_MS_AFTER_SPEECH,
    "no_speech_timeout": LISTEN_NO_SPEECH_TIMEOUT_MS,
    "post_tts_mute": POST_TTS_MUTE_MS,
    "tts_backend": TTS_BACKEND,
    "piper_language": PIPER_LANGUAGE,
    "piper_voice": PIPER_VOICE,
    "piper_quality": PIPER_QUALITY,
    "wakeword_keywords": WAKEWORD_KEYWORDS,
    "wakeword_model_paths": None,
    "wakeword_window_ms": WAKEWORD_WINDOW_MS,
    "wakeword_ema_alpha": WAKEWORD_EMA_ALPHA,
    "wakeword_arm_thresh": WAKEWORD_ARM_THRESH,
    "wakeword_disarm_thresh": WAKEWORD_DISARM_THRESH,
    "vad_silero_threshold": VAD_SILERO_THRESHOLD,
    "vad_silero_window_ms": VAD_SILERO_WINDOW_MS,
    "vad_silero_min_speech_ms": VAD_SILERO_MIN_SPEECH_MS,
    "vad_silero_min_silence_ms": VAD_SILERO_MIN_SILENCE_MS,
    "save_transcript_audio": SAVE_TRANSCRIPT_AUDIO,
    "transcript_audio_dir": TRANSCRIPT_AUDIO_DIR,
    "print_transcripts": False,
    "print_status": False,
}


def _parse_list_arg(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    items = [item for item in items if item]
    return items or []


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aalap",
        description="Run the Aalap voice assistant loop.",
    )
    parser.add_argument("--model", help="Whisper model name or local converted model path.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="ASR device.")
    parser.add_argument("--mic-index", type=int, help="Microphone input device index.")
    parser.add_argument("--speaker-index", type=int, help="Speaker output device index.")
    parser.add_argument("--asr-timeout", type=float, help="Maximum seconds to wait for ASR output.")
    parser.add_argument("--tts-backend", choices=["piper", "gtts"], help="TTS backend.")
    parser.add_argument("--piper-language", help="Piper language code, e.g. en_US.")
    parser.add_argument("--piper-voice", help="Piper voice name.")
    parser.add_argument("--piper-quality", help="Piper quality level.")
    parser.add_argument(
        "--wakeword-keywords",
        help="Comma-separated wakeword labels. Use an empty string to disable wakeword detection.",
    )
    parser.add_argument(
        "--wakeword-model-paths",
        help="Comma-separated custom wakeword model paths matching --wakeword-keywords.",
    )
    parser.add_argument("--wakeword-window-ms", type=int, help="Wakeword inference window size.")
    parser.add_argument("--wakeword-ema-alpha", type=float, help="Wakeword EMA smoothing factor.")
    parser.add_argument("--wakeword-arm-thresh", type=float, help="Wakeword trigger threshold.")
    parser.add_argument("--wakeword-disarm-thresh", type=float, help="Wakeword re-arm threshold.")
    parser.add_argument("--vad-silero-threshold", type=float, help="Silero VAD threshold 0-1.")
    parser.add_argument("--vad-silero-window-ms", type=int, help="Silero VAD rolling window size.")
    parser.add_argument("--vad-silero-min-speech-ms", type=int, help="Silero minimum speech duration.")
    parser.add_argument("--vad-silero-min-silence-ms", type=int, help="Silero minimum silence duration.")
    parser.add_argument("--silence-ms-after-speech", type=int, help="Endpoint silence hangover.")
    parser.add_argument("--no-speech-timeout", type=int, help="Session inactivity timeout in ms.")
    parser.add_argument("--post-tts-mute", type=int, help="Ignore VAD after TTS for this many ms.")
    parser.add_argument("--transcript-audio-dir", help="Directory for saved transcript audio.")
    parser.add_argument(
        "--save-transcript-audio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable saving transcript audio for debugging.",
    )
    parser.add_argument(
        "--print-transcripts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print transcript text to stdout.",
    )
    parser.add_argument(
        "--print-status",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print status transitions to stdout.",
    )
    return parser


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(CLI_DEFAULTS)
    overrides = {
        "model": args.model,
        "device": args.device,
        "mic_index": args.mic_index,
        "speaker_index": args.speaker_index,
        "asr_timeout": args.asr_timeout,
        "tts_backend": args.tts_backend,
        "piper_language": args.piper_language,
        "piper_voice": args.piper_voice,
        "piper_quality": args.piper_quality,
        "wakeword_keywords": _parse_list_arg(args.wakeword_keywords),
        "wakeword_model_paths": _parse_list_arg(args.wakeword_model_paths),
        "wakeword_window_ms": args.wakeword_window_ms,
        "wakeword_ema_alpha": args.wakeword_ema_alpha,
        "wakeword_arm_thresh": args.wakeword_arm_thresh,
        "wakeword_disarm_thresh": args.wakeword_disarm_thresh,
        "vad_silero_threshold": args.vad_silero_threshold,
        "vad_silero_window_ms": args.vad_silero_window_ms,
        "vad_silero_min_speech_ms": args.vad_silero_min_speech_ms,
        "vad_silero_min_silence_ms": args.vad_silero_min_silence_ms,
        "silence_ms_after_speech": args.silence_ms_after_speech,
        "no_speech_timeout": args.no_speech_timeout,
        "post_tts_mute": args.post_tts_mute,
        "save_transcript_audio": args.save_transcript_audio,
        "transcript_audio_dir": args.transcript_audio_dir,
        "print_transcripts": args.print_transcripts,
        "print_status": args.print_status,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _dialog_manager_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"],
        "device": config["device"],
        "mic_index": config["mic_index"],
        "speaker_index": config["speaker_index"],
        "asr_timeout": config["asr_timeout"],
        "silence_ms_after_speech": config["silence_ms_after_speech"],
        "no_speech_timeout": config["no_speech_timeout"],
        "post_tts_mute": config["post_tts_mute"],
        "tts_backend": config["tts_backend"],
        "piper_language": config["piper_language"],
        "piper_voice": config["piper_voice"],
        "piper_quality": config["piper_quality"],
        "wakeword_keywords": config["wakeword_keywords"],
        "wakeword_model_paths": config["wakeword_model_paths"],
        "wakeword_window_ms": config["wakeword_window_ms"],
        "wakeword_ema_alpha": config["wakeword_ema_alpha"],
        "wakeword_arm_thresh": config["wakeword_arm_thresh"],
        "wakeword_disarm_thresh": config["wakeword_disarm_thresh"],
        "vad_silero_threshold": config["vad_silero_threshold"],
        "vad_silero_window_ms": config["vad_silero_window_ms"],
        "vad_silero_min_speech_ms": config["vad_silero_min_speech_ms"],
        "vad_silero_min_silence_ms": config["vad_silero_min_silence_ms"],
        "save_transcript_audio": config["save_transcript_audio"],
        "transcript_audio_dir": config["transcript_audio_dir"],
    }


def run(config: Optional[dict[str, Any]] = None) -> int:
    merged_config = dict(CLI_DEFAULTS if config is None else config)

    def _on_transcript(text: str) -> None:
        if merged_config["print_transcripts"] and text:
            print(text, flush=True)

    def _on_status(status: str) -> None:
        if merged_config["print_status"]:
            print(status, flush=True)

    dm = DialogManager(
        **_dialog_manager_kwargs(merged_config),
        on_transcript=_on_transcript if merged_config["print_transcripts"] else None,
        on_status=_on_status if merged_config["print_status"] else None,
        external_policy=None,
    )
    dm.start()

    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        return 0
    finally:
        dm.stop()


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    return run(_build_config(args))


if __name__ == "__main__":
    raise SystemExit(main())
