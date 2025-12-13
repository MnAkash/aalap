#!/usr/bin/env python3
"""
Interactive voice loop with wake word activation, VAD, streaming ASR, and TTS playback.

Core pieces:
- Wake word detection via openWakeWord (custom model support) and programmatic trigger.
- WebRTC/Silero VAD at 16 kHz / 20 ms frames with barge-in handling.
- Streaming ASR using faster-whisper and Piper TTS playback on shared PortAudio devices.
- Optional transcript saving and a simple policy hook for responses.

Usable as a CLI entry point (`aalap`) or as a library component.
"""

import os, time, threading, queue, sys, subprocess, shutil
import multiprocessing as mp
from collections import deque
from pathlib import Path
from typing import Callable, Optional, List, Union

import numpy as np
import sounddevice as sd

os.environ.setdefault("MKL_NUM_THREADS", "1")      # limit MKL threads to avoid CPU thrash
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")  # limit numexpr threads
os.environ["ORT_NUM_THREADS"] = "1"                # onnxruntime intra-op threads
os.environ["ORT_DISABLE_THREAD_SPIN"] = "1"        # prevent ORT busy-spin
os.environ["OMP_NUM_THREADS"] = "1"                # cap OpenMP threads

from faster_whisper import WhisperModel


# Allow running both as a module and as a script.
try:
    from .vad import VAD, to_float32
    from .wakeword import WakeWord
    from .tts_piper import PiperTTS
    from .tts_player import TTSPlayer
except ImportError:
    from vad import VAD, to_float32  # type: ignore
    from wakeword import WakeWord  # type: ignore
    from tts_piper import PiperTTS  # type: ignore
    from tts_player import TTSPlayer  # type: ignore





# ----------- CONFIG -----------
CAPTURE_FRAME_MS = 20
SAMPLE_RATE = 16000
CAPTURE_FRAME_SAMPLES = SAMPLE_RATE * CAPTURE_FRAME_MS // 1000  # 320


MAX_LISTEN_MS               = 10_000           # 10s cap to avoid huge arrays to Whisper
MIN_SPEECH_MS               = 100            # require at least 100ms of speech before endpoint
LISTEN_NO_SPEECH_TIMEOUT_MS = 5000  # return to IDLE if no speech within 5s (will need wakeword to re-trigger)
SILENCE_MS_AFTER_SPEECH     = 800
PRE_SPEECH_MS               = 200  # prepend a bit of audio before VAD fires
SAVE_TRANSCRIPT_AUDIO       = False
TRANSCRIPT_AUDIO_DIR        = "debug_transcripts" # When SAVE_TRANSCRIPT_AUDIO = True save all transcribed audio to this dir
PRE_SPEECH_FRAMES           = max(0, PRE_SPEECH_MS // CAPTURE_FRAME_MS)
VAD_CALIBRATION_FRAMES      = 20  # 20*CAPTURE_FRAME_MS = 400 ms to calibrate VAD noise floor


# Wake word
USE_WAKEWORD = True
WAKEWORD_KEYWORDS = ("hey_jarvis",)
WAKEWORD_WINDOW_MS = 500
WAKEWORD_EMA_ALPHA     = 0.30   # smoothing, 0..1
WAKEWORD_ARM_THRESH    = 0.001   # cross up -> fire
WAKEWORD_DISARM_THRESH = 0.01   # cross down -> re-arm
# set this True to evaluate once per window (no overlapping windows)
WAKEWORD_NON_OVERLAP   = True



# VAD
WEBRTC_AGGRESSIVENESS       = 1  # 0..3 (lower the value to make it more sensitive to voice)
VAD_BACKEND                 = "silero" # "silero" or "webrtc"
VAD_SILERO_THRESHOLD        = 0.5  # 0..1, lower = more sensitive
VAD_SILERO_WINDOW_MS        = 320
VAD_SILERO_MIN_SPEECH_MS    = 60
VAD_SILERO_MIN_SILENCE_MS   = 0



# ASR
WHISPER_MODEL   = "base.en"
WHISPER_DEVICE  = "auto"  # "cuda" if GPU available
WHISPER_COMPUTE = "auto"

# Piper config
PIPER_LENGTH_SCALE  = 1.0  # >1 slower/deeper; <1 faster
PIPER_NOISE_SCALE   = 0.667
PIPER_NOISE_W       = 0.8



# --------------------------------------------------------------------

# ---------------- TTS Player ----------------


# ---------------- Audio capture ----------------
class AudioCapture:
    def __init__(self, device=None):
        self.q = queue.Queue(maxsize=10)
        self.stream = None
        self.device = device
        self.list_audio_devices()

        if device is None:
            default_in, default_out = sd.default.device
            print(f"[Info] Using default input index: {default_in}, output index: {default_out}")
        else:
            print(f"[Info] Using input device index: {device}")

    def _cb(self, indata, frames, time_info, status):
        # Keep the callback *tiny* — no allocations, no blocking.
        if status:
            pass  # you can print(status) if you want diagnostics
        # indata is already int16 (dtype="int16" in the stream), mono slice is a view
        # get mean of two channels if stereo
        if indata.shape[1] == 1:
            mono = indata[:, 0].copy()  # copy is cheap here; avoids referencing PortAudio buffer
        else:
            mono = np.mean(indata, axis=1).astype(np.int16)
        
        try:
            self.q.put_nowait(mono)  # push one 20 ms frame
        except queue.Full:
            # Drop the oldest frame and try once more (bounded queue = backpressure)
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(mono)
            except queue.Full:
                # If still full, just drop this frame — never block the callback
                pass
    def list_audio_devices(self):
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                print(f"[{i:2d}] {d['name']}  (in:{d['max_input_channels']}, out:{d['max_output_channels']})")
        except Exception as e:
            print(f"[WARN] Could not query audio devices: {e}")

    def start(self):

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=CAPTURE_FRAME_SAMPLES, callback=self._cb, device=self.device
        )
        self.stream.start()

    def read_frame(self) -> np.ndarray:
        f = self.q.get()
        if len(f) < CAPTURE_FRAME_SAMPLES:
            pad = np.zeros(CAPTURE_FRAME_SAMPLES-len(f), dtype=np.int16)
            f = np.concatenate([f, pad])
        elif len(f) > CAPTURE_FRAME_SAMPLES:
            f = f[:CAPTURE_FRAME_SAMPLES]
        return f

    def drain(self):
        """Drop all currently queued frames (non-blocking)."""
        dropped = 0
        try:
            while True:
                self.q.get_nowait()
                dropped += 1
        except queue.Empty:
            pass
        # optional: print for debugging
        # if dropped:
        #     print(f"[AudioCapture] Drained {dropped} frames from queue.")

    def stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close()

# ---------------- ASR ----------------
class ASRWorker(mp.Process):
    def __init__(self, in_q: mp.Queue, out_q: mp.Queue, model, device, compute_type):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.model_name = model
        self.device = device
        self.compute_type = compute_type

        
    def run(self):
        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type,
                             cpu_threads=4, num_workers=1)
        
        while True:
            pcm16 = self.in_q.get()
            if pcm16 is None:
                break
            try:
                audio = pcm16.astype(np.float32) / 32768.0
                segments, info = model.transcribe(audio, vad_filter=False, beam_size=1)
                text = "".join(seg.text for seg in segments).strip()
            except Exception as e:
                text = f"[ASR_ERROR]{e}"
            # don’t block parent indefinitely
            try:
                self.out_q.put_nowait(text)
            except:
                pass
class StreamingASR:
    def __init__(self, model=WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE):
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self._spawn()

    def _spawn(self):
        self.in_q = mp.Queue(maxsize=1)
        self.out_q = mp.Queue(maxsize=1)
        self.proc = ASRWorker(self.in_q, self.out_q, self.model, self.device, self.compute_type)
        self.proc.start()

    def _ensure(self):
        if not self.proc.is_alive():
            self._spawn()

    def transcribe_blocking(self, pcm16: np.ndarray, timeout_s: float = 10.0) -> str:
        self._ensure()
        # drop any stale results from previous calls
        try:
            while True:
                self.out_q.get_nowait()
        except queue.Empty:
            pass
        # drop if worker is busy
        try:
            self.in_q.put_nowait(pcm16)
        except:
            return ""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                return self.out_q.get_nowait()
            except:
                time.sleep(0.01)
        return ""  # timeout -> treat as silence

# ---------------- Dialog manager ----------------
class DialogManager:
    IDLE, LISTENING, RECORDING, SPEAKING, TRANSCRIBING = "IDLE", "LISTENING", "RECORDING", "SPEAKING", "TRANSCRIBING"
    '''
    IDLE            - waiting for wake word or programmatic trigger
    LISTENING       - waiting for user speech to start recording
    RECORDING       - recording user speech
    SPEAKING        - playing back TTS
    TRANSCRIBING    - transcribing recorded user speech
    '''
    def __init__(self,
                 model: str = "base.en",
                 mic_index: int = None,
                 speaker_index: int = None,
                 silence_ms_after_speech=SILENCE_MS_AFTER_SPEECH,
                 wakeword_keywords: Union[str, List[str]] = WAKEWORD_KEYWORDS, # set to None to disable
                 wakeword_model_paths: Optional[Union[str, List[str]]] = None,
                 vad_aggressiveness=WEBRTC_AGGRESSIVENESS,
                 vad_backend: str = VAD_BACKEND,
                 vad_silero_threshold: float = VAD_SILERO_THRESHOLD,
                 vad_silero_window_ms: int = VAD_SILERO_WINDOW_MS,
                 vad_silero_min_speech_ms: int = VAD_SILERO_MIN_SPEECH_MS,
                 vad_silero_min_silence_ms: int = VAD_SILERO_MIN_SILENCE_MS,
                 save_transcript_audio: bool = SAVE_TRANSCRIPT_AUDIO,
                 transcript_audio_dir = TRANSCRIPT_AUDIO_DIR,
                 on_transcript: Optional[Callable[[str], None]] = None,
                 on_status: Optional[Callable[[str], None]] = None):
        """
        Dialog manager that wires together wake word, VAD, streaming ASR, and Piper TTS.

        Args:
            model (str, default="base.en"): Specifies the size of the transcription
                model to use or the path to a converted model directory.
                Valid options are 'tiny', 'tiny.en', 'base', 'base.en',
                'small', 'small.en', 'medium', 'medium.en', 'large-v1',
                'large-v2'.
                If a specific size is provided, the model is downloaded
                from the Hugging Face Hub.

            mic_index (int, optional): Input device index for microphone.

            speaker_index (int, optional): Output device index for speakers.

            silence_ms_after_speech (int): How many ms of silence after speech to
                    consider the utterance ended.

            wakeword_keywords (Union[str, List[str]], optional): Wake word(s) (string or list of srtings) 
                    keywords to listen for. If None or empty list, wake word detection is disabled.
                    Default is "hey_jarvis". Options include "hey_jarvis", "alexa".

            wakeword_model_paths (List[str], optional): List of paths to wake word model path
                    corresponding to the keywords. By default pretrained models are used.
                    Make sure to name the model <wakeword>.onnx

            vad_aggressiveness (int): Aggressiveness level for WebRTC VAD (0-3).

            vad_backend (str): VAD backend to use: "webrtc" or "silero".
                    Default is "silero".

            vad_silero_threshold (float): Sensitivity threshold for Silero VAD (0-1).

            vad_silero_window_ms (int): Window size in ms for Silero VAD.

            vad_silero_min_speech_ms (int): Minimum speech duration in ms for Silero VAD.

            vad_silero_min_silence_ms (int): Minimum silence duration in ms for Silero VAD.

            save_transcript_audio (bool): If True, saves each transcribed utterance
                to an mp3 file in transcript_audio_dir for debugging.

            transcript_audio_dir (str): Directory to save transcribed audio files.

            on_transcript (Callable[[str], None], optional): Callback function that
                is called with the transcribed text after each utterance.

            on_status (Callable[[str], None], optional): Callback function that
                is called with status updates when the dialog state changes.
            
        """
        self.state = self.IDLE
        self.package_dir         = Path(__file__).resolve().parent.parent
        self.piper_model_path    = str(self.package_dir / "resources" / "models" / "en_US-amy-medium.onnx")

        self.asr        = StreamingASR(model=model, device="auto", compute_type="auto")
        self.tts_engine = PiperTTS(self.piper_model_path, PIPER_LENGTH_SCALE, PIPER_NOISE_SCALE, PIPER_NOISE_W)
        self.tts_player = TTSPlayer(device=speaker_index, sample_rate=SAMPLE_RATE, capture_frame_samples=CAPTURE_FRAME_SAMPLES)
        self.vad        = VAD(
                            sample_rate=SAMPLE_RATE,
                            backend=vad_backend,
                            webrtc_aggressiveness=vad_aggressiveness,
                            silero_threshold=vad_silero_threshold,
                            silero_window_ms=vad_silero_window_ms,
                            silero_min_speech_ms=vad_silero_min_speech_ms,
                            silero_min_silence_ms=vad_silero_min_silence_ms,
                        )
        self.ww         = WakeWord(wakeword_keywords, wakeword_model_paths=wakeword_model_paths)
        self.mic        = AudioCapture(device=mic_index)

        self.save_transcript_audio = save_transcript_audio
        self.transcript_audio_dir = transcript_audio_dir
        self.system_trigger_q = queue.Queue(maxsize=1)
        self.silence_ms_after_speech = silence_ms_after_speech  # endpoint hangover
        self.on_transcript = on_transcript
        self.on_status = on_status
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_status = None
        self._emit_status(self.state)

    def _emit_status(self, status: str, poststring: str = ""):
        if status == self._last_status:
            return
        self._last_status = status
        print(f"[Status]: {status} {poststring}")
        if self.on_status:
            try:
                self.on_status(status)
            except Exception as e:
                print(f"[StatusCallback] Error: {e}")

    def _set_state(self, new_state: str, poststring: str = ""):
        if self.state != new_state:
            self.state = new_state
            self._emit_status(new_state, poststring)

    def start(self):
        """Non-blocking start"""
        if self._thread and self._thread.is_alive():
            return self._thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self._thread

    def start_blocking(self):
        """Blocking start"""
        t = self.start()
        if t:
            t.join()

    def stop(self):
        self._stop_event.set()
        try:
            self.tts_player.stop()
        except Exception:
            pass
        try:
            self.mic.stop()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def trigger_wakeword(self):
        """Programmatically trigger listening. """
        try: self.system_trigger_q.put_nowait(True)
        except queue.Full: pass

    def speak(self, text: str):
        self._set_state(self.SPEAKING, poststring=f"({text})")
        self.tts_player.stop()
        # print(f"[Robot]: {text}")
        # def _run():
        pcm = self.tts_engine.synth(text)
        self.tts_player.play(pcm)
        # threading.Thread(target=_run, daemon=True).start()


    def policy(self, user_text: str) -> str:
        # Replace with your planner/LLM/BT
        if not user_text.strip():
            return ""
        # policy = f"You said: {user_text}"
        if "how are you" in user_text.lower():
            policy = "I'm just a computer program, but thanks for asking!"
        elif "what is your name" in user_text.lower():
            policy = "I am your virtual assistant."
        elif "time" in user_text.lower():
            policy = f"The current time is {time.strftime('%I:%M %p')}."
        else:
            policy = ""
            
        return policy

    def _save_audio_debug(self, pcm16: np.ndarray, sample_rate: int, transcript: str = ""):
        """
        Save the final utterance PCM to an mp3 file for debugging.
        Uses ffmpeg if available; silently skips on failure.
        """
        if pcm16.size == 0:
            return
        if not shutil.which("ffmpeg"):
            print("[DebugAudio] ffmpeg not found; skipping save.")
            return
        try:
            os.makedirs(self.transcript_audio_dir, exist_ok=True)
        except Exception as e:
            print(f"[DebugAudio] Could not create dir {self.transcript_audio_dir}: {e}")
            return

        ts = time.strftime("%Y%m%d-%H%M%S")
        words = []
        if transcript:
            for w in transcript.strip().split():
                clean = "".join(ch for ch in w if ch.isalnum() or ch in ("-", "_"))
                if clean:
                    words.append(clean.lower())
                if len(words) >= 2:
                    break
        suffix = ""
        if words:
            suffix = "_" + "-".join(words)

        fname = f"chunk_{ts}_{int(time.time()*1000)%1000:03d}{suffix}.mp3"
        out_path = os.path.join(self.transcript_audio_dir, fname)
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "s16le",
                    "-ar", str(sample_rate),
                    "-ac", "1",
                    "-i", "-",
                    "-codec:a", "libmp3lame",
                    out_path,
                ],
                input=pcm16.tobytes(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.returncode != 0:
                print(f"[DebugAudio] ffmpeg encode failed (code {proc.returncode}).")
            else:
                print(f"[DebugAudio] Saved utterance to {out_path}")
        except FileNotFoundError:
            print("[DebugAudio] ffmpeg not found when attempting to save.")
        except Exception as e:
            print(f"[DebugAudio] Failed to save audio: {e}")

    def _run_loop(self):
        self.mic.start()
        self.tts_player.start()

        utterance_frames: list[np.ndarray] = []
        silence_ms_accum = 0
        preroll_frames = deque(maxlen=PRE_SPEECH_FRAMES)

        utterance_ms = 0
        had_any_speech = False

        session_active = False
        last_activity_ms = 0  # updated on every user-speech VAD=true

        try:
            # calibrate VAD noise floor if "webrtcVAD" backend in use
            if VAD_BACKEND == "webrtc":
                print("[VAD] Calibrating noise floor, please be silent...")
                for _ in range(VAD_CALIBRATION_FRAMES):
                    if self._stop_event.is_set():
                        break
                    frame = self.mic.read_frame()
                    self.vad.is_speech(frame)
            if self.ww.labels:
                print(f"[System] Ready: say the wake word {self.ww.labels} or call trigger_wakeword().")
            else:
                print(f"[System] Ready: Wakeword disabled. Call trigger_wakeword() to start listening.")
            while not self._stop_event.is_set():
                cap = self.mic.read_frame()  # int16
                clean = cap  # raw audio, no AEC
                clean_f32 = to_float32(clean)

                # ========================= SPEAKING mode =========================
                if self.state == self.SPEAKING:
                    last_activity_ms = int(time.time() * 1000.0)

                    # If playback finished
                    if not self.tts_player.is_playing():
                        # return to LISTENING if session is active, else idle
                        if session_active:
                            self._set_state(self.LISTENING)
                            utterance_frames = []
                            self.mic.drain()  # drop any frames captured during TTS playback
                            silence_ms_accum = 0
                            preroll_frames.clear()
                            utterance_ms = 0
                            had_any_speech = False
                        else:
                            self._set_state(self.IDLE)

                    continue

                # Check programmatic trigger
                forced = False
                if self.state == self.IDLE:
                    try:
                        self.system_trigger_q.get_nowait()
                        forced = True
                    except queue.Empty:
                        forced = False

                # ========================== IDLE mode ==========================
                if self.state == self.IDLE:
                    ww_hit = False
                    # Check for wake word
                    if self.ww.enabled:
                        ema, fired = self.ww.step(clean_f32)
                        # print(f"\r[WakeWord] ema: {ema:0.3f}   ", end="", flush=True)
                        ww_hit = fired

                    if ww_hit or forced:
                        print(f"\n[System] Triggered by {'wake word' if ww_hit else 'system'}")
                        session_active = True
                        self._set_state(self.LISTENING)
                        utterance_frames = []
                        self.mic.drain()  # ensure no stale frames from previous state
                        silence_ms_accum = 0
                        preroll_frames.clear()
                        utterance_ms = 0
                        had_any_speech = False
                        last_activity_ms = int(time.time() * 1000.0)
                        continue


                # ===================== TRANSCRIBING mode =====================
                if self.state == self.TRANSCRIBING:
                    total_ms = utterance_ms  # fallback in case safety cap forced endpoint
                    pcm_for_asr = np.concatenate(utterance_frames) if utterance_frames else np.array([], dtype=np.int16)
                    print(f"[Voice] Transcribing… ({total_ms} ms)")
                    text = self.asr.transcribe_blocking(pcm_for_asr, timeout_s=5.0)  # isolated in a worker
                    print(f"[User]: {text}")
                    try:
                        if self.on_transcript:
                            self.on_transcript(text)
                    except Exception as e:
                        print(f"[TranscriptCallback] Error: {e}")
                    last_activity_ms = int(time.time() * 1000.0)
                    if self.save_transcript_audio:
                        self._save_audio_debug(pcm_for_asr, SAMPLE_RATE, transcript=text)

                    reply = self.policy(text)
                    if len(reply) > 0:
                        self.speak(reply) # Sets state to SPEAKING inside
                    else:
                        if session_active:
                            self._set_state(self.LISTENING)
                        else:
                            self._set_state(self.IDLE)
                        
                    utterance_frames = []
                    silence_ms_accum = 0
                    preroll_frames.clear()
                    utterance_ms = 0
                    had_any_speech = False
                    self.vad.reset()
                    self.mic.drain()

                    continue

                # ======================== LISTENING / RECORDING mode ======================
                if self.state in (self.LISTENING, self.RECORDING):
                    
                    if self.vad.is_speech(clean):
                        if not had_any_speech and preroll_frames:
                            utterance_frames.extend(preroll_frames)
                            preroll_frames.clear()
                        utterance_frames.append(clean.copy())
                        utterance_ms += CAPTURE_FRAME_MS

                        silence_ms_accum = 0
                        last_activity_ms = int(time.time() * 1000.0)

                        if not had_any_speech:
                            had_any_speech = True
                            self._set_state(self.RECORDING)
                        
                    else:
                        if had_any_speech:
                            # We are in the middle of an utterance: KEEP buffering silence frames.
                            utterance_frames.append(clean.copy())
                            utterance_ms += CAPTURE_FRAME_MS
                            silence_ms_accum += CAPTURE_FRAME_MS
                            if utterance_frames and silence_ms_accum >= self.silence_ms_after_speech:
                                # endpoint conditions met
                                total_ms = utterance_ms
                                if total_ms < MIN_SPEECH_MS:
                                    utterance_frames = []
                                    silence_ms_accum = 0
                                    utterance_ms = 0
                                    had_any_speech = False
                                    self._set_state(self.LISTENING)
                                    self.vad.reset()
                                    self.mic.drain()
                                else:
                                    self._set_state(self.TRANSCRIBING)
                        else:
                            # build a small pre-roll before first VAD hit
                            if PRE_SPEECH_FRAMES > 0:
                                preroll_frames.append(clean.copy())

                    
                    # Force endpoint if user keeps talking forever (safety cap)
                    if self.state in (self.LISTENING, self.RECORDING) and utterance_ms >= MAX_LISTEN_MS and had_any_speech:
                        self._set_state(self.TRANSCRIBING)


                # --- session-wide inactivity timeout (works even if user never spoke) ---
                if session_active and self.state != self.SPEAKING:
                    now_ms = int(time.time() * 1000.0)
                    if (now_ms - last_activity_ms) >= LISTEN_NO_SPEECH_TIMEOUT_MS:
                        print("[System] Inactivity timeout.")
                        session_active = False
                        self._set_state(self.IDLE)
                        utterance_frames = []
                        utterance_ms = 0
                        had_any_speech = False
                        self.vad.reset()
                        self.mic.drain()
                        # (optional) tiny guard so the same WW window doesn't immediately refire
                        # time.sleep(0.05)
                        continue

                time.sleep(0.001)

        finally:
            self.mic.stop()

# ---------------- Main ----------------
def main():
    transcript_q: queue.Queue[str] = queue.Queue()
    status_q: queue.Queue[str] = queue.Queue()

    def _on_transcript(text: str):
        transcript_q.put(text)

    def _on_status(status: str):
        status_q.put(status)

    dm = DialogManager(
        model=WHISPER_MODEL,
        on_transcript=_on_transcript,
        on_status=_on_status,
        wakeword_keywords="hey_jarvis",
        wakeword_model_paths= None,
        vad_silero_threshold = 0.5
    )
    dm.start()

    try:
        while True:
            try:
                status = status_q.get_nowait()
                # print(f"[Status]: {status}")
                pass
            except queue.Empty:
                pass
            try:
                text = transcript_q.get_nowait()
                if text:
                    # print(f"[Transcript]: {text}")
                    pass
            except queue.Empty:
                pass
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        dm.stop()

def cli():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

if __name__ == "__main__":
    cli()
