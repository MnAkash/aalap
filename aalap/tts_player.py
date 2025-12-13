
import threading
import queue
import numpy as np
import sounddevice as sd

class TTSPlayer:
    """
    Output playback via sounddevice (PortAudio), same backend as mic.
    """
    def __init__(self, device=None, sample_rate=16000, capture_frame_samples=320):
        self._playing = threading.Event()
        self._stop = threading.Event()
        self._q = queue.Queue(maxsize=4096)     # bigger jitter buffer
        self._frames_left = 0                   # how many real frames remain
        self._silence_runs = 0                  # consecutive silent callbacks
        self.capture_frame_samples = capture_frame_samples
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=capture_frame_samples,    # keep 20 ms granularity
            device=device,
            latency=0.35,                   # extra slack to reduce underruns
            callback=self._cb,
        )

    # ---- internal callback ----
    def _cb(self, outdata, frames, time_info, status):
        if status:
            # print(status)  # uncomment for debugging
            pass

        try:
            frame = self._q.get_nowait()
            had_real = True
        except queue.Empty:
            had_real = False
            frame = np.zeros(self.capture_frame_samples, dtype=np.int16)

        # pad/trim
        if len(frame) < self.capture_frame_samples:
            pad = np.zeros(self.capture_frame_samples - len(frame), dtype=np.int16)
            frame = np.concatenate([frame, pad])
        elif len(frame) > self.capture_frame_samples:
            frame = frame[:self.capture_frame_samples]

        outdata[:, 0] = frame

        # playback state machine
        if self._stop.is_set():
            # drain and stop
            while not self._q.empty():
                try: self._q.get_nowait()
                except queue.Empty: break
            self._frames_left = 0
            self._playing.clear()
            self._stop.clear()
            self._silence_runs = 0
            return

        if had_real:
            # only count down on real frames we emitted
            if self._frames_left > 0:
                self._frames_left -= 1
            self._silence_runs = 0
        else:
            # no frame available -> silent; keep a small counter
            self._silence_runs += 1
            # if we've emitted all frames and we see a bit of silence, mark not playing
            if self._frames_left <= 0 and self._silence_runs >= 10:  # ~200 ms grace
                self._playing.clear()

    # ---- public API ----
    def is_playing(self) -> bool:
        return self._playing.is_set()
        # print("Speaking status:", (not self._stop.is_set()), self._frames_left)
        # return (not self._stop.is_set()) and (self._frames_left > 0)

    def start(self):
        self._stream.start()

    def stop(self):
        # print("[TTSPlayer] Stopping playback...")
        self._stop.set()
        self._playing.clear()

    def play(self, pcm16: np.ndarray):
        """
        Non-blocking enqueue + preroll. Splits into 20 ms frames and queues them.
        """
        self._playing.set()
        # split
        frames = [pcm16[i:i + self.capture_frame_samples]
                  for i in range(0, len(pcm16), self.capture_frame_samples)]

        # small preroll (3 silent frames = 60 ms) to reduce ALSA underruns
        preroll = [np.zeros(self.capture_frame_samples, dtype=np.int16) for _ in range(3)]
        for f in preroll + frames:
            try:
                self._q.put_nowait(f)
            except queue.Full:
                try: _ = self._q.get_nowait()
                except queue.Empty: pass
                try: self._q.put_nowait(f)
                except queue.Full: pass

        self._frames_left = len(frames)           # only “real” frames count
        self._silence_runs = 0

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass