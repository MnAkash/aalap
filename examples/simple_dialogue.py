import multiprocessing as mp
import queue
import time

from aalap.dialogue_manager import DialogManager


def main():
    transcript_q: queue.Queue[str] = queue.Queue()
    status_q: queue.Queue[str] = queue.Queue()

    def _on_transcript(text: str):
        transcript_q.put(text)

    def _on_status(status: str):
        status_q.put(status)

    def _my_policy(user_text: str) -> str:
        # do API/LLM call here, blocking is fine
        time.sleep(1.0)  # simulate thinking time
        return f"You said: {user_text}"

    dm = DialogManager(
        model="base.en",
        device="auto",
        tts_backend = "piper", #or "gtts" (online)
        on_transcript=_on_transcript,
        on_status=_on_status,
        external_policy=_my_policy,
        wakeword_keywords="hey_jarvis",
        wakeword_model_paths=None,
        vad_silero_threshold=0.5,
    )
    dm.start()

    try:
        while True:
            try:
                _ = status_q.get_nowait()
            except queue.Empty:
                pass
            try:
                _ = transcript_q.get_nowait()
            except queue.Empty:
                pass
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        dm.stop()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
