import io
import numpy as np
from gtts import gTTS
from pydub import AudioSegment


class GTtsTTS:
    """
    Google Text-to-Speech wrapper that returns int16 PCM at 16 kHz.
    Uses gTTS to synthesize to MP3 in-memory, then decodes via pydub.
    """
    def __init__(self, language: str = "en", tld: str = "com", slow: bool = False):
        self.language = language
        self.tld = tld
        self.slow = slow

    @staticmethod
    def _to_int16(segment: AudioSegment) -> np.ndarray:
        samples = np.array(segment.get_array_of_samples()).astype(np.int16)
        if segment.channels > 1:
            samples = samples.reshape((-1, segment.channels)).mean(axis=1).astype(np.int16)
        return samples

    def synth(self, text: str) -> np.ndarray:
        tts = gTTS(text=text, lang=self.language, tld=self.tld, slow=self.slow)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio = AudioSegment.from_file(buf, format="mp3")
        # resample to 16 kHz mono int16
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        return self._to_int16(audio)

