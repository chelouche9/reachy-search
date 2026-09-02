"""Text to speech, locally, with Piper.

Piper is built for Pi-class hardware, which is what makes the Wireless robot
viable. Voices download once, on demand, into the user's cache directory.

macOS caveat: piper-tts 1.7.0's bundled espeak-ng on mac arm64 has a baked-in
build path and *exits the process* on first phonemization. We probe for that in
a throwaway subprocess and, when broken, fall back to the system `say` command
for development. The robot itself runs Linux, where Piper works.
"""

import logging
import os
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

from . import audio_utils

logger = logging.getLogger(__name__)

VOICE_DIR = Path(
    os.environ.get("REACHY_SEARCH_VOICE_DIR")
    or Path.home() / ".cache" / "reachy_search" / "voices"
)
DEFAULT_VOICE = "en_US-amy-medium"
SAY_RATE = 22_050

_espeak_ok: bool | None = None


def _piper_espeak_ok() -> bool:
    """Probe piper's espeak in a subprocess — when broken it exit(1)s, and we
    would rather sacrifice a child process than the app."""
    global _espeak_ok
    if _espeak_ok is None:
        if sys.platform != "darwin":
            _espeak_ok = True
        else:
            probe = subprocess.run(
                [sys.executable, "-c",
                 "from piper.phonemize_espeak import EspeakPhonemizer, ESPEAK_DATA_DIR;"
                 "EspeakPhonemizer(ESPEAK_DATA_DIR).phonemize('en-us', 'hi')"],
                capture_output=True, timeout=60,
            )
            _espeak_ok = probe.returncode == 0
            if not _espeak_ok:
                logger.warning(
                    "Piper's espeak is broken on this Mac (known wheel issue); "
                    "using macOS 'say' for development. The robot uses Piper."
                )
    return _espeak_ok


class Speaker:
    """Wraps a Piper voice, or the macOS `say` fallback. Thread-confined to the
    pipeline worker."""

    def __init__(self, voice: str = DEFAULT_VOICE):
        self._voice_name = voice
        self._voice = None
        self._use_say = False
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        self._ensure_voice()
        if self._use_say:
            return SAY_RATE
        return self._voice.config.sample_rate

    def warm_up(self) -> None:
        self._ensure_voice()

    def _ensure_voice(self):
        with self._lock:
            if self._voice is not None or self._use_say:
                return self._voice
            if not _piper_espeak_ok():
                self._use_say = True
                return None

            from piper import PiperVoice
            from piper.download_voices import download_voice

            VOICE_DIR.mkdir(parents=True, exist_ok=True)
            model_path = VOICE_DIR / f"{self._voice_name}.onnx"
            if not model_path.exists():
                logger.info("Downloading Piper voice %s...", self._voice_name)
                download_voice(self._voice_name, VOICE_DIR)

            logger.info("Loading Piper voice %s", self._voice_name)
            self._voice = PiperVoice.load(model_path)
        return self._voice

    def _say_to_array(self, text: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.wav"
            subprocess.run(
                ["say", "-o", str(path), f"--data-format=LEI16@{SAY_RATE}", text],
                check=True, capture_output=True, timeout=120,
            )
            with wave.open(str(path), "rb") as wav_file:
                raw = wav_file.readframes(wav_file.getnframes())
                channels = wav_file.getnchannels()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio.astype(np.float32)

    def synthesize(self, text: str):
        """Yield float32 mono chunks at `sample_rate`."""
        voice = self._ensure_voice()
        if self._use_say:
            yield self._say_to_array(text)
            return
        for chunk in voice.synthesize(text):
            yield audio_utils.int16_to_float32(chunk.audio_int16_array)

    def synthesize_to_array(self, text: str) -> np.ndarray:
        chunks = list(self.synthesize(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def synthesize_to_wav(self, text: str, path: Path) -> Path:
        """Used by the robot-free dev harness, which has no push_audio_sample."""
        audio = self.synthesize_to_array(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
        return path
