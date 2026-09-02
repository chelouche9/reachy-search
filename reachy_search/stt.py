"""Speech to text, locally, with faster-whisper.

Local on purpose: it keeps the app's install down to two API keys, and it means
the microphone audio never leaves the robot.
"""

import logging
import threading

import numpy as np

from . import audio_utils

logger = logging.getLogger(__name__)


class Transcriber:
    """Lazily-loaded Whisper. The model download happens on first use."""

    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        self._model = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        """Load the model ahead of time so the first question isn't slow."""
        self._ensure_model()

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                logger.info("Loading Whisper (%s)...", self._model_size)
                # int8 on CPU: the quality cost is invisible for a one-sentence
                # question, and it is what makes this viable on the CM4.
                self._model = WhisperModel(
                    self._model_size, device="cpu", compute_type="int8"
                )
                logger.info("Whisper ready.")
        return self._model

    def transcribe(self, audio: np.ndarray, source_rate: int) -> str:
        mono = audio_utils.to_mono(audio)
        resampled = audio_utils.resample(mono, source_rate, audio_utils.WHISPER_RATE)

        if resampled.size < audio_utils.WHISPER_RATE * 0.2:
            return ""

        model = self._ensure_model()
        segments, _ = model.transcribe(
            resampled,
            language="en",
            beam_size=1,          # greedy: one short sentence, and speed matters
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info("Heard: %r", text)
        return text
