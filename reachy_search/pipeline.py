"""Frame plus question in, a sentence to say out loud out.

Everything here blocks — models load, HTTP happens, Whisper runs. It is called
only from the worker thread. The control loop in `main.py` never enters this
module, which is what keeps the motion smooth while the network is slow.
"""

import logging
from dataclasses import dataclass

import numpy as np

from . import brain, search, stt, tts

logger = logging.getLogger(__name__)


@dataclass
class Outcome:
    """What the robot should say, and whether it counts as a success."""

    spoken: str
    ok: bool = True


# Spoken failure lines. Never leave the robot silent — silence reads as a crash,
# and this thing is meant to be charming when it fails.
NOT_HEARD = "I didn't catch that. Try again?"
NO_KEYS = "I need my API keys first. Open my settings page to add them."
REFUSED = "I'd rather not answer that one."
BROKE = "Something went wrong on my end. Give it another go?"


class Pipeline:
    def __init__(self, settings, transcriber=None, speaker=None, on_search=None):
        self._settings = settings
        self._searcher = search.Searcher(settings.tavily_api_key)
        self._brain = brain.Brain(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
            searcher=self._searcher,
            on_search=on_search,
        )
        # Shared with the app when it owns the voice stack (wake word and
        # prompts need voice even before API keys exist).
        self.transcriber = transcriber or stt.Transcriber(settings.whisper_model)
        self.speaker = speaker or tts.Speaker(settings.voice)

    def warm_up(self) -> None:
        """Load Whisper and Piper up front, so question one isn't the slow one."""
        try:
            self.transcriber.warm_up()
            self.speaker.warm_up()
        except Exception:
            logger.exception("Warm-up failed; models will load on first use")

    def answer(
        self,
        jpeg: bytes | None,
        audio: np.ndarray | None = None,
        samplerate: int = 0,
        question: str | None = None,
    ) -> Outcome:
        """`question` may arrive pre-transcribed (one-shot wake phrase);
        otherwise it is transcribed from `audio`. The agent decides for itself
        whether to search, so a missing frame or an unsearchable question are
        its problems to route around, not ours."""
        if question is None:
            if audio is None:
                return Outcome(NOT_HEARD, ok=False)
            try:
                question = self.transcriber.transcribe(audio, samplerate)
            except Exception:
                logger.exception("Transcription failed")
                return Outcome(BROKE, ok=False)

        if not question:
            return Outcome(NOT_HEARD, ok=False)

        try:
            spoken = self._brain.respond(jpeg, question)
        except brain.RefusalError as exc:
            logger.warning("Agent refused (%s)", exc)
            return Outcome(REFUSED, ok=False)
        except Exception:
            logger.exception("Agent call failed")
            return Outcome(BROKE, ok=False)

        return Outcome(spoken or BROKE, ok=bool(spoken))
