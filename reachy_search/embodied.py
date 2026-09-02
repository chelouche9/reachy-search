"""Embodied search for other Reachy Mini apps.

Raw web search is one HTTP call — that is not what this package offers. This
is search as a robot behavior: ask a question with a camera frame and the
robot visibly thinks (the drumroll move, the processing chirps, a spoken
"Searching..."), then pops up and speaks the answer.

    from reachy_search import EmbodiedSearch

    skill = EmbodiedSearch(mini, anthropic_api_key=..., tavily_api_key=...)
    answer = skill.ask("find me a cheaper one")   # blocks, performs, speaks

Rules of the road:
- `ask()` drives `set_target` for the duration of the call. Pause your own
  control loop while it runs (the SDK's one-writer rule applies).
- Call `mini.enable_wobbling()` once at startup if you want the head to move
  with the voice — that part is daemon-side and free.
- `speak=False` / `animate=False` give you just the answer text.
"""

import logging
import random
import threading
import time

import numpy as np

from . import audio_utils, brain, config, moves, search, sounds, tts

logger = logging.getLogger(__name__)


class EmbodiedSearch:
    def __init__(
        self,
        mini,
        anthropic_api_key: str,
        tavily_api_key: str,
        voice: str = tts.DEFAULT_VOICE,
        model: str = config.CLAUDE_MODEL,
        speak: bool = True,
        animate: bool = True,
    ):
        self._mini = mini
        self._speak = speak
        self._animate = animate
        self._speaker = tts.Speaker(voice) if speak else None
        self._search_clips: list[np.ndarray] = []
        self._brain = brain.Brain(
            api_key=anthropic_api_key,
            model=model,
            searcher=search.Searcher(tavily_api_key),
            on_search=self._announce_search if speak else None,
        )

    def warm_up(self) -> None:
        """Optional: pre-load the voice and pre-render the search lines."""
        if self._speaker is not None:
            self._speaker.warm_up()
            self._search_clips = [
                self._speaker.synthesize_to_array(line)
                for line in config.SEARCH_LINES
            ]

    def ask(self, question: str, frame: bytes | None = None) -> str:
        """The whole performance. Returns the spoken text either way."""
        if frame is None:
            try:
                frame = self._mini.media.get_frame_jpeg()
            except Exception:
                logger.warning("No camera frame available", exc_info=True)

        stop_thinking = threading.Event()
        if self._animate:
            performer = threading.Thread(
                target=self._perform_thinking, args=(stop_thinking,), daemon=True
            )
            performer.start()

        try:
            answer = self._brain.respond(frame, question)
        finally:
            stop_thinking.set()
            if self._animate:
                performer.join(timeout=1.0)

        if answer and self._speak:
            self._perform_answer(answer)
        return answer

    # ------------------------------------------------------------------

    def _perform_thinking(self, stop: threading.Event) -> None:
        """Drumroll plus processing chirps until the answer arrives."""
        media = self._mini.media
        out_rate = media.get_output_audio_samplerate()
        loop = sounds.searching_loop(out_rate) if out_rate else None
        slice_n = int(out_rate * 0.3) if out_rate else 0
        position = 0
        next_push = 0.0

        t0 = time.monotonic()
        while not stop.is_set():
            t = time.monotonic() - t0
            frame = moves.drumroll(t)
            self._mini.set_target(
                head=frame.head_pose(), antennas=frame.antennas_rad(),
                body_yaw=frame.body_yaw_rad(),
            )
            if loop is not None and t >= next_push and not self._voice_playing:
                chunk = loop[position:position + slice_n]
                position = position + slice_n if chunk.size == slice_n else 0
                if chunk.size == slice_n:
                    media.push_audio_sample(chunk)
                next_push = t + 0.28
            time.sleep(0.02)

    _voice_playing = False

    def _announce_search(self, query: str) -> None:
        if not self._search_clips:
            return
        media = self._mini.media
        out_rate = media.get_output_audio_samplerate()
        clip = random.choice(self._search_clips)
        resampled = audio_utils.resample(clip, self._speaker.sample_rate, out_rate)
        self._voice_playing = True
        media.push_audio_sample(resampled)
        threading.Timer(resampled.size / max(out_rate, 1) + 0.2,
                        lambda: setattr(self, "_voice_playing", False)).start()

    def _perform_answer(self, text: str) -> None:
        """The aha pop, then speak, swaying gently until the audio is done."""
        media = self._mini.media
        out_rate = media.get_output_audio_samplerate()
        pushed = 0
        t0 = time.monotonic()

        done = threading.Event()

        def synthesize():
            nonlocal pushed
            try:
                for chunk in self._speaker.synthesize(text):
                    resampled = audio_utils.resample(
                        chunk, self._speaker.sample_rate, out_rate)
                    media.push_audio_sample(resampled)
                    pushed += resampled.size
            except Exception:
                logger.exception("Speech synthesis failed")
            finally:
                done.set()

        threading.Thread(target=synthesize, daemon=True).start()

        while True:
            t = time.monotonic() - t0
            if self._animate:
                frame = moves.speaking(t)
                self._mini.set_target(
                    head=frame.head_pose(), antennas=frame.antennas_rad(),
                    body_yaw=frame.body_yaw_rad(),
                )
            if done.is_set() and t >= pushed / max(out_rate, 1):
                return
            time.sleep(0.02)
