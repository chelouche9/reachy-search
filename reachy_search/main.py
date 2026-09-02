"""Reachy Search — say "hey Reachy", ask about something, get an answer out loud.

Triggers, all equivalent: the wake word while idle, a boop on an antenna, or
the Ask button on the settings page. The wake path adds one beat: Reachy perks
up and asks "What can I do for you?" before listening.

Threading follows the SDK's rule: exactly one control loop, in exactly one
place, calling `set_target`. It runs at 50 Hz and never blocks. Everything slow
— wake listening, recording, Whisper, two Claude calls, Tavily, Piper — happens
on one worker thread, which is also the only thread that touches the mic
buffer. The two talk through a state enum and a job queue, and that separation
is why the animation stays smooth while the network does not.
"""

import logging
import logging.handlers
import os
import queue
import re
import threading
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp

from . import audio_utils, camera, config, moves, pipeline, sounds, stt, tts

LOG_PATH = Path.home() / ".cache" / "reachy_search" / "app.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Mirror to a file so the log survives the terminal and can be tailed
        # from anywhere (~1MB x 2 rotations).
        logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000,
                                             backupCount=2),
    ],
)
logger = logging.getLogger("reachy_search")

CONTROL_HZ = 50.0
TICK = 1.0 / CONTROL_HZ
ANTENNA_POLL_EVERY = 4      # ticks; the position read is a round-trip
FADE_S = 0.35               # cross-fade between states
ERROR_HOLD_S = 1.6
GREET_S = 2.0

_WORDS = re.compile(r"[a-z']+")


class State(Enum):
    STARTING = auto()
    IDLE = auto()
    ATTENTIVE = auto()   # awake window: follow-ups need no wake word
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()


def match_wake(text: str) -> str | None:
    """Return what followed the wake word, "" if nothing did, None if no wake.

    Whisper mangles "Reachy" freely, so we match a family of spellings.
    "Hey Reachy, search for moka pots" -> "search for moka pots" (one-shot).
    """
    words = _WORDS.findall(text.lower())
    for i, word in enumerate(words):
        if word in config.WAKE_WORDS:
            rest = " ".join(words[i + 1:])
            # A word or two after the name is noise ("hey reachy there");
            # a real one-shot question has some meat on it.
            return rest if len(rest.split()) >= 3 else ""
    return None


class ReachySearch(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # Default media backend (camera + mic). Override per-run when an environment
    # has broken media — e.g. REACHY_SEARCH_MEDIA=no_media against the
    # simulator, where the app then runs motion-only.
    request_media_backend: str | None = os.environ.get("REACHY_SEARCH_MEDIA") or None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        self._mini = reachy_mini
        self._stop_event = stop_event
        self._settings = config.load()
        self._frames = camera.FrameSource(reachy_mini)
        # The app owns the voice stack so the wake word and spoken prompts work
        # even before any API key exists; the pipeline borrows these.
        self._transcriber = stt.Transcriber(self._settings.whisper_model)
        self._speaker = tts.Speaker(self._settings.voice)
        self._prompt_audio: np.ndarray | None = None
        self._search_clips: list[np.ndarray] = []
        self._voice_active = False  # ambient beeps yield to any speech
        self._pipeline: pipeline.Pipeline | None = None
        self._pipeline_lock = threading.Lock()

        self._state = State.STARTING
        self._state_t0 = time.monotonic()
        self._thinking_name, self._thinking_move = moves.pick_thinking_move()
        self._last_commanded = np.zeros(2)
        self._ask_requested = False
        self._mic_stale = True
        self._jobs: queue.Queue = queue.Queue(maxsize=1)

        self._mount_settings_ui()

        logger.info("Reachy Search starting (keys: %s, wake word: %s)",
                    "ok" if self._settings.ready else
                    f"missing {', '.join(self._settings.missing())}",
                    "on" if config.WAKE_ENABLED else "off")

        reachy_mini.media.start_recording()
        reachy_mini.media.start_playing()
        # Speech-driven head motion, composed onto our target pose daemon-side.
        # Free personality while the robot talks.
        try:
            reachy_mini.enable_wobbling()
        except Exception:
            logger.warning("Head wobbling unavailable", exc_info=True)

        worker = threading.Thread(target=self._worker, name="interaction", daemon=True)
        worker.start()
        threading.Thread(target=self._ambient_loop, name="ambient", daemon=True).start()
        threading.Thread(target=self._warm_up, name="warmup", daemon=True).start()

        try:
            self._control_loop()
        finally:
            try:
                self._jobs.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=3.0)
            self._frames.close()
            try:
                reachy_mini.disable_wobbling()
                reachy_mini.media.stop_recording()
                reachy_mini.media.stop_playing()
            except Exception:
                logger.debug("Teardown noise", exc_info=True)
            logger.info("Reachy Search stopped.")

    def _warm_up(self) -> None:
        """Load Whisper and Piper in the background, and pre-render the prompt
        line so the wake word gets an instant response."""
        try:
            self._transcriber.warm_up()
            self._speaker.warm_up()
            self._prompt_audio = self._speaker.synthesize_to_array(config.PROMPT_TEXT)
            self._search_clips = [self._speaker.synthesize_to_array(line)
                                  for line in config.SEARCH_LINES]
            logger.info("Models warm.")
        except Exception:
            logger.exception("Warm-up failed; models will load on first use")
        self._ensure_pipeline()

    def _ensure_pipeline(self) -> pipeline.Pipeline | None:
        """Built lazily — keys may arrive from the settings page after startup."""
        if not self._settings.ready:
            return None
        with self._pipeline_lock:
            if self._pipeline is None:
                self._pipeline = pipeline.Pipeline(
                    self._settings, self._transcriber, self._speaker,
                    on_search=self._announce_search,
                )
        return self._pipeline

    # ------------------------------------------------------------------
    # Control loop — the only place that calls set_target
    # ------------------------------------------------------------------

    def _set_state(self, state: State) -> None:
        if state is self._state:
            return
        logger.debug("%s -> %s", self._state.name, state.name)
        self._fade_from = self._last_frame
        self._state = state
        self._state_t0 = time.monotonic()

    def _frame_for(self, state: State, t: float) -> moves.Frame:
        if state is State.STARTING:
            return moves.greet(t)
        if state is State.IDLE:
            return moves.idle(t)
        if state is State.ATTENTIVE:
            return moves.attentive(t)
        if state is State.LISTENING:
            return moves.listening(t)
        if state is State.THINKING:
            return self._thinking_move(t)
        if state is State.SPEAKING:
            return moves.speaking(t)
        if state is State.ERROR:
            return moves.error_shake(t)
        return moves.REST

    def _control_loop(self) -> None:
        self._last_frame = moves.REST
        self._fade_from = moves.REST
        press_ticks = 0
        cooldown_until = 0.0
        tick = 0

        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed = now - self._state_t0

            # Timed states retire themselves; the worker owns the rest.
            if self._state is State.STARTING and elapsed > GREET_S:
                self._set_state(State.IDLE)
                elapsed = 0.0
            elif self._state is State.ERROR and elapsed > ERROR_HOLD_S:
                # A failed exchange still leaves it awake — "didn't catch
                # that" followed by demanding its name again would be rude.
                self._set_state(State.ATTENTIVE)
                elapsed = 0.0
            elif self._state is State.ATTENTIVE and elapsed > config.AWAKE_WINDOW_S:
                logger.info("Awake window expired; back to sleep")
                self._set_state(State.IDLE)
                elapsed = 0.0

            frame = self._frame_for(self._state, elapsed)
            # Cross-fade out of whatever we were doing, so state changes read as
            # movement rather than as a cut.
            if elapsed < FADE_S:
                frame = moves.blend(self._fade_from, frame, elapsed / FADE_S)

            antennas = frame.antennas_rad()
            self._mini.set_target(head=frame.head_pose(), antennas=antennas,
                                  body_yaw=frame.body_yaw_rad())
            self._last_frame = frame
            self._last_commanded = antennas

            # The settings page's Ask button — same effect as an antenna boop.
            if self._state in (State.IDLE, State.ATTENTIVE) and self._ask_requested:
                self._ask_requested = False
                cooldown_until = now + config.ANTENNA_COOLDOWN_S
                self._begin_question()

            # Antenna-as-button, only while idle.
            tick += 1
            if (self._state in (State.IDLE, State.ATTENTIVE)
                    and now > cooldown_until
                    and tick % ANTENNA_POLL_EVERY == 0):
                if self._antenna_pushed():
                    press_ticks += 1
                    if press_ticks >= config.ANTENNA_PRESS_TICKS:
                        press_ticks = 0
                        cooldown_until = now + config.ANTENNA_COOLDOWN_S
                        self._begin_question()
                else:
                    press_ticks = 0

            time.sleep(TICK)

    def _antenna_pushed(self) -> bool:
        """Antennas run at low P gain, so a push shows up as a gap between
        where we told them to be and where they actually are."""
        try:
            present = self._mini.get_present_antenna_joint_positions()
        except Exception:
            return False
        if present is None or len(present) != 2:
            return False
        deviation = np.abs(np.asarray(present) - self._last_commanded)
        return bool(deviation.max() > config.ANTENNA_PRESS_RAD)

    def _begin_question(self) -> None:
        if self._jobs.full():
            return
        # Grab the frame now, while the head is still where the user aimed it
        # and before the listening move tilts it away.
        jpeg = self._grab_frame()
        self._set_state(State.LISTENING)
        self._jobs.put(("press", jpeg))

    def _grab_frame(self) -> bytes | None:
        return self._frames.grab()

    # ------------------------------------------------------------------
    # Worker thread — everything that blocks, and the only mic consumer
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._jobs.get(timeout=0.15)
            except queue.Empty:
                job = ()
            if job is None:
                return

            try:
                if job:
                    kind, jpeg = job
                    self._pick_loader()
                    logger.info("Question via %s (loader: %s)", kind, self._thinking_name)
                    self._run_interaction(jpeg=jpeg, question=None, prompt_first=False)
                elif self._state is State.ATTENTIVE:
                    heard = self._listen_attentive()
                    if heard is None:
                        continue
                    self._pick_loader()
                    logger.info("Follow-up heard (loader: %s)", self._thinking_name)
                    self._run_interaction(jpeg=None, question=heard or None,
                                          prompt_first=heard == "")
                elif config.WAKE_ENABLED and self._state is State.IDLE:
                    heard = self._listen_for_wake()
                    if heard is None:
                        continue
                    self._pick_loader()
                    one_shot = heard or None
                    logger.info("Wake word heard (one-shot: %s, loader: %s)",
                                bool(one_shot), self._thinking_name)
                    self._run_interaction(jpeg=None, question=one_shot,
                                          prompt_first=one_shot is None)
            except Exception:
                logger.exception("Interaction failed")
                self._say_and_settle(pipeline.BROKE, ok=False)

    def _pick_loader(self) -> None:
        self._thinking_name, self._thinking_move = moves.pick_thinking_move()

    def _listen_for_wake(self) -> str | None:
        """One short energy-gated capture; transcribe it only if someone spoke."""
        media = self._mini.media
        samplerate = media.get_input_audio_samplerate()
        if not samplerate:
            time.sleep(0.5)
            return None

        if self._mic_stale:
            self._drain_mic()
            self._mic_stale = False

        burst = audio_utils.record_until_silence(
            pull=media.get_audio_sample,
            samplerate=samplerate,
            stop_event=self._stop_event,
            max_seconds=config.WAKE_MAX_S,
            hang_seconds=config.WAKE_HANG_S,
            silence_rms=config.SILENCE_RMS,
            min_speech_seconds=config.WAKE_MIN_SPEECH_S,
            should_abort=lambda: self._state is not State.IDLE
                                 or not self._jobs.empty(),
        )
        if burst.size < samplerate * 0.3:
            return None
        loudest = audio_utils.peak_rms(burst, samplerate)
        if loudest < config.SILENCE_RMS:
            return None
        logger.debug("Wake burst: %.1fs, peak rms %.4f", burst.size / samplerate, loudest)

        text = self._transcriber.transcribe(burst, samplerate)
        return match_wake(text) if text else None

    def _listen_attentive(self) -> str | None:
        """While the awake window is open, any speech is a question — no name
        needed. Returns the question, "" for a bare "hey Reachy" (prompt
        again), or None for silence / a dismissal / window closed."""
        media = self._mini.media
        samplerate = media.get_input_audio_samplerate()
        if not samplerate:
            time.sleep(0.5)
            return None

        if self._mic_stale:
            self._drain_mic()
            self._mic_stale = False

        burst = audio_utils.record_until_silence(
            pull=media.get_audio_sample,
            samplerate=samplerate,
            stop_event=self._stop_event,
            max_seconds=config.MAX_QUESTION_S,
            hang_seconds=config.SILENCE_HANG_S,
            silence_rms=config.SILENCE_RMS,
            min_speech_seconds=config.MIN_SPEECH_S,
            should_abort=lambda: self._state is not State.ATTENTIVE
                                 or not self._jobs.empty(),
        )
        if self._state is not State.ATTENTIVE:
            return None
        if burst.size < samplerate * 0.3:
            return None
        if audio_utils.peak_rms(burst, samplerate) < config.SILENCE_RMS:
            return None

        text = self._transcriber.transcribe(burst, samplerate)
        if not text:
            return None

        normalized = " ".join(_WORDS.findall(text.lower()))
        if any(phrase in normalized for phrase in config.SLEEP_PHRASES):
            logger.info("Dismissed (%r); back to sleep", text)
            self._set_state(State.IDLE)
            return None
        # Saying the name mid-window is fine — strip it and use the rest.
        wake_rest = match_wake(text)
        if wake_rest is not None:
            return wake_rest
        # One stray word is more likely mic noise or a Whisper hallucination
        # ("thank you", "you") than a question worth interrupting someone for.
        if len(normalized.split()) < 2:
            return None
        return text

    def _run_interaction(self, jpeg: bytes | None, question: str | None,
                         prompt_first: bool) -> None:
        audio = None
        samplerate = 0

        if prompt_first and self._prompt_audio is not None:
            self._set_state(State.SPEAKING)
            self._push_audio(self._prompt_audio)

        if question is None:
            self._set_state(State.LISTENING)
            audio, samplerate = self._record_question()

        if jpeg is None:
            # Wake path: grab the frame after they finish talking — that is
            # when the object they raised is actually in front of the camera.
            jpeg = self._grab_frame()

        self._set_state(State.THINKING)

        engine = self._ensure_pipeline()
        if engine is None:
            self._say_and_settle(pipeline.NO_KEYS, ok=False)
            return

        outcome = engine.answer(jpeg, audio=audio, samplerate=samplerate,
                                question=question)
        self._say_and_settle(outcome.spoken, ok=outcome.ok)

    def _record_question(self) -> tuple[np.ndarray, int]:
        media = self._mini.media
        samplerate = media.get_input_audio_samplerate()
        self._drain_mic()

        audio = audio_utils.record_until_silence(
            pull=media.get_audio_sample,
            samplerate=samplerate,
            stop_event=self._stop_event,
            max_seconds=config.MAX_QUESTION_S,
            hang_seconds=config.SILENCE_HANG_S,
            silence_rms=config.SILENCE_RMS,
            min_speech_seconds=config.MIN_SPEECH_S,
        )
        logger.info("Recorded %.1fs of audio", audio.size / max(samplerate, 1))
        return audio, samplerate

    def _drain_mic(self) -> None:
        """Drop whatever accumulated while we weren't listening."""
        for _ in range(64):
            if self._mini.media.get_audio_sample() is None:
                break

    def _ambient_loop(self) -> None:
        """Soft processing chirps while thinking. Pushed in short slices so
        the sound stops within a beat of the answer arriving, and pauses
        whenever the voice has the floor."""
        if os.environ.get("REACHY_SEARCH_BEEPS", "1") == "0":
            return
        out_rate = self._mini.media.get_output_audio_samplerate()
        if not out_rate:
            return
        loop = sounds.searching_loop(out_rate)
        slice_n = int(out_rate * 0.3)
        position = 0
        while not self._stop_event.is_set():
            if self._state is State.THINKING and not self._voice_active:
                chunk = loop[position:position + slice_n]
                if chunk.size < slice_n:
                    position = 0
                    continue
                try:
                    self._mini.media.push_audio_sample(chunk)
                except Exception:
                    logger.debug("Ambient push failed", exc_info=True)
                    time.sleep(1.0)
                position += slice_n
                time.sleep(0.28)  # just under the slice, keeps the buffer shallow
            else:
                position = 0
                time.sleep(0.05)

    def _announce_search(self, query: str) -> None:
        """Called from inside the agent loop as the search tool fires. Push a
        pre-rendered line and return immediately — the search runs while the
        voice plays over the thinking move."""
        if not self._search_clips:
            return
        import random as _random
        clip = _random.choice(self._search_clips)
        out_rate = self._mini.media.get_output_audio_samplerate()
        resampled = audio_utils.resample(clip, self._speaker.sample_rate, out_rate)
        self._voice_active = True
        self._mini.media.push_audio_sample(resampled)
        # Hold the floor for the line's duration, then let the beeps resume.
        threading.Timer(resampled.size / max(out_rate, 1) + 0.2,
                        lambda: setattr(self, "_voice_active", False)).start()

    def _say_and_settle(self, text: str, ok: bool = True) -> None:
        if text:
            self._set_state(State.SPEAKING)
            self._speak(text)
        self._mic_stale = True
        self._set_state(State.ATTENTIVE if ok else State.ERROR)

    def _push_audio(self, audio: np.ndarray) -> None:
        """Push one pre-rendered utterance and hold until it has been heard."""
        media = self._mini.media
        out_rate = media.get_output_audio_samplerate()
        resampled = audio_utils.resample(audio, self._speaker.sample_rate, out_rate)
        self._voice_active = True
        try:
            media.push_audio_sample(resampled)
            duration = resampled.size / max(out_rate, 1)
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline and not self._stop_event.is_set():
                time.sleep(0.05)
        finally:
            self._voice_active = False

    def _speak(self, text: str) -> None:
        media = self._mini.media
        out_rate = media.get_output_audio_samplerate()
        voice_rate = self._speaker.sample_rate
        pushed_samples = 0
        started = time.monotonic()

        self._voice_active = True
        try:
            for chunk in self._speaker.synthesize(text):
                if self._stop_event.is_set():
                    return
                resampled = audio_utils.resample(chunk, voice_rate, out_rate)
                media.push_audio_sample(resampled)
                pushed_samples += resampled.size
        except Exception:
            logger.exception("Speech synthesis failed")
            return

        # Synthesis outruns real time, so the buffer drains long after the last
        # push. Hold the speaking animation for the audio's actual duration.
        try:
            duration = pushed_samples / max(out_rate, 1)
            while (time.monotonic() - started) < duration:
                if self._stop_event.is_set():
                    return
                time.sleep(0.05)
        finally:
            self._voice_active = False

    # ------------------------------------------------------------------
    # Settings page — API keys and a manual trigger, no terminal needed
    # ------------------------------------------------------------------

    def _mount_settings_ui(self) -> None:
        class Keys(BaseModel):
            anthropic_api_key: str = ""
            tavily_api_key: str = ""
            voice: str = ""

        @self.settings_app.get("/api/status")
        def status():
            return {
                "ready": self._settings.ready,
                "missing": self._settings.missing(),
                "state": self._state.name,
                "voice": self._settings.voice,
                "model": self._settings.claude_model,
                "wake_word": config.WAKE_ENABLED,
                # Never echo the keys back — only whether they are set.
                "anthropic_set": bool(self._settings.anthropic_api_key),
                "tavily_set": bool(self._settings.tavily_api_key),
            }

        @self.settings_app.post("/api/ask")
        def ask():
            if self._state is not State.IDLE:
                return {"accepted": False, "reason": self._state.name.lower()}
            self._ask_requested = True
            return {"accepted": True, "reason": ""}

        @self.settings_app.post("/api/keys")
        def set_keys(keys: Keys):
            self._settings.update(
                anthropic_api_key=keys.anthropic_api_key.strip(),
                tavily_api_key=keys.tavily_api_key.strip(),
                voice=keys.voice.strip(),
            )
            # Rebuild against the new credentials on the next question.
            with self._pipeline_lock:
                self._pipeline = None
            threading.Thread(target=self._warm_up, daemon=True).start()
            return {"ready": self._settings.ready,
                    "missing": self._settings.missing()}

        static_dir = Path(__file__).parent / "static"
        if static_dir.is_dir():
            self.settings_app.mount(
                "/", StaticFiles(directory=str(static_dir), html=True), name="static"
            )


if __name__ == "__main__":
    app = ReachySearch()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
