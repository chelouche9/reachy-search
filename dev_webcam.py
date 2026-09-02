#!/usr/bin/env python3
"""Run the whole pipeline with no robot and no daemon.

Uses the laptop webcam and speakers instead of Reachy's. Same `pipeline.py` the
app uses, so anything tuned here — prompts, queries, voice, timing — carries
straight over to the robot.

    uv pip install -e '.[dev]'
    ANTHROPIC_API_KEY=... TAVILY_API_KEY=... python dev_webcam.py

Press Enter to ask a question (this stands in for booping an antenna).
"""

import argparse
import logging
import queue
import sys
import threading
import time

import numpy as np

from reachy_search import audio_utils, config, moves, pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-22s %(message)s")
logger = logging.getLogger("dev")

MIC_BLOCK = 1024


def grab_jpeg(camera) -> bytes | None:
    import cv2

    # Throw away a few frames — most webcams need a moment to expose properly,
    # and a dark first frame makes the vision call look worse than it is.
    for _ in range(5):
        ok, frame = camera.read()
    if not ok:
        return None
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else None


def record_question(samplerate: int) -> np.ndarray:
    import sounddevice as sd

    chunks: queue.Queue = queue.Queue()

    def on_audio(indata, _frames, _time, status):
        if status:
            logger.debug("mic: %s", status)
        chunks.put(indata.copy().reshape(-1).astype(np.float32))

    def pull():
        try:
            return chunks.get(timeout=0.1)
        except queue.Empty:
            return None

    with sd.InputStream(samplerate=samplerate, channels=1,
                        blocksize=MIC_BLOCK, dtype="float32", callback=on_audio):
        return audio_utils.record_until_silence(
            pull=pull,
            samplerate=samplerate,
            stop_event=threading.Event(),
            max_seconds=config.MAX_QUESTION_S,
            hang_seconds=config.SILENCE_HANG_S,
            silence_rms=config.SILENCE_RMS,
            min_speech_seconds=config.MIN_SPEECH_S,
        )


def preview_moves() -> None:
    """Print what the loader moves do, without a robot to do them on.

    Not a substitute for watching them run, but it catches a move that has
    drifted outside the safe envelope or forgotten to move the antennas at all.
    """
    for name, move in moves.THINKING_MOVES.items():
        frames = [move(t / 50.0).clamped() for t in range(int(50 * 4))]
        span = lambda field: (
            min(getattr(f, field) for f in frames),
            max(getattr(f, field) for f in frames),
        )
        print(f"\n{name}  (4s @ 50Hz)")
        for field in ("pitch", "roll", "yaw", "right_antenna", "left_antenna"):
            low, high = span(field)
            print(f"    {field:<16} {low:7.1f} .. {high:6.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--moves", action="store_true",
                        help="print the loader move envelopes and exit")
    args = parser.parse_args()

    if args.moves:
        preview_moves()
        return 0

    import cv2
    import sounddevice as sd

    settings = config.load()
    if not settings.ready:
        print(f"Missing API key(s): {', '.join(settings.missing())}", file=sys.stderr)
        return 1

    engine = pipeline.Pipeline(settings)
    print("Loading models...")
    engine.warm_up()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    mic_rate = 16_000
    try:
        while True:
            input("\nHold something up, then press Enter and ask (Ctrl-C to quit)... ")

            jpeg = grab_jpeg(camera)
            print("Listening...")
            audio = record_question(mic_rate)

            started = time.monotonic()
            print("Thinking...")
            outcome = engine.answer(jpeg, audio, mic_rate)
            print(f"({time.monotonic() - started:.1f}s)  {outcome.spoken}")

            speech = engine.speaker.synthesize_to_array(outcome.spoken)
            if speech.size:
                sd.play(speech, engine.speaker.sample_rate)
                sd.wait()
    except KeyboardInterrupt:
        print()
    finally:
        camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
