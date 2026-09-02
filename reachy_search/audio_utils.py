"""Shared audio plumbing.

The robot's mic and speaker run at whatever rate the device reports; Whisper
wants 16 kHz mono and Piper emits at its voice's own rate. Everything that
converts between those lives here so the STT and TTS modules stay readable.
"""

import numpy as np
from scipy.signal import resample_poly

WHISPER_RATE = 16_000


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse (n, channels) to (n,). The mic array is multi-channel."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.reshape(-1)


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32)
    # Rational resampling on the reduced ratio — much cheaper than a naive
    # resample() and it does not smear the signal the way interpolation does.
    from math import gcd

    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def peak_rms(audio: np.ndarray, samplerate: int, block_s: float = 0.25) -> float:
    """Loudest short block in the clip. A word at the end of a long quiet
    capture vanishes in a whole-clip average but not in this."""
    block = max(1, int(samplerate * block_s))
    if audio.size < block:
        return rms(audio)
    trimmed = audio[: (audio.size // block) * block].reshape(-1, block)
    return float(np.sqrt(np.mean(np.square(trimmed, dtype=np.float64), axis=1)).max())


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    return (np.asarray(audio, dtype=np.float32) / 32768.0).astype(np.float32)


def record_until_silence(
    pull,
    samplerate: int,
    stop_event,
    max_seconds: float,
    hang_seconds: float,
    silence_rms: float,
    min_speech_seconds: float,
    should_abort=None,
) -> np.ndarray:
    """Collect mic audio until the speaker stops, or `max_seconds` elapse.

    `pull` returns the next chunk of float32 audio, or None when nothing is
    buffered yet. Deliberately a plain energy gate rather than a neural VAD:
    the question is one short sentence started by a deliberate button press, so
    there is no wake-word ambiguity to resolve and no reason to pay for a model.
    """
    import time

    collected: list[np.ndarray] = []
    total_samples = 0
    silent_samples = 0
    speech_samples = 0
    heard_speech = False

    max_samples = int(max_seconds * samplerate)
    hang_samples = int(hang_seconds * samplerate)
    min_speech_samples = int(min_speech_seconds * samplerate)
    # Wall-clock bound as well as a sample bound: with no working audio device
    # `pull` returns None forever and samples never accumulate.
    deadline = time.monotonic() + max_seconds * 2.0

    while total_samples < max_samples and not stop_event.is_set():
        if time.monotonic() > deadline:
            break
        if should_abort is not None and should_abort():
            break
        chunk = pull()
        if chunk is None or len(chunk) == 0:
            time.sleep(0.01)
            continue

        mono = to_mono(chunk)
        collected.append(mono)
        total_samples += mono.size

        if rms(mono) > silence_rms:
            heard_speech = True
            speech_samples += mono.size
            silent_samples = 0
        elif heard_speech:
            silent_samples += mono.size
            # Stop on a real pause, but only once we have enough speech that
            # the pause means "done" rather than "drew breath".
            if silent_samples > hang_samples and speech_samples > min_speech_samples:
                break

    if not collected:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(collected)
