"""Synthesized ambient sounds. No asset files — everything is a few sines.

The searching loop is tempo-matched to the drumroll move (2.2 taps/sec), so
the beeps and the antenna taps read as one performance even though they are
not hard-synced.
"""

import numpy as np

TAP_HZ = 2.2  # keep in step with moves.drumroll


def _blip(samplerate: int, freq: float, dur: float, amp: float) -> np.ndarray:
    """One soft rounded beep — sine with a smooth rise/fall envelope."""
    n = int(samplerate * dur)
    t = np.arange(n) / samplerate
    envelope = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return (amp * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def searching_loop(samplerate: int, volume: float = 0.10) -> np.ndarray:
    """~1.8s of gentle computer-processing chirps, loopable seamlessly.

    Two alternating blips per tap period (a low tick and a higher answer),
    a faint hum underneath, and a little rising three-note figure once per
    loop so it doesn't feel like a metronome. Quiet by design: it sits under
    the voice, never competes with it.
    """
    period = 1.0 / TAP_HZ
    loop_s = 4 * period  # 4 taps per loop
    n = int(samplerate * loop_s)
    out = np.zeros(n, dtype=np.float32)

    # Faint hum bed with a slow swell.
    t = np.arange(n) / samplerate
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * t / loop_s)
    out += (0.18 * volume * swell * np.sin(2 * np.pi * 96.0 * t)).astype(np.float32)

    # Tick... tock... alternating pitches on the tap grid.
    for i in range(4):
        start = int(i * period * samplerate)
        freq = 740.0 if i % 2 == 0 else 988.0
        blip = _blip(samplerate, freq, 0.07, volume)
        out[start:start + blip.size] += blip[: max(0, n - start)]

    # A rising three-note "working on it" figure at the loop's midpoint.
    for j, freq in enumerate((1175.0, 1480.0, 1760.0)):
        start = int((2 * period + j * 0.09) * samplerate)
        blip = _blip(samplerate, freq, 0.06, volume * 0.7)
        if start + blip.size < n:
            out[start:start + blip.size] += blip

    return np.clip(out, -1.0, 1.0)
