"""Choreography.

Every move is a pure function of elapsed time returning a `Frame`. Nothing here
touches the robot — `main.py` owns the single control loop that does. Motion is
defined symbolically rather than recorded, so it is cheap to tune and scales to
any duration (see the SDK's `skills/symbolic-motion.md`).

The thinking moves are the point of this app. They are what people will film.
"""

import math
import random
from dataclasses import dataclass, replace

import numpy as np
from reachy_mini.utils import create_head_pose

# The SDK clamps to +/-40 deg on pitch and roll. Stay well inside — motion that
# rides the limit looks strained rather than expressive.
MAX_TILT_DEG = 28.0
MAX_YAW_DEG = 55.0
MAX_ANTENNA_DEG = 65.0
# Body can physically do +/-160, but the head-minus-body delta is capped at 65
# by the SDK, so loaders keep the body modest and let the head lead.
MAX_BODY_DEG = 45.0


@dataclass(frozen=True)
class Frame:
    """One instant of pose. Translations in mm, rotations in degrees."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    # set_target takes antennas as [right, left], in radians.
    right_antenna: float = 0.0
    left_antenna: float = 0.0
    # Whole-torso rotation, degrees. Head yaw is world-frame, so a move that
    # turns the body must carry the head with it (or deliberately not).
    body_yaw: float = 0.0

    def __add__(self, other: "Frame") -> "Frame":
        return Frame(*(getattr(self, f) + getattr(other, f) for f in _FIELDS))

    def scaled(self, k: float) -> "Frame":
        return Frame(*(getattr(self, f) * k for f in _FIELDS))

    def clamped(self) -> "Frame":
        return replace(
            self,
            roll=_clamp(self.roll, MAX_TILT_DEG),
            pitch=_clamp(self.pitch, MAX_TILT_DEG),
            yaw=_clamp(self.yaw, MAX_YAW_DEG),
            right_antenna=_clamp(self.right_antenna, MAX_ANTENNA_DEG),
            left_antenna=_clamp(self.left_antenna, MAX_ANTENNA_DEG),
            body_yaw=_clamp(self.body_yaw, MAX_BODY_DEG),
        )

    def head_pose(self) -> np.ndarray:
        f = self.clamped()
        return create_head_pose(
            x=f.x, y=f.y, z=f.z,
            roll=f.roll, pitch=f.pitch, yaw=f.yaw,
            mm=True, degrees=True,
        )

    def antennas_rad(self) -> np.ndarray:
        f = self.clamped()
        return np.deg2rad([f.right_antenna, f.left_antenna])

    def body_yaw_rad(self) -> float:
        return math.radians(self.clamped().body_yaw)


_FIELDS = (
    "x", "y", "z", "roll", "pitch", "yaw",
    "right_antenna", "left_antenna", "body_yaw",
)

REST = Frame()


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def blend(a: Frame, b: Frame, k: float) -> Frame:
    """Linear blend, k=0 -> a, k=1 -> b. Used to cross-fade between states."""
    k = max(0.0, min(1.0, k))
    return Frame(*(
        getattr(a, f) * (1.0 - k) + getattr(b, f) * k for f in _FIELDS
    ))


def _ease_in(t: float, duration: float) -> float:
    """0 -> 1 over `duration`, smoothly. Keeps moves from starting with a jerk."""
    if duration <= 0:
        return 1.0
    k = max(0.0, min(1.0, t / duration))
    return k * k * (3.0 - 2.0 * k)


def _flick(t: float, period: float, phase: float = 0.0, rise: float = 0.16) -> float:
    """A quick snap out and a slower settle back. Antennas, not sine waves."""
    u = ((t - phase) % period) / period
    edge = rise / period
    if u < edge:
        return math.sin(math.pi * 0.5 * (u / edge))
    decay = (u - edge) / (1.0 - edge)
    return math.cos(math.pi * 0.5 * decay) ** 2 * math.exp(-3.0 * decay)


# --------------------------------------------------------------------------
# Ambient states
# --------------------------------------------------------------------------

def idle(t: float) -> Frame:
    """Breathing. The illusion of life costs almost nothing, so never stop."""
    breath = 2.4 * math.sin(2 * math.pi * 0.16 * t)
    # A slow, irrational-ratio drift so the loop never visibly repeats.
    drift = 3.0 * math.sin(2 * math.pi * 0.037 * t)
    # An occasional ear twitch, roughly every 9 seconds.
    twitch = 7.0 * _flick(t, 9.0, phase=2.0, rise=0.08)
    return Frame(
        z=breath,
        pitch=0.7 * breath,
        yaw=drift,
        right_antenna=twitch,
        left_antenna=-0.6 * twitch,
    )


def attentive(t: float) -> Frame:
    """Awake and available: antennas half-perked, chin slightly up, glancing
    around a little faster than idle. Reads as "still with you" from across
    the room — the point is that you can see it hasn't gone back to sleep."""
    breath = 2.0 * math.sin(2 * math.pi * 0.2 * t)
    glance = 7.0 * math.sin(2 * math.pi * 0.055 * t + 1.0)
    perk = 14.0 + 2.5 * math.sin(2 * math.pi * 0.35 * t)
    return Frame(
        z=breath,
        pitch=-3.0 + 0.6 * breath,
        yaw=glance,
        right_antenna=perk,
        left_antenna=perk - 1.5 * math.sin(2 * math.pi * 0.35 * t),
    )


def listening(t: float) -> Frame:
    """Attention: chin up toward the held object, antennas perked and alert."""
    settle = _ease_in(t, 0.45)
    tremor = 1.6 * math.sin(2 * math.pi * 5.5 * t)
    breath = 1.2 * math.sin(2 * math.pi * 0.5 * t)
    return Frame(
        z=3.0 * settle + breath,
        pitch=-9.0 * settle,
        right_antenna=(34.0 + tremor) * settle,
        left_antenna=(34.0 - tremor) * settle,
    )


def speaking(t: float) -> Frame:
    """Open with the payoff beat — a quick "aha!" pop up to face the listener —
    then settle and let `enable_wobbling()` carry the voice. Anything more than
    a gentle sway after the pop fights the speech-driven motion."""
    aha = 0.55
    if t < aha:
        k = t / aha
        pop = math.sin(math.pi * min(1.0, k * 1.25)) * (1.0 - 0.3 * k)
        return Frame(
            z=7.0 * pop,
            pitch=-6.0 * pop,
            right_antenna=30.0 * pop + 10.0 * k,
            left_antenna=24.0 * pop + 10.0 * k,
        )
    ts = t - aha
    sway = 3.4 * math.sin(2 * math.pi * 0.28 * ts)
    return Frame(
        yaw=sway,
        roll=0.3 * sway,
        z=1.5 * math.sin(2 * math.pi * 0.22 * ts),
        right_antenna=10.0,
        left_antenna=10.0,
    )


def error_shake(t: float) -> Frame:
    """A quick 'nope' — fast antenna shake with a small head shake under it."""
    damp = math.exp(-1.6 * t)
    shake = math.sin(2 * math.pi * 3.2 * t) * damp
    return Frame(
        yaw=7.0 * shake,
        pitch=3.0 * damp,
        right_antenna=20.0 * shake,
        left_antenna=-20.0 * shake,
    )


def greet(t: float) -> Frame:
    """Played once at startup so you can see the app is alive and listening."""
    damp = math.exp(-1.1 * t)
    return Frame(
        pitch=-7.0 * math.sin(2 * math.pi * 0.9 * t) * damp,
        right_antenna=30.0 * math.sin(2 * math.pi * 1.3 * t) * damp,
        left_antenna=30.0 * math.sin(2 * math.pi * 1.3 * t + 0.6) * damp,
    )


# --------------------------------------------------------------------------
# Beat machinery
#
# The craft (measured from Pollen's animator-recorded emotion moves, and per
# Anki's Vector design guide): thinking is snap-to-a-pose, HOLD, snap again —
# not continuous drift. 16-31% of the recorded frames are near-still, antennas
# are asymmetric (L/R correlation +0.24 in curious1), and head yaw swings +/-45
# at up to ~140 deg/s. Continuous sinusoids read as a screensaver.
# --------------------------------------------------------------------------

def _beat(t: float, pattern: tuple[float, ...]) -> tuple[int, float, float]:
    """Which beat are we in, how far into it, how long is it."""
    total = sum(pattern)
    cycle, u = int(t // total), t % total
    acc = 0.0
    for i, duration in enumerate(pattern):
        if u < acc + duration:
            return cycle * len(pattern) + i, u - acc, duration
        acc += duration
    return cycle * len(pattern), 0.0, pattern[0]


def _wobble_free(k: int, seed: float, lo: float, hi: float) -> float:
    """Deterministic pseudo-random value per beat — variety without state."""
    x = math.sin(k * 127.1 + seed * 311.7) * 43758.5453
    return lo + (x - math.floor(x)) * (hi - lo)


def _snap(u: float, snap_s: float) -> float:
    """0 -> 1 smoothstep over the first `snap_s` of a beat, then hold at 1."""
    if u >= snap_s:
        return 1.0
    k = u / snap_s
    return k * k * (3.0 - 2.0 * k)


def _held(a: float, b: float, u: float, snap_s: float) -> float:
    """Ease from pose value a to b at the start of a beat, then hold."""
    return a + (b - a) * _snap(u, snap_s)


# --------------------------------------------------------------------------
# The loader moves — one is picked at random per search
# --------------------------------------------------------------------------

def radar(t: float) -> Frame:
    """The searchlight: the whole torso sweeps the room in held steps, head
    leading into each turn, chin up, antennas swept back in concentration.
    The body is the robot's biggest gesture — nothing about this is subtle."""
    pattern = (1.5, 1.1, 1.7, 1.2)
    k, u, _ = _beat(t, pattern)
    prev = k - 1

    # Alternate sweep direction with varied reach; hold at each bearing.
    def bearing(i: int) -> float:
        side = 1.0 if i % 2 == 0 else -1.0
        return side * _wobble_free(i, 11.0, 20, 36)

    body = _held(bearing(prev), bearing(k), u, 0.95)
    settle = _ease_in(t, 0.5)
    # Head leads the body into the turn, then relaxes back onto it.
    lead = 10.0 * _snap(u, 0.35) * math.exp(-2.2 * u)
    direction = 1.0 if bearing(k) > bearing(prev) else -1.0
    return Frame(
        z=3.0 * math.sin(2 * math.pi * 0.24 * t),
        pitch=(-8.0 - 2.0 * math.sin(2 * math.pi * 0.24 * t)) * settle,
        yaw=(body + direction * lead) * settle,
        body_yaw=body * settle,
        right_antenna=-24.0 * settle,
        left_antenna=-24.0 * settle + 3.0 * math.sin(2 * math.pi * 1.1 * t),
    )


def drumroll(t: float) -> Frame:
    """Impatient fingers on a desk: head tips down-forward, antennas drum an
    alternating rhythm, head nodding faintly on the bar. Every few bars the
    head cocks to a new angle, like shifting weight while you wait."""
    bar = 1.6
    k, u, _ = _beat(t, (bar,))
    prev = k - 1

    # Alternating taps — sin^2 keeps them distinct, and the rate is what the
    # antenna servos can actually articulate rather than a blur.
    tap_hz = 2.2
    phase = 2 * math.pi * tap_hz * t
    tap_r = math.sin(phase) ** 2 * (1.0 if math.sin(phase) > 0 else 0.0)
    tap_l = math.sin(phase) ** 2 * (1.0 if math.sin(phase) < 0 else 0.0)

    cock = _held(_wobble_free(prev, 13.0, -14, 14), _wobble_free(k, 13.0, -14, 14), u, 0.4)
    settle = _ease_in(t, 0.4)
    nod = 2.0 * math.sin(2 * math.pi * t / bar)
    return Frame(
        z=-2.0 * settle,
        pitch=(9.0 + nod) * settle,
        roll=cock * settle,
        yaw=0.3 * cock * settle,
        right_antenna=(8.0 + 24.0 * tap_r) * settle,
        left_antenna=(8.0 + 24.0 * tap_l) * settle,
    )


def scan(t: float) -> Frame:
    """Active searching: quick darts to bold poses with short holds — the
    "flurry of thought" register. Antennas flick one at a time, off-beat."""
    pattern = (0.55, 0.45, 0.75, 0.5, 0.65)
    k, u, _ = _beat(t, pattern)
    prev = k - 1

    yaw = _held(_wobble_free(prev, 4.0, -34, 34), _wobble_free(k, 4.0, -34, 34), u, 0.42)
    pitch = _held(_wobble_free(prev, 5.0, -12, 8), _wobble_free(k, 5.0, -12, 8), u, 0.42)
    roll = _held(_wobble_free(prev, 6.0, -7, 7), _wobble_free(k, 6.0, -7, 7), u, 0.45)

    # One antenna flicks per beat — whichever, decided by the beat hash.
    # Fast decay so the flick has died before the beat boundary hands the
    # flick to the other antenna (a leftover angle would teleport to zero).
    flick = 52.0 * _snap(u, 0.2) * math.exp(-6.0 * u)
    right_flicks = _wobble_free(k, 7.0, 0, 1) > 0.5
    settle = _ease_in(t, 0.4)
    return Frame(
        z=2.5 * math.sin(2 * math.pi * 0.5 * t),
        roll=roll * settle,
        pitch=pitch * settle,
        yaw=yaw * settle,
        right_antenna=(10 + (flick if right_flicks else 0.0)) * settle,
        left_antenna=(10 + (0.0 if right_flicks else flick)) * settle,
    )


def spin_up(t: float) -> Frame:
    """The literal loading spinner, now with anticipation: antennas pull back
    against the spin for a third of a second, then release into it. Frequency
    is integrated into a phase so the ramp doesn't stutter."""
    windup = 0.35
    pull_frame = Frame(z=-2.0, pitch=3.0, right_antenna=-18.0, left_antenna=18.0)
    if t < windup:
        return pull_frame.scaled(_snap(t, windup))

    ts = t - windup
    f0, f1, ramp = 0.55, 1.45, 1.3
    if ts < ramp:
        turns = f0 * ts + (f1 - f0) * ts * ts / (2.0 * ramp)
    else:
        turns = f0 * ramp + (f1 - f0) * ramp / 2.0 + f1 * (ts - ramp)
    phase = 2 * math.pi * turns
    speed = _ease_in(ts, ramp)

    # Head lifts as it "loads", with a little pop on each full turn.
    spin = Frame(
        z=5.0 * math.sin(phase * 0.5),
        pitch=-2.0 - 5.0 * speed - 3.0 * math.sin(phase * 0.5) * speed,
        yaw=9.0 * math.sin(2 * math.pi * 0.21 * ts),
        right_antenna=42.0 * math.sin(phase) * speed,
        left_antenna=42.0 * math.cos(phase) * speed,
    )
    # Release the wound-up pose into the spin instead of cutting from it.
    return blend(pull_frame, spin, _snap(ts, 0.3))


# The loader. Drumroll won the audition (Yonatan, 2026-09-01); the others stay
# on the bench for the sim player and for changes of heart.
THINKING_MOVES = {
    "drumroll": drumroll,
}
BENCH_MOVES = {
    "radar": radar,
    "scan": scan,
    "spin_up": spin_up,
}


def pick_thinking_move(rng: random.Random | None = None) -> tuple[str, callable]:
    """Pick a loader at random so repeated searches don't look canned."""
    rng = rng or random
    name = rng.choice(list(THINKING_MOVES))
    return name, THINKING_MOVES[name]
