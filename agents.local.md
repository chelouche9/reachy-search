# Session notes

## Setup
- Robot: **not yet owned**. Ordering as of Sep 2026 — Pollen store quotes up to
  90 days lead; Seeed lists the Wireless kit ($499) in stock, likely the fast
  path. Lite (~$299) would be technically sufficient since the heavy lifting is
  off-robot, but Wireless films better (no cable).
- Everything so far was built and verified **robot-free** on macOS. `dev_webcam.py`
  is the harness; `dev_webcam.py --moves` prints motion envelopes.
- `reachy-mini` 1.10.0 installs cleanly on macOS via uv. `pyenv` python 3.11.9.

## Decisions already made — don't reopen
- **Python app, not JS.** `AGENTS.md` says default to JS, but JS apps are not yet
  discoverable in the app store (`skills/create-app.md`), and store presence is
  the distribution plan.
- **default template, not `conversation`.** The conversation template forks an
  OpenAI-Realtime speech-to-speech pipeline we don't use.
- Claude for vision + answer, Tavily for search, **local** faster-whisper and
  Piper for voice. Two API keys total, both enterable from the settings page.
- Triggers (revised 2026-09-01, Yonatan's call — demo should be "live"):
  wake word "hey Reachy" (energy gate + local Whisper + fuzzy name match in
  `main.match_wake`; official conversation app has no wake code to copy),
  plus antenna boop and the settings-page Ask button. Wake path speaks
  "What can I do for you?" (pre-rendered at warm-up) before listening;
  a one-shot "hey Reachy, search for X" skips the prompt.

## Gotchas found the hard way
- `set_target` takes antennas as `[right, left]` in **radians**. The SDK source
  says this; `skills/interaction-patterns.md` says `left, right` — trust the source.
- Antenna press must be detected as *deviation from the last commanded angle*,
  not raw angle, because the app animates the antennas continuously. Raw
  thresholding self-triggers.
- Antenna slew rates above ~380°/s stop being tracked by the servos, so a "fast"
  spin reads as a mushy wobble. Motion is tuned under that ceiling. **Re-check
  on real hardware** — the ceiling is an estimate, not a measured spec.
- Piper renders faster than real time, so the speaking animation has to be held
  for the audio's computed duration, not until synthesis returns.
- `reachy_mini.enable_wobbling()` takes no arguments (the `MediaManager` method
  of the same name takes a callback — different thing).

## Verified end-to-end (2026-09-01, real API calls)
- Full pipeline on a Bialetti photo + say-synthesized "What is this? Find me a
  cheaper one.": Whisper heard it exactly, Claude identified the 3-cup Moka
  Express, Tavily found prices, answer spoke them naturally. **11.7s total**
  (incl. Whisper load; the thinking move covers the API portion, ~8s).
- Noise-frame test: recognized=false + a usable spoken fallback line.
- TTS→STT round trip: 89% word overlap ("moka"→"mocha" is the only drift).
- API keys live in `.env` (mode 600, gitignored), copied from
  `~/projects/chelouche/cloud-agents/.env`. `config.load()` reads `.env` itself.

## macOS-only breakage (does not affect the robot)
- **piper-tts macOS arm64 wheels (1.6.0 AND 1.7.0) are broken**: the compiled
  espeak bridge ignores its data-dir argument and hits a baked-in CI path
  (`/Users/runner/...`), and the C code calls exit(1) — not catchable
  in-process. `tts.py` probes Piper in a subprocess and falls back to macOS
  `say` for dev. Linux (the robot) is unaffected. Recheck newer piper releases.
- Terminal has no camera permission yet (macOS TCC); grant it to test with the
  real webcam via `dev_webcam.py`.

## Simulation on this Mac (verified 2026-09-01)
- Launch (viewer): `.venv/bin/mjpython -c "from reachy_mini.daemon.app.main import main; main()" --sim`
  — plain `reachy-mini-daemon --sim` FAILS on macOS ("launch_passive requires
  mjpython"); headless (`--headless`) works under regular python.
- Media needed two fixes: `reachy_mini[mujoco]` extra, and a symlink of pyenv's
  `libpython3.11.dylib` into `.venv/.../gstreamer_libs/lib/` (uv venvs don't
  expose it; without it the daemon hangs silently before the API comes up).
- With media up: **mic and speaker are the Mac's default devices** — voice flows
  through the sim. **Camera reports `mujoco` but `get_frame_jpeg()` returns
  None** in daemon 1.10.0, viewer or not, so the vision leg takes the graceful
  fallback in sim. Yonatan recalls a sim session where it "watched" them —
  differently wired setup (desktop app / dashboard head-tracking?); unresolved.
- `dev_sim_moves.py` cycles all moves against the daemon (verified: sim tracks them).

- Camera in sim renders an empty checkerboard world (verified by looking at a
  frame). For live desk demos the app takes frames from the Mac webcam:
  `REACHY_SEARCH_CAMERA=webcam` (`camera.py`; default `auto` = robot first,
  webcam fallback — robot stays correct on hardware). Webcam needs macOS camera
  permission for the user's terminal; Claude's own process is TCC-blocked.

## Still to do
- Watch the three thinking moves on hardware and retune amplitudes.
- Test `dev_webcam.py` interactively (mic + webcam) once camera permission is granted.
- Record `reachy_search/assets/demo.mp4` for the landing page.
- `hf auth login`, then publish (see the handover doc's distribution playbook).
