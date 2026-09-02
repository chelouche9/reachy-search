# Reachy Search — implementation plan

Show Reachy Mini an object, boop an antenna, ask a question out loud. It plays a
"thinking" move while it searches the web, then speaks a short answer.

Standalone open-source toy. Not a product. One object, one search, one animation,
one spoken answer. No accounts, no settings beyond API keys, no roadmap.

## Decisions (settled)

| Decision | Choice | Why |
|---|---|---|
| App flavour | **Python** app, not JS | JS apps are not yet discoverable in the Reachy app store (`skills/create-app.md`). Store presence is the distribution plan. |
| Template | **default**, not `conversation` | The conversation template forks an OpenAI-Realtime speech-to-speech pipeline. We use Claude + local STT/TTS, so that fork is baggage, not scaffolding. |
| Brain | Claude (`claude-opus-5`) | Hosted, one multimodal call does object ID + query authoring. |
| Search | Tavily | Text-in only — which is exactly why the VLM is the bridge from the physical world to the query. |
| STT | faster-whisper (local) | No extra API key. `base` model is enough for short questions. |
| TTS | Piper (local) | Built for Pi-class hardware, so it works on Wireless. Keeps the install to two API keys. |
| Trigger | **Antenna touch** | Antennas are semi-passive (low P gain) so they double as physical buttons. Zero false triggers, no always-on mic, and it films well. |

Two API keys total: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`. Both enterable from the
app's own settings page — store users never touch a terminal.

## Architecture

```
antenna touch
   └─> grab JPEG frame (before the head moves)
   └─> record until silence  ──> faster-whisper ──> transcript
              │
              ├─> Claude vision: frame + transcript
              │      -> { recognized, object, search_query, fallback_line }
              │
              ├─> Tavily search (search_query)
              │
              └─> Claude compose: results -> 2-3 spoken sentences
                     -> Piper -> push_audio_sample -> speaker
```

Latency is not hidden — the thinking move *is* the UX, and it is the shareable
part of the demo. Budget real time on the choreography.

## Threading model

The SDK's rule is one control loop, one place calling `set_target`. So:

- **Control loop** (`main.py`, in `run()`, 50 Hz): owns all motion. Reads the app
  state, computes a pose from `moves.py`, calls `set_target`. Never blocks.
- **Worker thread** (`pipeline.py`): all blocking work — audio capture, Whisper,
  two Claude calls, Tavily, Piper. Never touches `set_target`.
- They communicate through an enum `state` + a `queue.Queue` for results.

Antenna press = deviation between *present* antenna angle and the angle we last
*commanded*, not raw angle — because we animate the antennas continuously, so a
raw threshold would self-trigger. Polled every 4th tick with a cooldown.

Speech-synced head motion comes free from `reachy_mini.enable_wobbling()`, which
composes offsets from played audio onto our target pose daemon-side.

## States

`IDLE` -> (antenna) -> `LISTENING` -> `THINKING` -> `SPEAKING` -> `IDLE`,
with `ERROR` as a short detour that always returns to `IDLE`.

| State | Motion |
|---|---|
| IDLE | Slow breathing: gentle z-bob + micro pitch, antennas at rest with a rare twitch |
| LISTENING | Head tilts up toward the object, antennas perk forward and hold with a fine tremor |
| THINKING | One of three loader moves, picked at random per search (see below) |
| SPEAKING | Head returns to face the user; wobbling drives the rest |
| ERROR | Quick antenna shake, then back to idle |

### The three loader moves

Defined symbolically (functions of time), per `skills/symbolic-motion.md` — cheap
to tune, no recorded data.

1. **ponder** — head cocks to one side and holds, slow yaw sweep like reading a
   shelf; antennas counter-rotate slowly, a dial being turned.
2. **scan** — head traces a small figure-eight; antennas flick alternately.
3. **spin_up** — the literal loading spinner: antennas counter-rotate fast and
   continuously while the head bobs, accelerating slightly over the first second.

## Failure modes (each gets a spoken line, never silence)

| Failure | Behaviour |
|---|---|
| Nothing recognisable in frame | "Hold it a bit closer?" + error shake |
| Empty/failed transcript | "I didn't catch that." |
| Search returns nothing | Answer from the vision pass alone, flagged as not-searched |
| API key missing | Spoken once at startup, and shown on the settings page |
| Any exception in the worker | Logged, generic spoken apology, state returns to IDLE |

## Robot-free development

`dev_webcam.py` runs the whole pipeline against a laptop webcam and speakers with
no robot and no daemon — same `pipeline.py`, stubbed motion. This is what gets
built and tuned until hardware arrives.

## Open items (do not block the build)

- **Store name.** Package is `reachy_search`. The display name in `README.md`
  frontmatter and `index.html` is cosmetic and free to change once the demo video
  exists — "Reachy Lookup" / "ShowMe" / "Curious Reachy" were the candidates.
- **Model latency.** Defaulting to `claude-opus-5` at `effort: "low"`, which is
  the sanctioned way to trade depth for speed inside one model. If the thinking
  move still outlasts its welcome, the next lever is the user's call.
- **Demo video** must land in `reachy_search/assets/demo.mp4` before publishing —
  the landing page and the README both reference it.
