---
title: Reachy Search
emoji: 🔎
colorFrom: yellow
colorTo: pink
sdk: static
pinned: false
short_description: Hold something up, ask about it, and Reachy searches the web and answers out loud.
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Search

Say **"hey Reachy"**. It perks up and asks what it can do for you. Ask —
about something you're holding up, or anything at all.

It looks at the object, spins its antennas while it searches the web, and tells
you the answer out loud.

> "What is this?" — *"That's a stovetop moka pot. Water boils in the bottom
> chamber and pressure pushes it up through the coffee. Yours looks like a
> six-cup aluminium one."*

Ask it anything you'd ask while holding the thing: *what is this, is this
dishwasher safe, find me a cheaper one, what do I do with this.*

## How it works

1. **Say "hey Reachy".** A local Whisper model listens for its name — audio
   never leaves the machine. It answers "What can I do for you?" and listens.
   Say it all in one breath ("hey Reachy, search for…") and it skips the
   pleasantries. Booping an antenna works too — the antennas are semi-passive,
   so they double as physical buttons.
2. **Ask out loud.** It grabs a camera frame as you finish talking and records
   until you stop.
3. **It thinks — visibly.** Claude identifies the object and writes a search
   query, Tavily searches, Claude writes a spoken answer. Meanwhile Reachy plays
   one of three thinking animations, picked at random.
4. **It answers.** Piper speaks it, and the head moves with the voice.

## Setup

Two API keys, both with usable free tiers:

- [Anthropic](https://console.anthropic.com/settings/keys) — identifies the
  object and writes the answer
- [Tavily](https://app.tavily.com/home) — searches the web

Install the app from the Reachy Mini dashboard, then open its settings page (the
gear icon) and paste the keys in. Nothing else to configure.

Developers can skip the settings page by exporting `ANTHROPIC_API_KEY` and
`TAVILY_API_KEY` before starting the daemon — the app inherits its environment,
and the environment wins over saved settings.

Speech recognition and speech synthesis are local. Both models download
themselves the first time you ask a question, so give the first one a moment.

## Running it without a robot

The whole pipeline works on a laptop webcam, no robot and no daemon needed:

```bash
uv pip install -e '.[dev]'
ANTHROPIC_API_KEY=... TAVILY_API_KEY=... python dev_webcam.py
```

Press Enter instead of booping an antenna. Prompts, queries and voice tuned here
carry straight over to the robot.

To inspect the thinking animations without hardware:

```bash
python dev_webcam.py --moves
```

## The thinking moves

Latency isn't hidden here — the animation *is* the interface. Three of them,
chosen at random per search so repeat questions don't look canned:

| Move | What it does |
|---|---|
| **ponder** | Head cocks to one side and holds, sweeping slowly like it's reading a shelf. Antennas counter-rotate, a dial being turned. |
| **scan** | A small figure-eight, antennas flicking alternately, off the head's rhythm. |
| **spin_up** | The literal loading spinner: antennas counter-rotate, accelerating from a lazy turn to a blur. |

They're defined as functions of time in `reachy_search/moves.py` — no recorded
data, so amplitudes and frequencies are a one-line change. Antenna slew rates
are deliberately held under ~380°/s; past that the servos stop tracking and a
fast spin reads as a mushy wobble.

## Layout

| File | What's in it |
|---|---|
| `main.py` | The app: control loop, state machine, worker thread, settings API |
| `moves.py` | All choreography |
| `pipeline.py` | Frame + question in, a sentence to say out loud out |
| `brain.py` | The two Claude calls — identify, then compose |
| `search.py` | Tavily |
| `stt.py` / `tts.py` | Whisper and Piper |
| `dev_webcam.py` | Robot-free harness |

One control loop owns all motion at 50 Hz and never blocks. Everything slow runs
on a worker thread that talks to it only by setting a state enum. That's why the
animation stays smooth while the network doesn't.

## Use it in your own app

This is a stack of independent layers, not one flow. Take the top one for the
whole act, or reach in at any depth and ignore the rest — nothing forces our
interaction model on your app:

| Layer | Import | What you get, standalone |
|---|---|---|
| The whole act | `EmbodiedSearch(mini, ...).ask(q)` | drumroll + chirps + "Searching..." + spoken answer |
| The agent | `brain.Brain(...).respond(jpeg, q)` | frame + question → answer text; decides when to search; `searcher=None` makes it pure look-and-answer; `reset()` clears memory |
| Search only | `search.Searcher(key).search(q)` | Tavily results shaped for spoken answers |
| The performance kit | `moves.*`, `sounds.*`, `tts.Speaker` | beat-based choreography, processing chirps, local voice — usable with your own logic entirely |

`ask()` runs the whole act inside your app: drumroll, processing chirps, a
spoken "Searching...", the aha pop, and the answer out loud.

```python
from reachy_search import EmbodiedSearch

skill = EmbodiedSearch(mini, anthropic_api_key=..., tavily_api_key=...)
skill.warm_up()                     # optional, once
answer = skill.ask("find me a cheaper one")   # frame comes from the camera
```

`ask()` owns `set_target` while it runs — pause your own control loop for the
call (the SDK's one-writer rule). `speak=False` / `animate=False` give you just
the text. The pieces are also usable à la carte: `reachy_search.moves` is a
beat-based choreography kit (thinking loaders, an attentive idle, an error
shake), `reachy_search.sounds` synthesizes the processing chirps, and
`reachy_search.main.match_wake` is the wake-word matcher.

## Production integrations: bring your own agent

If you already run an agent stack — your model, your persona, your
conversation history — you don't want our agent composing answers. You want
the robot-specific hard part handled and structured results back. That is
`GroundedSearch`: it extends a 2D agent into the 3D world. The robot's camera
frame becomes part of search planning; your agent keeps all the reasoning.

```python
from reachy_search import GroundedSearch

gs = GroundedSearch(anthropic_api_key=A, tavily_api_key=T)
r = gs.research("any news about them this week?",
                frame=jpeg,                       # optional
                context=my_agent.summary())       # your context steers the query
r.to_dict()   # -> feed your own LLM as a tool result
```

What it does for you, per call: resolves the question against the frame
(identifies the presented object, folds brand/model/material into the query)
and against your context (pronouns, domain), routes the search (general /
news / finance, time ranges, image results when the question wants visuals),
and returns `{query, search_type, object_seen, results[], images[],
engine_answer}` — or `recognized=False` plus a `clarification` line to relay
when it needs a better look. Your model writes every word your users hear.

## Using it inside the official conversation app

`integrations/conversation_app_tool.py` is a self-contained web-search tool
for [reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app).
Copy the file into the app's external tools directory
(`external_content/external_tools/`), add `TAVILY_API_KEY` to its `.env`,
and enable it under Tools → Tool access. No extra installs.

## Not a product

A weekend toy, MIT licensed. No accounts, no telemetry, no roadmap.
