"""Runtime configuration.

Keys come from the environment when it has them, otherwise from a small JSON
file the settings page writes. The app subprocess inherits the daemon's
environment, so `ANTHROPIC_API_KEY=... reachy-mini-daemon` works for developers,
while store users just paste keys into the settings page.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(
    os.environ.get("REACHY_SEARCH_CONFIG")
    or Path.home() / ".config" / "reachy_search" / "settings.json"
)

# Claude does the object identification and writes the spoken answer. Effort is
# pinned low deliberately: the thinking animation covers latency, it doesn't
# excuse it.
CLAUDE_MODEL = "claude-opus-5"
CLAUDE_EFFORT = "low"

# Whisper size. "base" transcribes a one-sentence question well enough and keeps
# the first-run download small; the Wireless CM4 wants "tiny".
WHISPER_MODEL = "base"

# Antenna press detection. Antennas run at low P gain, so a push shows up as a
# gap between what we commanded and where they actually are.
ANTENNA_PRESS_RAD = 0.25
ANTENNA_PRESS_TICKS = 3
ANTENNA_COOLDOWN_S = 2.0

# Wake word. Matched against a local Whisper transcript of short speech
# bursts, so include the ways Whisper typically mishears "Reachy".
WAKE_ENABLED = True
WAKE_WORDS = ("reachy", "reachie", "richie", "ritchie", "ricci", "richi", "reechee")
WAKE_MAX_S = 3.0
WAKE_HANG_S = 0.35
WAKE_MIN_SPEECH_S = 0.25
PROMPT_TEXT = "What can I do for you?"

# After answering, stay awake this long: follow-up questions need no wake word.
# Every exchange resets the window; "never mind" / "go to sleep" ends it early.
AWAKE_WINDOW_S = float(os.environ.get("REACHY_SEARCH_AWAKE_S", "60"))

# Spoken the moment the agent reaches for the search tool — the voice, like the
# loader move, honestly signals what is happening. Pre-rendered at warm-up.
SEARCH_LINES = ("Searching...", "Let me look that up.", "One moment.",
                "Hmm, let me check.")
SLEEP_PHRASES = ("go to sleep", "never mind", "nevermind", "that's all",
                 "stop listening", "goodbye", "good bye")

# Voice capture.
MAX_QUESTION_S = 8.0
SILENCE_HANG_S = 0.9
# Speech/silence gate. The GStreamer default source on macOS runs at very low
# gain (ambient measured ~0.0003 rms), so this sits ~8x above ambient rather
# than at a "normal" mic level. Override per machine if needed:
#   REACHY_SEARCH_SILENCE_RMS=0.001 python -m reachy_search.main
# and use dev_mic_meter.py to see live levels against the gate.
SILENCE_RMS = float(os.environ.get("REACHY_SEARCH_SILENCE_RMS", "0.0025"))
MIN_SPEECH_S = 0.4


@dataclass
class Settings:
    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    voice: str = "en_US-amy-medium"
    whisper_model: str = WHISPER_MODEL
    claude_model: str = CLAUDE_MODEL
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    sources: dict = field(default_factory=dict, repr=False)

    def describe_keys(self) -> str:
        """For the startup log: where each key came from, last 6 chars only."""
        parts = []
        for label, attr in (("anthropic", "anthropic_api_key"), ("tavily", "tavily_api_key")):
            value = getattr(self, attr)
            if value:
                parts.append(f"{label} ...{value[-6:]} from {self.sources.get(attr, '?')}")
            else:
                parts.append(f"{label} MISSING")
        return "; ".join(parts)

    @property
    def ready(self) -> bool:
        return bool(self.anthropic_api_key and self.tavily_api_key)

    def missing(self) -> list[str]:
        out = []
        if not self.anthropic_api_key:
            out.append("Anthropic")
        if not self.tavily_api_key:
            out.append("Tavily")
        return out

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if value and hasattr(self, key) and not key.startswith("_"):
                    setattr(self, key, value)
        self.save()

    def save(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "anthropic_api_key": self.anthropic_api_key,
            "tavily_api_key": self.tavily_api_key,
            "voice": self.voice,
            "whisper_model": self.whisper_model,
            "claude_model": self.claude_model,
        }
        SETTINGS_PATH.write_text(json.dumps(payload, indent=2))
        # Keys at rest — don't leave them world-readable.
        try:
            SETTINGS_PATH.chmod(0o600)
        except OSError:
            pass


def _load_dotenv() -> None:
    """Read a `.env` sitting next to the package into os.environ (no override).

    Keeps the dev loop to `python dev_webcam.py` with no exports. Deliberately
    tiny rather than a python-dotenv dependency.
    """
    for candidate in (Path.cwd() / ".env", Path(__file__).parent.parent / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value
        break


# A project-local .env next to pyproject.toml (the checkout root). Explicit,
# per-project, gitignored — and it outranks an ambient shell export, because a
# stale key lingering in a terminal is a classic way to keep hitting a dead
# account after you have already pasted a fresh one.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            name, _, value = line.partition("=")
            values[name.strip()] = value.strip().strip("\"'")
    except OSError as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
    return values


def load() -> Settings:
    """Precedence, lowest to highest: settings.json, shell environment,
    project .env. `sources` records where each key came from, for the log."""
    settings = Settings()

    if SETTINGS_PATH.exists():
        try:
            stored = json.loads(SETTINGS_PATH.read_text())
            for key in ("anthropic_api_key", "tavily_api_key", "voice",
                        "whisper_model", "claude_model"):
                if stored.get(key):
                    setattr(settings, key, stored[key])
                    settings.sources[key] = "settings.json"
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable settings file: %s", exc)

    pairs = (("ANTHROPIC_API_KEY", "anthropic_api_key"),
             ("TAVILY_API_KEY", "tavily_api_key"))
    for env_name, attr in pairs:
        if os.environ.get(env_name):
            setattr(settings, attr, os.environ[env_name])
            settings.sources[attr] = "shell environment"

    if ENV_FILE.exists():
        file_values = _read_env_file(ENV_FILE)
        for env_name, attr in pairs:
            if file_values.get(env_name):
                setattr(settings, attr, file_values[env_name])
                settings.sources[attr] = ".env"

    return settings
