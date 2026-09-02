"""The public API: one object, three verbs.

    from reachy_search import ReachySearch

    rs = ReachySearch(anthropic_api_key=A, tavily_api_key=T, mini=mini)

    rs.perform("find me a cheaper one")     # the whole act: moves, chirps, voice
    rs.answer("what is this?", frame=jpeg)  # our agent composes; you get text
    rs.research("any news on them?",        # structured results; YOU compose
                context=my_agent.summary())

The verbs are ordered by how much you delegate to us. `perform` needs a
connected robot; `answer` and `research` run anywhere (frame optional).
`reset()` clears the conversational memory behind `answer`.

Everything else in this package is machinery behind these three verbs — plus
the à-la-carte kit (`reachy_search.moves`, `reachy_search.sounds`) for apps
that only want the personality.
"""

from __future__ import annotations

from . import config, tts


class ReachySearch:
    def __init__(
        self,
        anthropic_api_key: str,
        tavily_api_key: str,
        mini=None,
        model: str = config.CLAUDE_MODEL,
        voice: str = tts.DEFAULT_VOICE,
        effort: str = "low",
    ):
        self._anthropic_key = anthropic_api_key
        self._tavily_key = tavily_api_key
        self._mini = mini
        self._model = model
        self._voice = voice
        self._effort = effort
        # Each verb builds its engine on first use, so `research`-only users
        # never load a voice and `answer`-only users never touch the robot.
        self._performer = None
        self._agent = None
        self._researcher = None

    # ------------------------------------------------------------------
    # The three verbs
    # ------------------------------------------------------------------

    def perform(self, question: str, frame: bytes | None = None) -> str:
        """The whole act on the robot: thinking move, processing chirps, a
        spoken "Searching...", and the answer out loud. Returns the text too.
        Owns `set_target` for the duration — pause your control loop."""
        if self._mini is None:
            raise ValueError("perform() needs a connected robot: "
                             "ReachySearch(..., mini=<ReachyMini>)")
        if self._performer is None:
            from .embodied import EmbodiedSearch

            self._performer = EmbodiedSearch(
                self._mini,
                anthropic_api_key=self._anthropic_key,
                tavily_api_key=self._tavily_key,
                voice=self._voice, model=self._model,
            )
            self._performer.warm_up()
        return self._performer.ask(question, frame=frame)

    def answer(self, question: str, frame: bytes | None = None) -> str:
        """Our agent looks, decides whether to search, and composes a short
        spoken-style answer. You do the speaking and the moving. Keeps a
        conversational memory across calls — clear it with `reset()`."""
        if self._agent is None:
            from .brain import Brain
            from .search import Searcher

            self._agent = Brain(
                api_key=self._anthropic_key, model=self._model,
                searcher=Searcher(self._tavily_key), effort=self._effort,
            )
        if frame is None and self._mini is not None:
            frame = self._grab()
        return self._agent.respond(frame, question)

    def research(self, question: str, frame: bytes | None = None,
                 context: str = ""):
        """Grounded retrieval only: we plan and run the right search from the
        frame, the question, and YOUR context; your model composes from the
        structured `ResearchResult` (`.to_dict()` for a tool-result slot)."""
        if self._researcher is None:
            from .research import GroundedSearch

            self._researcher = GroundedSearch(
                anthropic_api_key=self._anthropic_key,
                tavily_api_key=self._tavily_key,
                model=self._model, effort=self._effort,
            )
        if frame is None and self._mini is not None:
            frame = self._grab()
        return self._researcher.research(question, frame=frame, context=context)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the conversation behind `answer()`."""
        if self._agent is not None:
            self._agent.reset()

    def _grab(self) -> bytes | None:
        try:
            return self._mini.media.get_frame_jpeg()
        except Exception:
            return None
