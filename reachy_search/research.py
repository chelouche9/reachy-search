"""Grounded retrieval for production integrators.

The split that production apps want: WE handle the robot-specific hard part —
looking at the camera frame, merging it with the question and the caller's own
conversational context into the right search, routed to the right search type —
and hand back STRUCTURED results. THEIR model composes the answer, in their
persona, with their history. No text of ours reaches their users.

    from reachy_search import GroundedSearch

    gs = GroundedSearch(anthropic_api_key=A, tavily_api_key=T)
    r = gs.research("find me a cheaper one", frame=jpeg,
                    context="User runs a cafe; prior turn was about espresso gear.")
    r.object_seen    # "a 6-cup aluminium moka pot"
    r.query          # "aluminium 6-cup moka pot price"
    r.search_type    # "general" | "news" | "finance"
    r.results        # [{title, url, snippet}, ...]
    r.images         # image URLs when the question wants visuals
    r.to_dict()      # feed it straight to your own LLM as a tool result

Also works frameless (question-only) and recognizes when it can't see well
enough (`recognized=False` + `clarification` to relay to the user).
"""

import base64
import logging
from dataclasses import asdict, dataclass, field
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from . import config
from .search import MAX_SNIPPET_CHARS

logger = logging.getLogger(__name__)

MAX_TOKENS = 1500

PLAN_SYSTEM = """\
You plan one web search for a small camera-equipped robot. You get what the
person said, optionally a frame from the robot's camera (what it sees right
now), and optionally context from the host application about the ongoing
conversation or domain.

Write the single best search query. If the words point at something physical
in frame ("what is this", "find a cheaper one"), identify the presented object
and fold what you see — brand, model, material, size — into the query. If the
request stands alone, build the query from the words and context alone.

Route it: search_type "news" for current events, "finance" for market and
company financials, otherwise "general". Set time_range only when recency
matters. Set want_images true only if the person asked to see or find
something visual.

recognized=false ONLY if the request needs a visible object you cannot make
out; then write a short clarification the robot can relay."""


class _Plan(BaseModel):
    recognized: bool = True
    object_seen: str = Field(default="", description="What is presented, if anything.")
    query: str = ""
    search_type: Literal["general", "news", "finance"] = "general"
    time_range: Literal["", "day", "week", "month", "year"] = ""
    want_images: bool = False
    clarification: str = ""


@dataclass
class ResearchResult:
    question: str
    recognized: bool
    object_seen: str
    query: str
    search_type: str
    clarification: str
    engine_answer: str = ""
    results: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """The shape to hand your own LLM as a tool result."""
        return asdict(self)


class GroundedSearch:
    def __init__(self, anthropic_api_key: str, tavily_api_key: str,
                 model: str = config.CLAUDE_MODEL, effort: str = "low"):
        from tavily import TavilyClient

        self._client = anthropic.Anthropic(api_key=anthropic_api_key)
        self._tavily = TavilyClient(api_key=tavily_api_key)
        self._model = model
        self._output_config = {"effort": effort}

    def research(self, question: str, frame: bytes | None = None,
                 context: str = "") -> ResearchResult:
        plan = self._plan(question, frame, context)
        result = ResearchResult(
            question=question,
            recognized=plan.recognized,
            object_seen=plan.object_seen,
            query=plan.query,
            search_type=plan.search_type,
            clarification=plan.clarification,
        )
        if not plan.recognized or not plan.query:
            return result

        kwargs: dict = {
            "query": plan.query,
            "max_results": 5,
            "include_answer": True,
            "search_depth": "basic",
            "topic": plan.search_type,
        }
        if plan.time_range:
            kwargs["time_range"] = plan.time_range
        if plan.want_images:
            kwargs["include_images"] = True

        try:
            raw = self._tavily.search(**kwargs)
        except Exception:
            logger.exception("Search failed for %r", plan.query)
            return result

        result.engine_answer = raw.get("answer") or ""
        result.results = [
            {"title": item.get("title", "") or "",
             "url": item.get("url", "") or "",
             "snippet": (item.get("content") or "")[:MAX_SNIPPET_CHARS]}
            for item in raw.get("results", [])
        ]
        result.images = [img for img in (raw.get("images") or []) if isinstance(img, str)]
        logger.info("research: %r -> %s/%s, %d results, %d images",
                    question, plan.search_type, plan.query,
                    len(result.results), len(result.images))
        return result

    def _plan(self, question: str, frame: bytes | None, context: str) -> _Plan:
        content: list = []
        if frame is not None:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": base64.standard_b64encode(frame).decode("utf-8")},
            })
        text = f"They said: {question!r}"
        if context.strip():
            text += f"\n\nHost application context:\n{context.strip()}"
        content.append({"type": "text", "text": text})

        response = self._client.messages.parse(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=PLAN_SYSTEM,
            output_config=self._output_config,
            output_format=_Plan,
            messages=[{"role": "user", "content": content}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            return _Plan(recognized=False,
                         clarification="I'd rather not look that one up.")
        return response.parsed_output
