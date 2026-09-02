"""The agent.

One Claude call per question, with eyes and a tool. Every question arrives with
a camera frame attached — what Reachy is seeing right now — and a `web_search`
tool Claude may or may not reach for. "What do you see?" gets answered straight
from the frame; "find me a cheaper one" identifies the object and searches.
The model decides, which is the whole point.

A short text-only conversation memory rides along, so "and a cheaper one?"
works on the next wake."""

import base64
import logging

import anthropic
from anthropic import beta_tool

logger = logging.getLogger(__name__)

MAX_TOKENS = 2000
HISTORY_TURNS = 6  # question/answer pairs kept for follow-ups

SYSTEM = """\
You are Reachy Mini, a small expressive desk robot. Someone just spoke to you.
Each question arrives with a frame from your camera — that is what you are
seeing right now, so questions like "what do you see?" are answered directly
from it.

You have a web_search tool. Use it when the answer benefits from a lookup —
prices, availability, facts you're not sure of, anything current. Skip it when
you can already answer well: describing what you see, simple knowledge, chat.
When someone shows you an object and asks about it, fold what you can see into
the search query — brand, model, material, size make queries that actually
answer the question.

Everything you write is spoken aloud through a speech synthesiser, so:
- Two or three sentences. Under 55 words. Hard limit.
- Answer the question first; context only if it earns its place.
- No markdown, no lists, no URLs, no emoji.
- Say numbers the way a person says them out loud: "about thirty dollars".
- Plain, warm, a little curious. A small robot that just looked something up,
  not a search engine reading itself aloud.
- If you can't make out an object you clearly need to see, ask them to hold it
  closer instead of guessing. If a search fails or answers nothing, say so
  briefly — being wrong out loud is worse than being short."""


class RefusalError(RuntimeError):
    """Claude declined the request. Not a bug — say something and move on."""


class Brain:
    def __init__(self, api_key: str, model: str, searcher=None, effort: str = "low",
                 on_search=None):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._output_config = {"effort": effort}
        self._history: list[dict] = []
        self._tools = []
        if searcher is None:
            # No search tool registered: a pure look-and-answer agent.
            return

        @beta_tool
        def web_search(query: str, topic: str = "general",
                       time_range: str = "") -> str:
            """Search the web for current information.

            Args:
                query: A specific search query. Include brand, model, material
                    or size when visible on the object in question.
                topic: "news" for current events, "finance" for markets and
                    company financials, otherwise "general".
                time_range: Restrict recency when it matters: "day", "week",
                    "month" or "year". Empty string for no restriction.
            """
            if on_search is not None:
                try:
                    on_search(query)
                except Exception:
                    logger.debug("Search announcement failed", exc_info=True)
            try:
                results = searcher.search(query, topic=topic, time_range=time_range)
                return results.as_context() or "The search returned no results."
            except Exception as exc:
                logger.exception("Search tool failed")
                return f"The search failed ({type(exc).__name__}). Answer from what you know and say you couldn't look it up."

        self._tools = [web_search]

    def reset(self) -> None:
        """Forget the conversation so far."""
        self._history = []

    def respond(self, jpeg: bytes | None, question: str) -> str:
        content: list = []
        if jpeg is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(jpeg).decode("utf-8"),
                },
            })
        content.append({"type": "text", "text": question})

        runner = self._client.beta.messages.tool_runner(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            output_config=self._output_config,
            tools=self._tools,
            messages=self._history + [{"role": "user", "content": content}],
        )

        last = None
        for message in runner:
            last = message
            if message.stop_reason == "refusal":
                details = getattr(message, "stop_details", None)
                raise RefusalError(getattr(details, "category", None) or "unspecified")

        if last is None:
            return ""
        spoken = " ".join(
            block.text.strip() for block in last.content if block.type == "text"
        ).strip()

        # Text-only memory: the frame is only ever "now", so old frames would
        # just be stale context at image-token prices.
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": spoken or "..."})
        self._history = self._history[-HISTORY_TURNS * 2:]

        logger.info("Answer: %r", spoken)
        return spoken
