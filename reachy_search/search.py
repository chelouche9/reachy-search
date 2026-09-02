"""Tavily web search.

Tavily is text-in only — it can return images but cannot accept one. That
constraint is the whole reason the vision pass exists: Claude is the bridge from
the object in front of the camera to a query a search engine can answer.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 700


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class SearchResponse:
    query: str
    answer: str
    results: list[SearchResult]

    @property
    def empty(self) -> bool:
        return not self.answer and not self.results

    def as_context(self) -> str:
        """Flatten into something worth spending Claude's context on."""
        chunks = []
        if self.answer:
            chunks.append(f"Search engine summary: {self.answer}")
        for i, result in enumerate(self.results, 1):
            chunks.append(f"[{i}] {result.title}\n{result.snippet}")
        return "\n\n".join(chunks)


class Searcher:
    def __init__(self, api_key: str):
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, topic: str = "general",
               time_range: str = "") -> SearchResponse:
        kwargs = dict(
            query=query,
            max_results=MAX_RESULTS,
            # Tavily's own one-line synthesis is usually the single most useful
            # thing in the payload for a spoken answer.
            include_answer=True,
            # A robot is standing there visibly thinking: latency is quality.
            search_depth="fast",
        )
        if topic in ("news", "finance"):
            kwargs["topic"] = topic
        if time_range in ("day", "week", "month", "year"):
            kwargs["time_range"] = time_range
        raw = self._client.search(**kwargs)

        results = [
            SearchResult(
                title=item.get("title", "") or "",
                url=item.get("url", "") or "",
                snippet=(item.get("content", "") or "")[:MAX_SNIPPET_CHARS],
            )
            for item in raw.get("results", [])
        ]
        response = SearchResponse(
            query=query,
            answer=raw.get("answer", "") or "",
            results=results,
        )
        logger.info(
            "Tavily: %d results for %r (answer: %s)",
            len(results), query, "yes" if response.answer else "no",
        )
        return response
