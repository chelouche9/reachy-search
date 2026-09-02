"""Web search tool for the official Reachy Mini conversation app.

Drop this file into your conversation app's external tools directory
(`REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`, usually
`external_content/external_tools/`), enable it under Tools -> Tool access
(or set AUTOLOAD_EXTERNAL_TOOLS=1), and put a Tavily API key in the app's
`.env` as TAVILY_API_KEY. Free tier at https://app.tavily.com/home.

Self-contained on purpose: no extra pip installs in the conversation app's
environment. From the reachy-search project — the standalone app with the
full embodied performance lives at the repo this file shipped in.
"""

import logging
import os
from typing import Any

import requests

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class ReachyWebSearch(Tool):
    """Give the conversation LLM live web search."""

    name = "reachy_web_search"
    description = (
        "Search the web for current information: prices, availability, news, "
        "facts you are not sure of, details about an object the user showed "
        "you. Use a specific query; include brand, model, material or size "
        "when known."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query.",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        query = kwargs.get("query", "").strip()
        if not query:
            return {"status": "error", "message": "Empty search query."}
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return {"status": "error",
                    "message": "TAVILY_API_KEY is not set in the app's .env."}

        logger.info("reachy_web_search: %r", query)
        try:
            response = requests.post(
                TAVILY_ENDPOINT,
                json={"api_key": api_key, "query": query, "max_results": 5,
                      "include_answer": True, "search_depth": "basic"},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.exception("Tavily request failed")
            return {"status": "error",
                    "message": f"Search failed: {type(exc).__name__}"}

        return {
            "status": "success",
            "answer": data.get("answer") or "",
            "results": [
                {"title": item.get("title", ""),
                 "url": item.get("url", ""),
                 "snippet": (item.get("content") or "")[:600]}
                for item in data.get("results", [])[:5]
            ],
        }
