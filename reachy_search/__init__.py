"""Reachy Search — web search as a robot performance.

Runnable as a store app, importable as a library. One object, three verbs:

    from reachy_search import ReachySearch

    rs = ReachySearch(anthropic_api_key=..., tavily_api_key=..., mini=mini)
    rs.perform("find me a cheaper one")   # the whole act, on the robot
    rs.answer("what is this?", frame=j)   # our agent composes; you get text
    rs.research("news on them?", ...)     # structured results; you compose
"""

from .api import ReachySearch
from .research import ResearchResult

__all__ = ["ReachySearch", "ResearchResult"]
