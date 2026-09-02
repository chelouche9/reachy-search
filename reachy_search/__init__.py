"""Reachy Search — web search as a robot performance.

Runnable as a store app, importable as a library:

    from reachy_search import EmbodiedSearch     # the whole act
    from reachy_search import GroundedSearch     # retrieval only, you compose
"""

from .embodied import EmbodiedSearch
from .research import GroundedSearch, ResearchResult

__all__ = ["EmbodiedSearch", "GroundedSearch", "ResearchResult"]
