"""Parsing: the page-route classifier and the heuristic parse/chunk backend.

``ragsage`` owns the parse/chunk *ports* and their fakes; this package holds the
library's real, model-free implementations of them:

* :class:`LayoutPageClassifier` — the parser-agnostic ``TEXT``/``VISION`` routing
  policy (reads only a :class:`~ragsage.models.PageLayout`).
* :class:`HeuristicBackend` — a pure-Python backend that satisfies both the
  ``DocumentParser`` and ``Chunker`` ports, with a format router and the shared
  two-pass structure-aware Markdown chunker.

Both are re-exported here so callers import from ``ragsage.parsing`` regardless of
the internal module split.
"""

from ragsage.parsing.backend import HeuristicBackend
from ragsage.parsing.classifier import LayoutPageClassifier
from ragsage.parsing.router import DocumentFormat, route

__all__ = [
    "DocumentFormat",
    "HeuristicBackend",
    "LayoutPageClassifier",
    "route",
]
