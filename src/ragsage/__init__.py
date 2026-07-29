"""ragsage — a sage that only speaks from your corpus.

The reusable, tenancy-agnostic RAG engine. This package owns the RAG *domain*
(parsing, chunking, embedding, retrieval, reranking, generation, citations)
behind provider-agnostic ports. It never imports web, auth, or tenant concepts:
callers pass an opaque :class:`Scope` (namespace + metadata filters) and the
engine treats it as an untyped label.

The public surface is three façades — :class:`IngestionPipeline`,
:class:`QueryEngine`, :class:`Evaluator` — driven through the ports in
:mod:`ragsage.ports`, over the domain models in :mod:`ragsage.models`. Ready-made
in-memory adapters live in :mod:`ragsage.fakes` so the whole loop runs offline.

A few ports also have a *real*, model-free implementation shipped in the library
rather than a fake. They stay in their own modules so importing ``ragsage`` never
drags a parser or a tokenizer in:
:class:`ragsage.parsing.HeuristicBackend` (parse + chunk) and
:class:`ragsage.contextualizing.HeadingWindowContextualizer` (deterministic
contextual retrieval).
"""

from ragsage.config import IngestionConfig, QueryOptions
from ragsage.evaluation import (
    EvalCaseResult,
    EvalExample,
    EvalReport,
    EvalThresholds,
    Evaluator,
    ThresholdCheck,
)
from ragsage.goldens import GoldenExample, GoldenSet, default_golden_set, run_golden_eval
from ragsage.ingestion import IngestionPipeline, IngestionResult
from ragsage.models import (
    Answer,
    AnswerComplete,
    AnswerEvent,
    AnswerToken,
    Chunk,
    Citation,
    Document,
    EmbeddedChunk,
    Outcome,
    Page,
    PageImage,
    PageLayout,
    PageRoute,
    ParsedDocument,
    RawSource,
    ScoredChunk,
    Turn,
    Usage,
    Vector,
)
from ragsage.query import NOT_FOUND_MESSAGE, QueryEngine
from ragsage.scope import Scope
from ragsage.smalltalk import SMALL_TALK, is_small_talk

__version__ = "0.1.0"

__all__ = [
    "NOT_FOUND_MESSAGE",
    "SMALL_TALK",
    "Answer",
    "AnswerComplete",
    "AnswerEvent",
    "AnswerToken",
    "Chunk",
    "Citation",
    "Document",
    "EmbeddedChunk",
    "EvalCaseResult",
    "EvalExample",
    "EvalReport",
    "EvalThresholds",
    "Evaluator",
    "GoldenExample",
    "GoldenSet",
    "IngestionConfig",
    "IngestionPipeline",
    "IngestionResult",
    "Outcome",
    "Page",
    "PageImage",
    "PageLayout",
    "PageRoute",
    "ParsedDocument",
    "QueryEngine",
    "QueryOptions",
    "RawSource",
    "Scope",
    "ScoredChunk",
    "ThresholdCheck",
    "Turn",
    "Usage",
    "Vector",
    "__version__",
    "default_golden_set",
    "is_small_talk",
    "run_golden_eval",
]
