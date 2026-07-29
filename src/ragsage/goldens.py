"""A built-in golden set and a runnable offline eval, shipped in the library.

The eval harness needs something to score. This module carries a small,
hand-labelled corpus plus the questions a correct pipeline must answer (and one
it must *decline*), so ``ragsage eval`` — and the CI test — run the full
ingest-then-score loop against the in-memory :mod:`ragsage.fakes` with no
network, exactly as the CLI proves the rest of the engine.

Examples reference their relevant documents by *filename*; :func:`run_golden_eval`
ingests the corpus, maps each filename to the document id the parser assigned,
and hands real :class:`~ragsage.evaluation.EvalExample`\\s to the evaluator. A
deploy points the same machinery at its own corpus and labels by supplying a
different :class:`GoldenSet`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ragsage.config import IngestionConfig
from ragsage.evaluation import EvalExample, EvalReport, EvalThresholds, Evaluator
from ragsage.fakes import FakeEngineKit
from ragsage.ingestion import IngestionPipeline
from ragsage.models import RawSource
from ragsage.ports import Contextualizer
from ragsage.query import NOT_FOUND_MESSAGE, QueryEngine
from ragsage.scope import Scope


@dataclass(frozen=True)
class GoldenExample:
    """A labelled question keyed to its relevant documents by filename."""

    question: str
    expected_answer: str | None = None
    ground_truth_answer: str | None = None
    relevant_sources: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class GoldenSet:
    """A corpus, its labelled questions, and the thresholds to gate against."""

    corpus: Mapping[str, bytes]
    examples: Sequence[GoldenExample]
    thresholds: EvalThresholds = field(default_factory=EvalThresholds)


_CORPUS: dict[str, bytes] = {
    "france.txt": b"Paris is the capital of France. It sits on the Seine river.",
    "biology.txt": b"The mitochondrion is the powerhouse of the cell and produces ATP.",
    "python.txt": b"Python is a high-level programming language created by Guido van Rossum.",
    "space.txt": b"The Moon is Earth's only natural satellite and orbits the planet monthly.",
}

_EXAMPLES: list[GoldenExample] = [
    GoldenExample(
        question="What is the capital of France?",
        expected_answer="Paris",
        ground_truth_answer="Paris is the capital of France.",
        relevant_sources=frozenset({"france.txt"}),
    ),
    GoldenExample(
        question="What produces ATP in the cell?",
        expected_answer="ATP",
        ground_truth_answer="The mitochondrion produces ATP in the cell.",
        relevant_sources=frozenset({"biology.txt"}),
    ),
    GoldenExample(
        question="Who created the Python programming language?",
        expected_answer="Guido",
        ground_truth_answer="Guido van Rossum created the Python programming language.",
        relevant_sources=frozenset({"python.txt"}),
    ),
    GoldenExample(
        question="What is Earth's only natural satellite?",
        expected_answer="Moon",
        ground_truth_answer="The Moon is Earth's only natural satellite.",
        relevant_sources=frozenset({"space.txt"}),
    ),
    # The labelled negative: out of corpus, so the right response is an honest
    # not-found. Its ground truth is that very message, so answer relevancy and
    # faithfulness reward the refusal instead of penalising it.
    GoldenExample(
        question="What is the boiling point of mercury in Kelvin?",
        ground_truth_answer=NOT_FOUND_MESSAGE,
    ),
]


def default_golden_set() -> GoldenSet:
    """The library's built-in golden set — a small multi-topic corpus."""
    return GoldenSet(corpus=dict(_CORPUS), examples=list(_EXAMPLES))


async def run_golden_eval(
    golden: GoldenSet | None = None,
    *,
    contextualizer: Contextualizer | None = None,
    config: IngestionConfig | None = None,
) -> EvalReport:
    """Ingest the golden corpus into fresh fakes and score its questions.

    Wires a :class:`FakeEngineKit` exactly as the CLI does, ingests every corpus
    document, resolves each example's filenames to the ids the parser assigned,
    and returns the evaluator's :class:`EvalReport`. Fully offline and
    deterministic — the same run every time, so it can gate CI.

    ``contextualizer`` and ``config`` exist so one corpus can be scored under
    different contextualisation arms — the comparison
    :class:`~ragsage.contextualizing.HeadingWindowContextualizer` had to justify
    itself against. They default to the fake and to
    :class:`~ragsage.config.IngestionConfig`'s own defaults, so an existing caller
    sees no change.
    """
    golden = golden or default_golden_set()
    kit = FakeEngineKit()
    pipeline = IngestionPipeline(
        parser=kit.parser,
        classifier=kit.classifier,
        chunker=kit.chunker,
        contextualizer=contextualizer if contextualizer is not None else kit.contextualizer,
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        llm=kit.llm,
        cache=kit.cache,
        tracer=kit.tracer,
    )
    scope = Scope(namespace="golden-eval")

    ids_by_source: dict[str, str] = {}
    for name, body in golden.corpus.items():
        result = await pipeline.ingest(RawSource(name=name, content=body), scope, config)
        ids_by_source[name] = result.document.id

    dataset = [
        EvalExample(
            question=ex.question,
            expected_answer=ex.expected_answer,
            ground_truth_answer=ex.ground_truth_answer,
            relevant_document_ids=frozenset(
                ids_by_source[name] for name in ex.relevant_sources if name in ids_by_source
            ),
        )
        for ex in golden.examples
    ]

    engine = QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
        tracer=kit.tracer,
    )
    return await Evaluator(engine, scope).evaluate(dataset)
