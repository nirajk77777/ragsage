"""The evaluation façade — measure answer quality on a fixed golden set.

:class:`Evaluator` runs a labelled dataset through a :class:`~ragsage.query.QueryEngine`
and scores the answers it gets back on the four standard RAG metrics —
**faithfulness**, **answer relevancy**, **context precision**, and **context
recall** — alongside the answer/citation correctness backbone the Seam-1 tests
rely on. It is deliberately a thin, offline harness over the public query seam:
the same deterministic-fakes setup the tests use, so quality can be tracked
against a golden set with no network and gated in CI against agreed thresholds.

The metrics here are **reference-based and deterministic** — computed from the
answer, its citations, and the golden labels by content-token overlap, *not* by
an LLM judge. That is a deliberate fit for this stack: the whole engine runs
offline against fakes in CI (no API keys), so a judge-model RAGAS/deepeval run
would not run there. Swapping in an LLM-judged scorer for a richer, subjective
read is a follow-up that plugs in behind this same :class:`Evaluator` seam; what
lives here is the gate that *can* run on every push.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragsage.config import QueryOptions
from ragsage.models import Answer
from ragsage.query import QueryEngine
from ragsage.scope import Scope

_WORD = re.compile(r"[a-z0-9]+")
_MARKER = re.compile(r"\[\d+\]")
# A tiny stop-word list so scoring keys on *content* words: shared filler never
# on its own makes an answer look faithful or relevant.
_STOPWORDS = frozenset(
    "a an and are as at be by do for from how in is it its of on or that the "
    "this to was were what when where which who will with".split()
)


def _content_tokens(text: str) -> set[str]:
    """Lower-cased content tokens (stop-words and ``[n]`` citation markers dropped).

    Citation markers are stripped first so a ``[1]`` in an answer never counts as
    an unsupported "claim" against the context it points at.
    """
    stripped = _MARKER.sub(" ", text)
    return {t for t in _WORD.findall(stripped.lower()) if t not in _STOPWORDS}


def _coverage(reference: str, produced: str) -> float:
    """Fraction of ``reference``'s content tokens that appear in ``produced``.

    1.0 when the reference carries no content tokens (nothing to cover), so a
    vacuous reference never drags an aggregate down.
    """
    ref = _content_tokens(reference)
    if not ref:
        return 1.0
    return len(ref & _content_tokens(produced)) / len(ref)


@dataclass(frozen=True)
class EvalExample:
    """One labelled question: what to ask and what a right answer looks like.

    ``expected_answer`` is a substring the answer should contain (case-folded).
    ``ground_truth_answer`` is the reference answer relevancy and faithfulness
    score against — for an out-of-corpus question set it to the engine's
    not-found message, so an honest refusal scores as the relevant response.
    ``relevant_document_ids`` are the documents a good answer should draw on
    (context precision/recall). Any may be omitted to score only the others.
    """

    question: str
    expected_answer: str | None = None
    ground_truth_answer: str | None = None
    relevant_document_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EvalThresholds:
    """The agreed metric floors an eval run is gated against in CI.

    Defaults are conservative pass marks for the deterministic golden set; a
    deploy can tighten them. :meth:`EvalReport.check` compares aggregate means
    against these.
    """

    faithfulness: float = 0.70
    answer_relevancy: float = 0.60
    context_precision: float = 0.70
    context_recall: float = 0.70


@dataclass(frozen=True)
class EvalCaseResult:
    """The scored outcome for a single example."""

    example: EvalExample
    answer: Answer
    answer_matched: bool
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    # Citation precision/recall are the same measurement as context
    # precision/recall in this cited-grounded design (the surfaced context *is*
    # the citations); kept under their historical names for the Seam-1 tests.
    @property
    def citation_precision(self) -> float:
        return self.context_precision

    @property
    def citation_recall(self) -> float:
        return self.context_recall


@dataclass(frozen=True)
class ThresholdCheck:
    """Whether an :class:`EvalReport` cleared its thresholds, and what didn't."""

    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class EvalReport:
    """Aggregate scores across the dataset, plus the per-case breakdown."""

    results: Sequence[EvalCaseResult]
    answer_accuracy: float
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: float
    grounded_rate: float

    # Historical aliases kept so existing callers/tests keep reading.
    @property
    def mean_citation_precision(self) -> float:
        return self.mean_context_precision

    @property
    def mean_citation_recall(self) -> float:
        return self.mean_context_recall

    def check(self, thresholds: EvalThresholds) -> ThresholdCheck:
        """Compare the aggregate means against ``thresholds``.

        Returns which metrics fell short (empty when all clear), so CI can print
        the offenders and exit non-zero on any miss.
        """
        checks = {
            "faithfulness": (self.mean_faithfulness, thresholds.faithfulness),
            "answer_relevancy": (self.mean_answer_relevancy, thresholds.answer_relevancy),
            "context_precision": (self.mean_context_precision, thresholds.context_precision),
            "context_recall": (self.mean_context_recall, thresholds.context_recall),
        }
        failures = tuple(
            f"{name} {value:.3f} < {floor:.3f}"
            for name, (value, floor) in checks.items()
            if value < floor
        )
        return ThresholdCheck(passed=not failures, failures=failures)


def _context_scores(cited: set[str], relevant: frozenset[str]) -> tuple[float, float]:
    """Precision and recall of the surfaced (cited) documents against the labels.

    With no labelled relevant set the question isn't scored on context (both
    1.0); with labels but no citations it scored nothing right (both 0.0).
    """
    if not relevant:
        return (1.0, 1.0)
    if not cited:
        return (0.0, 0.0)
    hits = len(cited & relevant)
    return (hits / len(cited), hits / len(relevant))


def _faithfulness(answer: Answer) -> float:
    """How much of the answer is supported by the context it cited.

    An honest not-found invents nothing, so it is faithful by definition (1.0).
    A grounded answer scores the fraction of its content tokens present in the
    union of its cited passages — a hallucinated claim with no citation, or one
    citing unrelated text, drives this down.
    """
    if not answer.grounded:
        return 1.0
    context = " ".join(c.quote for c in answer.citations)
    if not context.strip():
        # Grounded yet citing nothing to stand on — no support at all.
        return 0.0
    return _coverage(answer.text, context)


def _answer_relevancy(example: EvalExample, answer: Answer) -> float:
    """How well the answer addresses the question.

    Scored against the ground-truth answer when the example provides one, else
    the expected-answer substring, else the question itself — the fraction of the
    reference's content tokens the answer covers.
    """
    reference = example.ground_truth_answer or example.expected_answer or example.question
    return _coverage(reference, answer.text)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class Evaluator:
    """Scores a :class:`QueryEngine` against a labelled dataset for a scope."""

    def __init__(
        self,
        engine: QueryEngine,
        scope: Scope,
        *,
        options: QueryOptions | None = None,
    ) -> None:
        self._engine = engine
        self._scope = scope
        self._options = options

    async def evaluate(self, dataset: Sequence[EvalExample]) -> EvalReport:
        """Run every example through the engine and aggregate the scores."""
        if not dataset:
            return EvalReport(
                results=(),
                answer_accuracy=0.0,
                mean_faithfulness=0.0,
                mean_answer_relevancy=0.0,
                mean_context_precision=0.0,
                mean_context_recall=0.0,
                grounded_rate=0.0,
            )

        results: list[EvalCaseResult] = []
        for example in dataset:
            answer = await self._engine.query(example.question, self._scope, self._options)
            matched = (
                example.expected_answer is None
                or example.expected_answer.casefold() in answer.text.casefold()
            )
            cited_docs = {c.document_id for c in answer.citations}
            precision, recall = _context_scores(cited_docs, example.relevant_document_ids)
            results.append(
                EvalCaseResult(
                    example=example,
                    answer=answer,
                    answer_matched=matched,
                    faithfulness=_faithfulness(answer),
                    answer_relevancy=_answer_relevancy(example, answer),
                    context_precision=precision,
                    context_recall=recall,
                )
            )

        return EvalReport(
            results=results,
            answer_accuracy=_mean([float(r.answer_matched) for r in results]),
            mean_faithfulness=_mean([r.faithfulness for r in results]),
            mean_answer_relevancy=_mean([r.answer_relevancy for r in results]),
            mean_context_precision=_mean([r.context_precision for r in results]),
            mean_context_recall=_mean([r.context_recall for r in results]),
            grounded_rate=_mean([float(r.answer.grounded) for r in results]),
        )
