"""Seam-1 tests for the evaluation façade.

The evaluator is a thin harness over ``QueryEngine.query``; these check that it
scores answer matches and citation precision/recall against a labelled set.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ragsage.evaluation import EvalExample, EvalThresholds
from ragsage.models import PageImage
from ragsage.query import NOT_FOUND_MESSAGE

from ragsage import (
    Evaluator,
    IngestionPipeline,
    QueryEngine,
    RawSource,
    Scope,
    default_golden_set,
    run_golden_eval,
)

FRANCE = b"Paris is the capital of France."
BIOLOGY = b"The mitochondrion produces ATP in the cell."


async def test_evaluate_scores_answers_and_citations(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    france = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)
    await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), scope)

    dataset = [
        EvalExample(
            question="What is the capital of France?",
            expected_answer="Paris",
            relevant_document_ids=frozenset({france.document.id}),
        ),
        EvalExample(question="Who won the 1998 world cup?", expected_answer=None),
    ]
    report = await Evaluator(engine, scope).evaluate(dataset)

    assert len(report.results) == 2
    # First example: answered correctly and cited the right document.
    first = report.results[0]
    assert first.answer_matched is True
    assert first.citation_precision == 1.0
    assert first.citation_recall == 1.0
    # Second example has no expected answer, so it matches vacuously but is
    # ungrounded (not in the corpus).
    assert report.results[1].answer.grounded is False
    assert 0.0 < report.grounded_rate < 1.0


async def test_evaluate_of_empty_dataset_is_zeroed(engine: QueryEngine, scope: Scope) -> None:
    report = await Evaluator(engine, scope).evaluate([])

    assert report.results == ()
    assert report.answer_accuracy == 0.0
    assert report.grounded_rate == 0.0


# --------------------------------------------------------------------------- #
# Labeled fixture set — the retrieval & citation correctness gate (issue 06)
# --------------------------------------------------------------------------- #

# A small multi-document corpus with hand-labeled questions. Each grounded
# example names the exact document a correct answer must cite, so the aggregate
# scores below measure both dimensions the product promises: answered from the
# corpus, and cited to the right source. The set deliberately mixes topics so a
# right answer requires actually retrieving the relevant document, not guessing.
_CORPUS: dict[str, bytes] = {
    "france.txt": b"Paris is the capital of France. It sits on the Seine river.",
    "biology.txt": b"The mitochondrion is the powerhouse of the cell and produces ATP.",
    "python.txt": b"Python is a high-level programming language created by Guido van Rossum.",
    "space.txt": b"The Moon is Earth's only natural satellite and orbits the planet monthly.",
}


async def test_labeled_fixture_set_scores_perfectly(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    ids = {
        name: (await pipeline.ingest(RawSource(name=name, content=body), scope)).document.id
        for name, body in _CORPUS.items()
    }

    dataset = [
        EvalExample(
            question="What is the capital of France?",
            expected_answer="Paris",
            relevant_document_ids=frozenset({ids["france.txt"]}),
        ),
        EvalExample(
            question="What produces ATP in the cell?",
            expected_answer="ATP",
            relevant_document_ids=frozenset({ids["biology.txt"]}),
        ),
        EvalExample(
            question="Who created the Python programming language?",
            expected_answer="Guido",
            relevant_document_ids=frozenset({ids["python.txt"]}),
        ),
        EvalExample(
            question="What is Earth's only natural satellite?",
            expected_answer="Moon",
            relevant_document_ids=frozenset({ids["space.txt"]}),
        ),
    ]

    report = await Evaluator(engine, scope).evaluate(dataset)

    # Every grounded example is answered from the corpus and cited to exactly the
    # labeled source — no misses, no citations to the wrong document.
    assert report.answer_accuracy == 1.0
    assert report.grounded_rate == 1.0
    assert report.mean_citation_precision == 1.0
    assert report.mean_citation_recall == 1.0


async def test_labeled_out_of_corpus_question_is_honest_not_found(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    for name, body in _CORPUS.items():
        await pipeline.ingest(RawSource(name=name, content=body), scope)

    # A question the corpus can't answer must not be grounded or cite anything —
    # the labeled negative that guards against confident hallucination.
    dataset = [EvalExample(question="What is the boiling point of mercury in Kelvin?")]
    report = await Evaluator(engine, scope).evaluate(dataset)

    assert report.grounded_rate == 0.0
    assert report.results[0].answer.citations == ()


# --------------------------------------------------------------------------- #
# The RAG metrics — faithfulness, relevancy, context precision/recall (issue 10)
# --------------------------------------------------------------------------- #


async def test_metrics_score_a_grounded_answer_well(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    france = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)

    dataset = [
        EvalExample(
            question="What is the capital of France?",
            expected_answer="Paris",
            ground_truth_answer="Paris is the capital of France.",
            relevant_document_ids=frozenset({france.document.id}),
        )
    ]
    report = await Evaluator(engine, scope).evaluate(dataset)

    case = report.results[0]
    # A grounded, correctly-cited answer scores full marks on every metric.
    assert case.faithfulness == 1.0
    assert case.answer_relevancy == 1.0
    assert case.context_precision == 1.0
    assert case.context_recall == 1.0
    # The historical citation aliases still read through.
    assert case.citation_precision == 1.0
    assert report.mean_citation_recall == 1.0


async def test_hallucinated_answer_scores_low_faithfulness(
    pipeline: IngestionPipeline, kit, scope: Scope
) -> None:
    # A model that ignores its context and invents an unsupported answer while
    # still citing the source: the failure faithfulness is meant to catch.
    await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)

    class HallucinatingLLM:
        async def generate(self, prompt: str) -> AsyncIterator[str]:
            for word in "The capital is Berlin according to legend".split():
                yield word + " "
            yield "[1]"

        async def transcribe(self, image: PageImage) -> str:  # pragma: no cover
            return ""

    engine = QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=HallucinatingLLM(),
    )
    dataset = [
        EvalExample(
            question="What is the capital of France?",
            ground_truth_answer="Paris is the capital of France.",
        )
    ]
    report = await Evaluator(engine, scope).evaluate(dataset)

    # "Berlin/legend" is not supported by the Paris source, so faithfulness drops
    # well below the default threshold even though the answer cites a real chunk.
    assert report.mean_faithfulness < EvalThresholds().faithfulness


async def test_golden_set_meets_its_thresholds() -> None:
    # The offline CI gate: the built-in golden set must clear every metric floor.
    golden = default_golden_set()
    report = await run_golden_eval(golden)

    check = report.check(golden.thresholds)
    assert check.passed, f"golden eval below threshold: {check.failures}"
    assert report.answer_accuracy == 1.0
    assert report.mean_context_precision == 1.0
    assert report.mean_context_recall == 1.0
    # Four grounded questions plus one honest not-found negative.
    assert report.grounded_rate == 0.8


def test_thresholds_report_each_metric_that_fell_short() -> None:
    from ragsage.evaluation import EvalReport

    report = EvalReport(
        results=(),
        answer_accuracy=1.0,
        mean_faithfulness=0.5,
        mean_answer_relevancy=1.0,
        mean_context_precision=0.4,
        mean_context_recall=1.0,
        grounded_rate=1.0,
    )
    check = report.check(EvalThresholds())

    assert check.passed is False
    joined = " ".join(check.failures)
    assert "faithfulness" in joined
    assert "context_precision" in joined
    assert "answer_relevancy" not in joined  # this one cleared its floor


def test_not_found_message_is_the_negative_example_ground_truth() -> None:
    # Guards the golden set's design: the out-of-corpus example rewards a refusal
    # by using the engine's own not-found message as its ground truth.
    golden = default_golden_set()
    negatives = [ex for ex in golden.examples if ex.ground_truth_answer == NOT_FOUND_MESSAGE]
    assert len(negatives) == 1
    assert negatives[0].relevant_sources == frozenset()
