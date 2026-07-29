"""Seam-1 tests for multi-turn context: history-aware query rewriting.

A follow-up like "what about its pricing?" only retrieves the right passage if
the pronoun is first resolved against the conversation. :class:`QueryEngine`
does that by condensing the follow-up plus history into a standalone question
(through the injected :class:`~ragsage.ports.QueryRewriter`) and retrieving on
*that*. These tests drive the behaviour through the public façade with fakes:
the same follow-up is not-found without history and grounded with it.
"""

from __future__ import annotations

from ragsage import IngestionPipeline, QueryEngine, RawSource, Scope
from ragsage.fakes import FakeEngineKit, FakeQueryRewriter
from ragsage.models import Turn

# One document with a clear antecedent ("the Acme plan") whose pricing a
# follow-up refers to only by pronoun, plus an unrelated document that must never
# be pulled in by the rewrite.
ACME = b"The Acme plan costs 50 dollars per month and includes priority support."
BICYCLE = b"A bicycle has two wheels, a frame, and a chain that drives them."


def _engine(kit: FakeEngineKit, *, rewriter: object | None) -> QueryEngine:
    return QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
        rewriter=rewriter,  # type: ignore[arg-type]
        tracer=kit.tracer,
    )


async def test_followup_is_not_found_without_history(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    # The pronoun-only follow-up shares no content word with the corpus, so on
    # its own it retrieves nothing — this is exactly what history must fix.
    await pipeline.ingest(RawSource(name="acme.txt", content=ACME), scope)
    engine = _engine(kit, rewriter=FakeQueryRewriter())

    answer = await engine.query("What about its pricing?", scope)

    assert answer.grounded is False


async def test_followup_resolves_against_history_and_grounds(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    ingested = await pipeline.ingest(RawSource(name="acme.txt", content=ACME), scope)
    await pipeline.ingest(RawSource(name="bicycle.txt", content=BICYCLE), scope)
    engine = _engine(kit, rewriter=FakeQueryRewriter())

    history = [
        Turn(question="Tell me about the Acme plan", answer="The Acme plan is a subscription.")
    ]
    answer = await engine.query("What about its pricing?", scope, history=history)

    # The rewrite carries "Acme plan" forward, so retrieval finds the priced
    # passage and the answer is grounded and cited to the right document.
    assert answer.grounded is True
    assert "50" in answer.text
    assert answer.citations
    assert answer.citations[0].document_id == ingested.document.id


async def test_rewrite_only_runs_when_there_is_history(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    await pipeline.ingest(RawSource(name="acme.txt", content=ACME), scope)
    engine = _engine(kit, rewriter=FakeQueryRewriter())

    # First turn: no history, so no rewrite happens.
    kit.tracer.events.clear()
    await engine.query("Tell me about the Acme plan", scope)
    assert "rewritten" not in kit.tracer.names()

    # Follow-up turn: history present, so the question is condensed first.
    kit.tracer.events.clear()
    await engine.query(
        "What about its pricing?",
        scope,
        history=[Turn(question="Tell me about the Acme plan", answer="It is a plan.")],
    )
    assert "rewritten" in kit.tracer.names()


async def test_history_without_a_rewriter_falls_back_to_the_raw_question(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    # An engine wired with no rewriter must not crash when handed history; it
    # simply retrieves on the raw follow-up (which here can't be resolved).
    await pipeline.ingest(RawSource(name="acme.txt", content=ACME), scope)
    engine = _engine(kit, rewriter=None)

    answer = await engine.query(
        "What about its pricing?",
        scope,
        history=[Turn(question="Tell me about the Acme plan", answer="It is a plan.")],
    )
    assert answer.grounded is False


async def test_query_and_stream_agree_with_history(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    from ragsage.models import AnswerComplete

    await pipeline.ingest(RawSource(name="acme.txt", content=ACME), scope)
    engine = _engine(kit, rewriter=FakeQueryRewriter())
    history = [Turn(question="Tell me about the Acme plan", answer="It is a plan.")]

    buffered = await engine.query("What about its pricing?", scope, history=history)
    events = [ev async for ev in engine.stream("What about its pricing?", scope, history=history)]
    complete = events[-1]
    assert isinstance(complete, AnswerComplete)

    assert buffered.text == complete.text
    assert buffered.grounded == complete.grounded
    assert tuple(buffered.citations) == tuple(complete.citations)
