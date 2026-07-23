"""Seam-1 tests for the query façade and its retrieval building blocks.

The load-bearing behaviours here — grounded citation binding and the honest
not-found path — are asserted through ``QueryEngine.query`` with fakes. The pure
fusion/arrangement helpers are unit-tested directly.
"""

from __future__ import annotations

from ragsage.fakes import FakeEngineKit
from ragsage.models import AnswerComplete, AnswerToken, Chunk, Citation, ScoredChunk, Usage
from ragsage.query import NOT_FOUND_MESSAGE, _arrange_for_edges, reciprocal_rank_fusion

from ragsage import IngestionPipeline, QueryEngine, QueryOptions, RawSource, Scope

FRANCE = b"Paris is the capital of France. It sits on the Seine river."
BIOLOGY = b"The mitochondrion is the powerhouse of the cell and produces ATP."


def _chunk(cid: str, text: str = "x") -> Chunk:
    return Chunk(id=cid, document_id="d", text=text, page=1, ordinal=0)


# --------------------------------------------------------------------------- #
# Pure retrieval helpers
# --------------------------------------------------------------------------- #


def test_rrf_reinforces_chunks_ranked_by_both_channels() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    dense = [ScoredChunk(a, 0.9), ScoredChunk(b, 0.8)]
    lexical = [ScoredChunk(b, 5.0), ScoredChunk(c, 4.0)]

    fused = reciprocal_rank_fusion(dense, lexical)

    # b is the only chunk in both lists, so it must fuse to the top.
    assert fused[0].chunk.id == "b"
    assert {s.chunk.id for s in fused} == {"a", "b", "c"}  # union, deduped by id


def test_rrf_of_nothing_is_empty() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_edge_arrangement_puts_best_first_and_second_best_last() -> None:
    ranked = [ScoredChunk(_chunk(str(i)), 1.0) for i in range(5)]

    arranged = _arrange_for_edges(ranked)

    assert arranged[0].chunk.id == "0"  # best at the front edge
    assert arranged[-1].chunk.id == "1"  # second-best at the back edge


# --------------------------------------------------------------------------- #
# The façade
# --------------------------------------------------------------------------- #


async def test_query_returns_grounded_answer_with_bound_citation(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    ingested = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)

    answer = await engine.query("What is the capital of France?", scope)

    assert answer.grounded is True
    assert "Paris" in answer.text
    assert answer.citations, "a grounded answer must cite its source"
    top = answer.citations[0]
    assert top.marker == 1
    assert "Paris" in top.quote  # the citation resolves to the passage used
    assert top.document_id == ingested.document.id


async def test_query_without_matching_corpus_says_not_found(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), scope)

    answer = await engine.query("Who won the 1998 football world cup?", scope)

    assert answer.grounded is False
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == ()


async def test_query_on_empty_corpus_says_not_found(engine: QueryEngine, scope: Scope) -> None:
    answer = await engine.query("anything at all", scope)

    assert answer.grounded is False
    assert answer.text == NOT_FOUND_MESSAGE


async def test_query_is_isolated_by_namespace(
    pipeline: IngestionPipeline, engine: QueryEngine
) -> None:
    a = Scope(namespace="tenant-a")
    b = Scope(namespace="tenant-b")
    await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), a)

    # The same question in a different namespace sees nothing.
    leaked = await engine.query("What is the capital of France?", b)
    assert leaked.grounded is False


async def test_document_filter_narrows_retrieval(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    france = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)
    await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), scope)

    # Restrict the query to the biology document; the France answer is out of scope.
    narrowed = scope.with_filters(document_ids=[france.document.id])
    answer = await engine.query("What produces ATP in the cell?", narrowed)

    assert answer.grounded is False  # ATP lives only in the excluded document


async def test_context_budget_is_respected(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    # Three form-feed pages all mentioning "reactor"; context_k caps how many
    # chunks reach the model, so at most that many citations can come back.
    body = b"reactor safety alpha\freactor safety beta\freactor safety gamma"
    await pipeline.ingest(RawSource(name="r.txt", content=body), scope)

    answer = await engine.query("tell me about reactor safety", scope, QueryOptions(context_k=2))

    assert answer.grounded is True
    assert len(answer.citations) <= 2


# --------------------------------------------------------------------------- #
# Streaming — the SSE-shaped event order the backend relays
# --------------------------------------------------------------------------- #


async def test_stream_emits_tokens_then_citations_then_usage_then_complete(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    ingested = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)

    events = [ev async for ev in engine.stream("What is the capital of France?", scope)]

    # The stream is well-ordered: every token precedes every citation, one usage
    # follows the citations, and a single complete event terminates it.
    kinds = [type(ev).__name__ for ev in events]
    assert kinds.count("AnswerComplete") == 1
    assert isinstance(events[-1], AnswerComplete)
    assert isinstance(events[-2], Usage)
    last_token = max(i for i, ev in enumerate(events) if isinstance(ev, AnswerToken))
    first_citation = min(i for i, ev in enumerate(events) if isinstance(ev, Citation))
    assert last_token < first_citation

    # The streamed tokens reassemble into the completed answer text…
    streamed = "".join(ev.text for ev in events if isinstance(ev, AnswerToken)).strip()
    complete = events[-1]
    assert isinstance(complete, AnswerComplete)
    assert complete.text == streamed
    assert complete.grounded is True
    assert "Paris" in complete.text

    # …and the streamed citations equal the completed answer's, bound to the source.
    citations = [ev for ev in events if isinstance(ev, Citation)]
    assert citations
    assert tuple(citations) == tuple(complete.citations)
    assert citations[0].document_id == ingested.document.id

    usage = events[-2]
    assert isinstance(usage, Usage)
    assert usage.sources >= 1


async def test_stream_not_found_streams_message_and_completes_ungrounded(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), scope)

    events = [ev async for ev in engine.stream("Who won the 1998 football world cup?", scope)]

    # Even an ungrounded reply arrives as streamed tokens, so the UI needs no
    # special case for "not found".
    streamed = "".join(ev.text for ev in events if isinstance(ev, AnswerToken)).strip()
    assert streamed == NOT_FOUND_MESSAGE
    assert not any(isinstance(ev, Citation) for ev in events)

    complete = events[-1]
    assert isinstance(complete, AnswerComplete)
    assert complete.grounded is False
    assert complete.text == NOT_FOUND_MESSAGE
    assert complete.citations == ()


async def test_model_declining_on_retrieved_context_is_ungrounded(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    # A nearest-neighbour store almost always returns *something*, so retrieval
    # rarely comes back empty. If the model then decides those chunks don't
    # actually answer the question and replies with the not-found message, that
    # must surface as an honest, citation-free not-found — not a grounded answer.
    await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)

    class DecliningLLM:
        async def generate(self, prompt: str):
            for word in NOT_FOUND_MESSAGE.split():
                yield word + " "

        async def transcribe(self, image):  # pragma: no cover - unused here
            return ""

    declining = QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=DecliningLLM(),
    )

    answer = await declining.query("What is the capital of France?", scope)

    assert answer.grounded is False
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == ()


async def test_query_and_stream_agree(
    pipeline: IngestionPipeline, engine: QueryEngine, scope: Scope
) -> None:
    # query() is a buffer over stream(); the two must never diverge.
    await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)

    buffered = await engine.query("What is the capital of France?", scope)
    events = [ev async for ev in engine.stream("What is the capital of France?", scope)]
    complete = events[-1]
    assert isinstance(complete, AnswerComplete)

    assert buffered.text == complete.text
    assert buffered.grounded == complete.grounded
    assert tuple(buffered.citations) == tuple(complete.citations)
