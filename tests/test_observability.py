"""Seam-1 tests for end-to-end query tracing (issue 10).

Every query opens one tenant-tagged root span with a child per stage — rewrite,
retrieve, rerank, prompt, generate — each carrying what it produced. These tests
prove that shape, that every node is tenant-scoped (so no trace straddles two
workspaces), and — the load-bearing one — that a deliberately wrong answer can be
tracked to the exact stage that failed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ragsage import IngestionPipeline, QueryEngine, RawSource, Scope
from ragsage.fakes import FakeEngineKit, FakeQueryRewriter, RecordedSpan
from ragsage.models import PageImage, Turn
from ragsage.query import NOT_FOUND_MESSAGE

FRANCE = b"Paris is the capital of France. It sits on the Seine river."
BIOLOGY = b"The mitochondrion is the powerhouse of the cell and produces ATP."


def _engine(kit: FakeEngineKit, **overrides: object) -> QueryEngine:
    kwargs: dict[str, object] = {
        "embedder": kit.embedder,
        "vector_store": kit.vector_store,
        "lexical_store": kit.lexical_store,
        "reranker": kit.reranker,
        "llm": kit.llm,
        "tracer": kit.tracer,
    }
    kwargs.update(overrides)
    return QueryEngine(**kwargs)


class _DecliningLLM:
    """A model that refuses even when the right context reached it."""

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        for word in NOT_FOUND_MESSAGE.split():
            yield word + " "

    async def transcribe(self, image: PageImage) -> str:  # pragma: no cover - unused
        return ""


async def test_query_emits_a_tenant_tagged_span_per_stage(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)
    engine = _engine(kit, rewriter=FakeQueryRewriter())

    # A follow-up (history present) exercises the rewrite stage too, so all five
    # stages of the pipeline show up under the one root.
    history = [Turn(question="What is the capital of France?", answer="Paris.")]
    _ = [ev async for ev in engine.stream("And what river is it on?", scope, history=history)]

    assert len(kit.tracer.spans) == 1
    root = kit.tracer.spans[0]
    assert root.name == "query"
    assert root.tenant == scope.namespace
    assert [child.name for child in root.children] == [
        "rewrite",
        "retrieve",
        "rerank",
        "prompt",
        "generate",
    ]
    # Every node of the trace — root and each stage — carries the tenant tag.
    assert all(span.tenant == scope.namespace for span in kit.tracer.all_spans())
    # And the stages recorded what they produced, not just that they ran.
    assert kit.tracer.find_span("retrieve").attributes["fused"] >= 1
    assert kit.tracer.find_span("generate").attributes["grounded"] is True


async def test_traces_are_tenant_scoped_with_no_cross_tenant_leakage(
    pipeline: IngestionPipeline, kit: FakeEngineKit
) -> None:
    a = Scope(namespace="tenant-a")
    b = Scope(namespace="tenant-b")
    france = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), a)
    await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), b)
    engine = _engine(kit)

    await engine.query("What is the capital of France?", a)
    await engine.query("What produces ATP in the cell?", b)

    # One root trace per request, each tagged only with its own tenant.
    assert [root.tenant for root in kit.tracer.spans] == ["tenant-a", "tenant-b"]

    def subtree(root: RecordedSpan) -> list[RecordedSpan]:
        out = [root]
        for child in root.children:
            out.extend(subtree(child))
        return out

    trace_a, trace_b = kit.tracer.spans
    assert all(span.tenant == "tenant-a" for span in subtree(trace_a))
    assert all(span.tenant == "tenant-b" for span in subtree(trace_b))

    # Tenant B's trace never references tenant A's document — no id leaks across.
    france_doc = france.document.id
    b_ids = [
        cid
        for span in subtree(trace_b)
        for key in ("fused_ids", "kept_ids", "context_ids")
        for cid in span.attributes.get(key, [])
    ]
    assert all(not str(cid).startswith(france_doc) for cid in b_ids)


async def test_wrong_answer_traces_to_a_generation_failure(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    france = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)
    engine = _engine(kit, llm=_DecliningLLM())

    answer = await engine.query("What is the capital of France?", scope)

    # The answer is wrong: the corpus *does* hold the answer, yet none came back.
    assert answer.grounded is False

    france_chunk = f"{france.document.id}:0"
    retrieve = kit.tracer.find_span("retrieve")
    prompt = kit.tracer.find_span("prompt")
    generate = kit.tracer.find_span("generate")
    assert retrieve is not None and prompt is not None and generate is not None

    # Retrieval and prompt did their job — the right chunk reached the model…
    assert france_chunk in retrieve.attributes["fused_ids"]
    assert france_chunk in prompt.attributes["context_ids"]
    # …so the trace pins the failure on generation: context was present, unused.
    assert generate.attributes["grounded"] is False
    assert kit.tracer.find_span("query").attributes["outcome"] == "not_found"


async def test_wrong_answer_traces_to_a_retrieval_failure(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    france = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)
    bio = await pipeline.ingest(RawSource(name="bio.txt", content=BIOLOGY), scope)
    engine = _engine(kit)

    # Restrict retrieval to the biology document; the France answer is now out of
    # reach, so the answer is a wrong (unhelpful) not-found.
    narrowed = scope.with_filters(document_ids=[bio.document.id])
    answer = await engine.query("What is the capital of France?", narrowed)
    assert answer.grounded is False

    france_chunk = f"{france.document.id}:0"
    retrieve = kit.tracer.find_span("retrieve")
    rerank = kit.tracer.find_span("rerank")
    assert retrieve is not None and rerank is not None

    # The trace shows the failure is upstream, in the retrieval funnel: the
    # relevant chunk was never a candidate, and nothing survived rerank…
    assert france_chunk not in retrieve.attributes["fused_ids"]
    assert rerank.attributes["kept"] == 0
    # …so generation never ran, and the root records the not-found outcome.
    assert kit.tracer.find_span("generate") is None
    assert kit.tracer.find_span("query").attributes["outcome"] == "not_found"
