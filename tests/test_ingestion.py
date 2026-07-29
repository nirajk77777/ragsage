"""Seam-1 tests for the ingestion façade, driven through fakes.

Every assertion is on the observable result of ``ingest`` or on what became
retrievable afterwards — never on the pipeline's internals.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

from ragsage import Chunk, IngestionConfig, IngestionPipeline, PageRoute, RawSource, Scope
from ragsage.fakes import FakeEngineKit, FakeLLMClient
from ragsage.models import (
    Document,
    Page,
    PageImage,
    PageLayout,
    ParsedDocument,
)

FRANCE = b"Paris is the capital of France. It sits on the Seine river.\fLyon is a French city."


async def test_ingest_stores_document_and_chunks(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    result = await pipeline.ingest(RawSource(name="france.txt", content=FRANCE), scope)

    assert result.deduplicated is False
    assert result.chunk_count == 2  # one page per form-feed block
    stored = await kit.document_store.list(scope)
    assert [d.source for d in stored] == ["france.txt"]
    assert result.document.content_hash  # identity is content-derived


async def test_reingesting_same_content_is_deduplicated(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    source = RawSource(name="france.txt", content=FRANCE)
    first = await pipeline.ingest(source, scope)
    second = await pipeline.ingest(source, scope)

    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.chunk_count == 0
    assert len(await kit.document_store.list(scope)) == 1  # not stored twice


async def test_scanned_page_routes_through_vision(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    # A ".png" source with no text layer must be read by the vision model and
    # still end up as retrievable text.
    scan = RawSource(name="memo.png", content=b"handwritten llama revenue memo")
    result = await pipeline.ingest(scan, scope)

    assert result.route_counts[PageRoute.VISION] == 1
    assert result.route_counts[PageRoute.TEXT] == 0
    assert result.chunk_count == 1
    assert "page_routed" in kit.tracer.names()


async def test_born_digital_page_routes_as_text(pipeline: IngestionPipeline, scope: Scope) -> None:
    result = await pipeline.ingest(RawSource(name="a.txt", content=b"plain typed text"), scope)

    assert result.route_counts[PageRoute.TEXT] == 1
    assert result.route_counts[PageRoute.VISION] == 0


class _MixedParser:
    """Parses a fixed report: two typed pages and two scanned pages.

    Stands in for a real parser to exercise per-page routing of a *mixed*
    document — the case where only the image pages should incur a vision call.
    Each page carries a measured :class:`PageLayout` so the classifier routes on
    the same signals a real one would.
    """

    def parse(self, source: RawSource) -> ParsedDocument:
        digest = hashlib.sha256(source.name.encode()).hexdigest()
        document = Document(id=digest[:16], source=source.name, content_hash=digest)
        pages = [
            Page(
                number=1,
                text="the quarterly revenue summary for the typed body of the report",
                layout=PageLayout(text_chars=200, image_area_ratio=0.0),
            ),
            Page(
                number=2,
                text="appendix figures and typed notes that continue the report body",
                layout=PageLayout(text_chars=180, image_area_ratio=0.05),
            ),
            Page(
                number=3,
                image=PageImage(ref="scan#3", data=b"handwritten llama census tally"),
                layout=PageLayout(text_chars=0, image_area_ratio=0.98, bitmap_area=400_000.0),
            ),
            Page(
                number=4,
                image=PageImage(ref="scan#4", data=b"photographed alpaca ledger totals"),
                layout=PageLayout(text_chars=0, image_area_ratio=0.98, bitmap_area=400_000.0),
            ),
        ]
        return ParsedDocument(document=document, pages=pages)


class _CountingVisionClient:
    """Wraps a fake LLM, tallying how many pages actually hit the vision model."""

    def __init__(self, inner: FakeLLMClient) -> None:
        self._inner = inner
        self.transcribe_calls = 0

    def generate(self, prompt: str) -> AsyncIterator[str]:
        return self._inner.generate(prompt)

    async def transcribe(self, image: PageImage) -> str:
        self.transcribe_calls += 1
        return await self._inner.transcribe(image)


def _mixed_pipeline(kit: FakeEngineKit, llm: _CountingVisionClient) -> IngestionPipeline:
    return IngestionPipeline(
        parser=_MixedParser(),
        classifier=kit.classifier,
        chunker=kit.chunker,
        contextualizer=kit.contextualizer,
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        llm=llm,
        cache=kit.cache,
        tracer=kit.tracer,
    )


async def test_mixed_document_only_image_pages_incur_a_vision_call(
    kit: FakeEngineKit, scope: Scope
) -> None:
    llm = _CountingVisionClient(kit.llm)
    result = await _mixed_pipeline(kit, llm).ingest(
        RawSource(name="report.pdf", content=b"x"), scope
    )

    # Two typed pages stay on the text path; the two scanned pages route to vision
    # and each incurs exactly one transcription — the typed pages incur none.
    assert result.route_counts == {PageRoute.TEXT: 2, PageRoute.VISION: 2}
    assert llm.transcribe_calls == 2


async def test_scanned_pages_of_mixed_document_are_retrievable(
    kit: FakeEngineKit, scope: Scope
) -> None:
    llm = _CountingVisionClient(kit.llm)
    await _mixed_pipeline(kit, llm).ingest(RawSource(name="report.pdf", content=b"x"), scope)

    # The vision-transcribed content of an image-only page is searchable, so a
    # scanned page ends up as queryable as a typed one.
    scanned = await _first_stored_chunk(kit, scope, "handwritten llama census")
    assert "llama" in scanned.text
    typed = await _first_stored_chunk(kit, scope, "quarterly revenue summary")
    assert "revenue" in typed.text


async def _first_stored_chunk(kit: FakeEngineKit, scope: Scope, probe: str) -> Chunk:
    """Fetch a stored chunk back through the public VectorStore search seam."""
    query_vector = (await kit.embedder.embed([probe]))[0]
    results = await kit.vector_store.search(scope, query_vector, k=1)
    assert results, "expected an embedded chunk"
    return results[0].chunk


async def test_contextualization_enriches_embed_text_and_is_cached(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    result = await pipeline.ingest(
        RawSource(name="a.txt", content=b"the mitochondrion makes ATP"), scope
    )

    chunk = await _first_stored_chunk(kit, scope, "mitochondrion")
    assert chunk.embed_text.startswith("[Context:")  # contextualizer ran
    assert chunk.text == "the mitochondrion makes ATP"  # display text stays clean
    # The contextualiser's output was cached under the content hash (public Cache seam).
    cache_key = f"ctx:{result.document.content_hash}:{chunk.id}"
    assert await kit.cache.get(cache_key) is not None


async def test_contextualization_can_be_disabled(
    pipeline: IngestionPipeline, kit: FakeEngineKit, scope: Scope
) -> None:
    config = IngestionConfig(contextualize=False)
    result = await pipeline.ingest(
        RawSource(name="a.txt", content=b"plain body text"), scope, config
    )

    chunk = await _first_stored_chunk(kit, scope, "plain body text")
    assert chunk.embed_text == chunk.text  # no context prepended
    # Nothing was contextualised, so no cache entry exists for the chunk.
    assert await kit.cache.get(f"ctx:{result.document.content_hash}:{chunk.id}") is None


async def test_ingestion_is_scoped_by_namespace(
    pipeline: IngestionPipeline, kit: FakeEngineKit
) -> None:
    a = Scope(namespace="tenant-a")
    b = Scope(namespace="tenant-b")
    await pipeline.ingest(RawSource(name="a.txt", content=b"alpha content here"), a)

    assert len(await kit.document_store.list(a)) == 1
    assert len(await kit.document_store.list(b)) == 0  # never leaks across namespaces
