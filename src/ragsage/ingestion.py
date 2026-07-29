"""The ingestion façade — one document in, retrievable chunks out.

:class:`IngestionPipeline` is the whole write-path of the engine expressed as a
single public method, :meth:`~IngestionPipeline.ingest`. It orchestrates the
ports and owns the *policy* between them — dedup, per-page routing,
contextualisation, and the fan-out to the three stores — while every step that
touches a model, a file, or a database lives behind a port it was handed.

This is the primary write seam: a Seam-1 test wires fakes for each port and
asserts on the returned :class:`IngestionResult` and the resulting store state,
never on anything private here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from ragsage.caching import decode_parsed, encode_parsed, parse_cache_key
from ragsage.config import IngestionConfig
from ragsage.models import (
    Chunk,
    Document,
    EmbeddedChunk,
    Page,
    PageRoute,
    ParsedDocument,
    RawSource,
)
from ragsage.ports import (
    Cache,
    Chunker,
    Contextualizer,
    DocumentParser,
    DocumentStore,
    Embedder,
    LexicalStore,
    LLMClient,
    NullTracer,
    PageClassifier,
    Tracer,
    VectorStore,
)
from ragsage.scope import Scope


@dataclass(frozen=True)
class IngestionResult:
    """What ``ingest`` reports back: identity, size, and how pages were read.

    ``deduplicated`` is ``True`` when the content hash already existed and
    nothing was reprocessed. ``route_counts`` records how many pages took each
    parser route, which is what makes per-page routing observable to a test
    without reaching inside the pipeline.
    """

    document: Document
    chunk_count: int
    route_counts: dict[PageRoute, int] = field(default_factory=dict)
    deduplicated: bool = False


class IngestionPipeline:
    """Ingests documents into a scope's stores through injected adapters."""

    def __init__(
        self,
        *,
        parser: DocumentParser,
        classifier: PageClassifier,
        chunker: Chunker,
        contextualizer: Contextualizer,
        embedder: Embedder,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        document_store: DocumentStore,
        llm: LLMClient,
        cache: Cache,
        tracer: Tracer | None = None,
    ) -> None:
        self._parser = parser
        self._classifier = classifier
        self._chunker = chunker
        self._contextualizer = contextualizer
        self._embedder = embedder
        self._vectors = vector_store
        self._lexical = lexical_store
        self._documents = document_store
        self._llm = llm
        self._cache = cache
        self._tracer: Tracer = tracer or NullTracer()

    async def ingest(
        self,
        source: RawSource,
        scope: Scope,
        config: IngestionConfig | None = None,
    ) -> IngestionResult:
        """Parse, route, chunk, contextualise, embed, and store one document.

        Returns immediately with a dedup result if the same content already
        lives in ``scope`` — a failed or repeated upload never reprocesses.
        """
        config = config or IngestionConfig()
        parsed = await self._parse(source)
        document = parsed.document

        existing = await self._documents.find_by_hash(scope, document.content_hash)
        if existing is not None:
            self._tracer.event("deduplicated", document_id=existing.id)
            return IngestionResult(document=existing, chunk_count=0, deduplicated=True)

        resolved, route_counts = await self._resolve_pages(parsed.pages)
        chunks = list(
            self._chunker.chunk(
                document, resolved, size=config.chunk_size, overlap=config.chunk_overlap
            )
        )
        self._tracer.event("chunked", document_id=document.id, chunks=len(chunks))

        if config.contextualize:
            chunks = await self._contextualize(document, chunks, resolved)

        vectors = await self._embedder.embed([c.embed_text for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("embedder returned a different number of vectors than chunks")
        embedded = [
            EmbeddedChunk(chunk=c, vector=tuple(v)) for c, v in zip(chunks, vectors, strict=True)
        ]

        await self._vectors.upsert(scope, embedded)
        await self._lexical.index(scope, chunks)
        await self._documents.save(scope, document, chunks)
        self._tracer.event("stored", document_id=document.id, chunks=len(chunks))

        return IngestionResult(
            document=document, chunk_count=len(chunks), route_counts=route_counts
        )

    async def _parse(self, source: RawSource) -> ParsedDocument:
        """Parse ``source``, or return the cached parse output for its bytes.

        Parsing is the most expensive deterministic step in the pipeline and its
        result depends only on the document's bytes and the parser reading them,
        so it is cached by content hash — never by path or filename, so the same
        bytes hit after a rename, a re-upload, or a move between stores.

        The cache is an optimisation and is treated as untrustworthy on both
        sides: a miss, a malformed entry, and a cache that raises are all handled
        by simply parsing. That is the :class:`~ragsage.ports.Cache` port's stated
        contract, and it is what makes a dead cache a slowdown rather than an
        outage.
        """
        key = parse_cache_key(hashlib.sha256(source.read()).hexdigest(), self._parser)

        cached = await self._cache_get(key)
        if cached is not None:
            parsed = decode_parsed(cached)
            if parsed is not None:
                self._tracer.event(
                    "parse_cache_hit", document_id=parsed.document.id, pages=len(parsed.pages)
                )
                return parsed

        parsed = self._parser.parse(source)
        self._tracer.event("parsed", document_id=parsed.document.id, pages=len(parsed.pages))
        await self._cache_set(key, encode_parsed(parsed))
        return parsed

    async def _cache_get(self, key: str) -> str | None:
        """Read through the cache, treating any failure as a miss."""
        try:
            return await self._cache.get(key)
        except Exception:
            # A cache that is down must not fail an ingest. Deliberately broad:
            # every adapter failure here has the same correct response.
            self._tracer.event("cache_unavailable", operation="get")
            return None

    async def _cache_set(self, key: str, value: str) -> None:
        """Write through the cache, tolerating failure (e.g. a value too large)."""
        try:
            await self._cache.set(key, value)
        except Exception:
            self._tracer.event("cache_unavailable", operation="set")

    async def _resolve_pages(
        self, pages: Sequence[Page]
    ) -> tuple[list[Page], dict[PageRoute, int]]:
        """Fill in text for every page, sending image pages to the vision model."""
        resolved: list[Page] = []
        counts: dict[PageRoute, int] = {PageRoute.TEXT: 0, PageRoute.VISION: 0}
        for page in pages:
            route = self._classifier.classify(page)
            counts[route] += 1
            if route is PageRoute.VISION and page.image is not None:
                text = await self._llm.transcribe(page.image)
                resolved.append(replace(page, text=text))
            else:
                resolved.append(page)
            self._tracer.event("page_routed", page=page.number, route=route.value)
        return resolved, counts

    async def _contextualize(
        self, document: Document, chunks: list[Chunk], pages: list[Page]
    ) -> list[Chunk]:
        """Prepend a document-situating context to each chunk's embed text.

        Cached by content hash + chunk id so re-ingesting or retrying never pays
        the model cost twice.
        """
        full_text = "\n".join(p.text for p in pages)
        out: list[Chunk] = []
        for chunk in chunks:
            key = f"ctx:{document.content_hash}:{chunk.id}"
            cached = await self._cache_get(key)
            if cached is not None:
                embed_text = cached
            else:
                embed_text = await self._contextualizer.contextualize(
                    document, chunk, full_text=full_text
                )
                await self._cache_set(key, embed_text)
            out.append(replace(chunk, embed_text=embed_text))
        self._tracer.event("contextualized", document_id=document.id, chunks=len(out))
        return out
