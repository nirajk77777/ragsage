"""The ports — every seam where a real provider plugs into the engine.

These are the *entire* contract between ``ragsage`` and the outside world.
Everything model-shaped or storage-shaped lives behind one of them, so the
library depends on no provider SDK and no database driver. The backend (or the
CLI, or a test) supplies an adapter for each; the pipelines in
:mod:`ragsage.ingestion` and :mod:`ragsage.query` orchestrate them and nothing
else.

They are :class:`typing.Protocol`s on purpose: an adapter conforms by shape, so
it need not import or subclass anything from ``ragsage``. That keeps the
dependency arrow pointing strictly inward — adapters know about the engine, the
engine never knows about them.

I/O-bound ports (models, stores, cache) are ``async``; CPU-bound local ports
(parser, classifier, chunker, tracer) are synchronous, mirroring how their real
implementations behave — Docling parses on a thread, an embedder awaits the
gateway.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Protocol, runtime_checkable

from ragsage.models import (
    Chunk,
    Document,
    EmbeddedChunk,
    Page,
    PageImage,
    PageRoute,
    ParsedDocument,
    RawSource,
    ScoredChunk,
    Turn,
    Vector,
)
from ragsage.scope import Scope

# --------------------------------------------------------------------------- #
# Ingestion-side ports
# --------------------------------------------------------------------------- #


@runtime_checkable
class DocumentParser(Protocol):
    """Turns raw bytes into a structured document with per-page content.

    Extracts the born-digital text layer where one exists and surfaces a
    :class:`PageImage` where a page must be read by vision instead. It does not
    itself run OCR — routing that decision is the classifier's and the
    pipeline's job, keeping parsers (Docling, others) swappable.
    """

    def parse(self, source: RawSource) -> ParsedDocument: ...


@runtime_checkable
class PageClassifier(Protocol):
    """Decides, per page, whether to trust its text layer or send it to vision."""

    def classify(self, page: Page) -> PageRoute: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document's resolved page text into structure-aware chunks.

    Receives the page text *after* vision transcription has filled in image
    pages, so it never has to care which route a page took.
    """

    def chunk(
        self, document: Document, pages: Sequence[Page], *, size: int, overlap: int
    ) -> Sequence[Chunk]: ...


@runtime_checkable
class Contextualizer(Protocol):
    """Writes a short context sentence situating a chunk in its parent document.

    Returns the text to embed (context + chunk). Implemented with a cheap,
    prompt-cached model in production; the engine stores this separately from
    the verbatim chunk so display stays clean.
    """

    async def contextualize(self, document: Document, chunk: Chunk, *, full_text: str) -> str: ...


# --------------------------------------------------------------------------- #
# Model ports (talk to an OpenAI-compatible gateway in production)
# --------------------------------------------------------------------------- #


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to dense vectors. Batched, because real embedders are."""

    async def embed(self, texts: Sequence[str]) -> Sequence[Vector]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidates against the query with a cross-encoder, keeping top-k."""

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int
    ) -> Sequence[ScoredChunk]: ...


@runtime_checkable
class LLMClient(Protocol):
    """The generation (and vision) model, behind an OpenAI-compatible shape.

    ``generate`` streams answer tokens so the backend can relay them over SSE;
    the engine assembles them into an :class:`~ragsage.models.Answer`.
    ``transcribe`` is the vision call the two-path parser routes image pages to
    — modelled here because, like every other model call, it goes through the
    same gateway rather than a bespoke OCR dependency.
    """

    def generate(self, prompt: str) -> AsyncIterator[str]: ...

    async def transcribe(self, image: PageImage) -> str: ...


@runtime_checkable
class QueryRewriter(Protocol):
    """Condenses a follow-up plus conversation history into a standalone question.

    Multi-turn chat asks things like "what about its pricing?" whose meaning
    lives in earlier turns. Given the latest ``question`` and the ``history``
    before it, this returns a self-contained question that names its own
    referents, so retrieval matches on resolved terms rather than a bare pronoun.
    It is its own port (not folded into :class:`LLMClient`) because production
    condensation runs on a cheap, fast model — the same reason
    :class:`Contextualizer` is separate from generation — and because a caller
    with no conversation history can skip it entirely.
    """

    async def rewrite(self, question: str, history: Sequence[Turn]) -> str: ...


# --------------------------------------------------------------------------- #
# Storage & infrastructure ports
# --------------------------------------------------------------------------- #


@runtime_checkable
class VectorStore(Protocol):
    """Dense-vector persistence and nearest-neighbour search, scoped.

    Every method takes a :class:`Scope`; the store keys on ``scope.namespace``
    and honours ``scope.filters`` as part of the query, never post-hoc. In
    production this is one pgvector table under Row-Level Security.
    """

    async def upsert(self, scope: Scope, records: Sequence[EmbeddedChunk]) -> None: ...

    async def search(
        self, scope: Scope, query_vector: Vector, *, k: int
    ) -> Sequence[ScoredChunk]: ...

    async def delete(self, scope: Scope, document_id: str) -> None: ...


@runtime_checkable
class LexicalStore(Protocol):
    """Keyword/BM25 persistence and search, scoped like the vector store.

    Its results are fused with the vector store's to form hybrid retrieval.
    """

    async def index(self, scope: Scope, chunks: Sequence[Chunk]) -> None: ...

    async def search(self, scope: Scope, query: str, *, k: int) -> Sequence[ScoredChunk]: ...

    async def delete(self, scope: Scope, document_id: str) -> None: ...


@runtime_checkable
class DocumentStore(Protocol):
    """The document-level record: dedup, listing, and purge-on-delete.

    ``find_by_hash`` powers content-hash dedup so re-uploading a file is a
    no-op; ``delete`` here removes the document row while the vector and lexical
    stores purge their own copies — together that is a true delete, not a hide.
    """

    async def save(self, scope: Scope, document: Document, chunks: Sequence[Chunk]) -> None: ...

    async def find_by_hash(self, scope: Scope, content_hash: str) -> Document | None: ...

    async def get(self, scope: Scope, document_id: str) -> Document | None: ...

    async def list(self, scope: Scope) -> Sequence[Document]: ...

    async def delete(self, scope: Scope, document_id: str) -> None: ...


@runtime_checkable
class Cache(Protocol):
    """A best-effort key/value cache — e.g. to skip recomputing contextualisation.

    A miss returns ``None``; the engine treats the cache as an optimisation and
    is always correct without it.
    """

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str) -> None: ...


@runtime_checkable
class Span(Protocol):
    """One node in an end-to-end trace — a single pipeline stage, timed.

    Opened by :meth:`Tracer.span` (the request root) or :meth:`Span.child` (a
    nested stage) and used as a context manager, so the stage's duration is the
    ``with`` block. :meth:`set` attaches attributes discovered mid-stage (how
    many candidates survived rerank, which chunk ids reached the prompt, whether
    the answer was grounded) — exactly the breadcrumbs that let a bad answer be
    tracked to the stage that produced it. A child inherits its parent's tenant
    tag, so every node in a trace is tenant-scoped and no trace can straddle two.

    Like :class:`Tracer`, no method may raise: observability must never break a
    query, even when the backing sink (Langfuse, a log) is down.
    """

    def set(self, **attributes: object) -> None: ...

    def child(self, name: str, **attributes: object) -> Span: ...

    def __enter__(self) -> Span: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """A structured sink for observability across pipeline stages.

    Two shapes, one sink. :meth:`event` records a flat point event (page routed,
    query rewritten) — the lightweight breadcrumb the CLI and tests read back.
    :meth:`span` opens the root of an end-to-end trace, tagged with the caller's
    tenant, under which the query engine nests one child :class:`Span` per stage
    (retrieval → rerank → prompt → generation); an adapter fans the whole tree to
    Langfuse. Neither may raise — observability is not allowed to break a query.
    """

    def event(self, name: str, **fields: object) -> None: ...

    def span(self, name: str, **attributes: object) -> Span: ...


class NullSpan:
    """A :class:`Span` that records nothing — the no-op the null tracer hands out."""

    def set(self, **attributes: object) -> None:
        return None

    def child(self, name: str, **attributes: object) -> NullSpan:
        return self

    def __enter__(self) -> NullSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class NullTracer:
    """A :class:`Tracer` that discards every event and span.

    The pipelines default to this when no tracer is injected, so tracing is
    always optional and never a required dependency of running the engine.
    """

    def event(self, name: str, **fields: object) -> None:
        return None

    def span(self, name: str, **attributes: object) -> NullSpan:
        return NullSpan()
