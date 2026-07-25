"""In-memory, offline adapters for every port — the standalone proof.

These are *fakes*, not mocks: each one is a genuine, working implementation that
happens to run in memory with no network. Together they let the pipelines run
the full ingest-and-query loop deterministically — which is exactly what the
Seam-1 tests and the CLI need. They are shipped in the package (not hidden in
tests) so `pip install ragsage` gives you a runnable engine out of the box.

The retrieval is real enough to be meaningful: the embedder hashes tokens into a
sparse vector so cosine similarity tracks term overlap, and unrelated text
scores zero — which is what lets the honest "not found" path actually fire. The
stores serialise to plain dicts so the CLI can persist a corpus between the
``ingest`` and ``query`` commands.

Nothing here is tuned for quality; swap in real adapters (voyage embeddings,
Cohere rerank, pgvector) behind the same ports for that.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

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
from ragsage.parsing.identity import document_for, read_bytes
from ragsage.query import NOT_FOUND_MESSAGE
from ragsage.scope import Scope

_EMBED_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")

# A tiny stop-word list. Lexical matching, reranking, and the extractive reader
# all score on *content* words, so shared filler ("the", "of", "is") never on
# its own makes a chunk look relevant or earns a spurious citation.
_STOPWORDS = frozenset(
    "a an and are as at be by do for from how in is it its of on or that the "
    "this to was were what when where which who will with".split()
)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _content_tokens(text: str) -> set[str]:
    """Tokens with stop-words removed — the signal used for relevance matching."""
    return {t for t in _tokens(text) if t not in _STOPWORDS}


def _bucket(token: str, dim: int) -> int:
    """Map a token to a vector index with a process-stable hash (not ``hash()``)."""
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _embed_one(text: str, dim: int = _EMBED_DIM) -> Vector:
    """Hash tokens into a sparse, L2-normalised term vector of width ``dim``."""
    vec = [0.0] * dim
    for tok in _tokens(text):
        vec[_bucket(tok, dim)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return tuple(vec)
    return tuple(v / norm for v in vec)


def _cosine(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _matches_filters(document_id: str, scope: Scope) -> bool:
    """Honour a ``document_ids`` filter in the scope, if present."""
    allowed = scope.filters.get("document_ids")
    if allowed is None:
        return True
    if isinstance(allowed, (set, frozenset, list, tuple)):
        return document_id in allowed
    return False


# --------------------------------------------------------------------------- #
# Chunk (de)serialisation shared by the stores
# --------------------------------------------------------------------------- #


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "page": chunk.page,
        "ordinal": chunk.ordinal,
        "embed_text": chunk.embed_text,
        "metadata": dict(chunk.metadata),
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        id=data["id"],
        document_id=data["document_id"],
        text=data["text"],
        page=data["page"],
        ordinal=data["ordinal"],
        embed_text=data.get("embed_text", ""),
        metadata=data.get("metadata", {}),
    )


# --------------------------------------------------------------------------- #
# Ingestion-side fakes
# --------------------------------------------------------------------------- #


class FakeDocumentParser:
    """Reads a source's bytes into pages, splitting on form-feeds.

    A text source becomes one page per form-feed-delimited block. A source whose
    media type or name says "image" becomes a single vision page carrying the
    bytes for :meth:`FakeLLMClient.transcribe` to read — so both parser routes
    are exercisable from real inputs.
    """

    _IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp")

    def parse(self, source: RawSource) -> ParsedDocument:
        content = read_bytes(source)
        # Content-derived identity through the one shared helper, so the fake keys
        # documents byte-for-byte the same way the real parser paths do.
        document = document_for(source, content)

        if self._is_image(source):
            pages = [Page(number=1, image=PageImage(ref=source.name, data=content))]
        else:
            blocks = content.decode("utf-8", "replace").split("\f")
            pages = [Page(number=i, text=block) for i, block in enumerate(blocks, start=1)]
        return ParsedDocument(document=document, pages=pages)

    def _is_image(self, source: RawSource) -> bool:
        if source.media_type and source.media_type.startswith("image/"):
            return True
        return source.name.lower().endswith(self._IMAGE_SUFFIXES)


class FakePageClassifier:
    """Routes a page by whether it has a usable text layer.

    When the parser measured a :class:`~ragsage.models.PageLayout`, this reads the
    same signals the real classifier does — a page with little text that a raster
    image dominates goes to vision; otherwise the presence of a text layer
    decides. With no layout it falls back to whether ``text`` is non-blank, which
    is all the offline fakes and the CLI need.
    """

    def classify(self, page: Page) -> PageRoute:
        if page.layout is not None:
            has_text = page.layout.text_chars >= 16 and page.layout.image_area_ratio < 0.5
            if not has_text and page.image is not None:
                return PageRoute.VISION
            return PageRoute.TEXT
        if page.text.strip():
            return PageRoute.TEXT
        if page.image is not None:
            return PageRoute.VISION
        return PageRoute.TEXT


class FakeChunker:
    """Word-window chunker: ``size`` words per chunk with ``overlap`` carried over.

    Tokens stand in for the real tokeniser's tokens; the windowing and stable
    per-document chunk ids are what the pipeline actually depends on.
    """

    def chunk(
        self, document: Document, pages: Sequence[Page], *, size: int, overlap: int
    ) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0
        step = max(1, size - overlap)
        for page in pages:
            words = page.text.split()
            if not words:
                continue
            for start in range(0, len(words), step):
                window = words[start : start + size]
                if not window:
                    break
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{ordinal}",
                        document_id=document.id,
                        text=" ".join(window),
                        page=page.number,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1
                if start + size >= len(words):
                    break
        return chunks


class FakeContextualizer:
    """Prepends a document-situating tag to the chunk's embed text."""

    async def contextualize(self, document: Document, chunk: Chunk, *, full_text: str) -> str:
        return f"[Context: {document.source}] {chunk.text}"


class FakeEmbedder:
    """Deterministic hashing embedder — cosine tracks term overlap.

    ``dim`` sizes the output vector; it defaults to the library's own width but is
    configurable so a caller whose store fixes a different dimension (e.g. a real
    pgvector column) can produce matching vectors without forking this algorithm.
    """

    def __init__(self, *, dim: int = _EMBED_DIM) -> None:
        self._dim = dim

    async def embed(self, texts: Sequence[str]) -> Sequence[Vector]:
        return [_embed_one(t, self._dim) for t in texts]


class FakeReranker:
    """Cross-encoder stand-in: rescore by query-term overlap, drop the misses.

    Uses a different signal (raw token overlap) than the dense embedder, so it
    genuinely reorders fused candidates and prunes ones with no lexical tie to
    the query — the pruning is what backstops the not-found path.
    """

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int
    ) -> Sequence[ScoredChunk]:
        q = _content_tokens(query)
        rescored: list[ScoredChunk] = []
        for cand in candidates:
            overlap = q & _content_tokens(cand.chunk.text)
            if not overlap:
                continue
            score = len(overlap) / len(q) if q else 0.0
            rescored.append(ScoredChunk(chunk=cand.chunk, score=score))
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored[:top_k]


class FakeLLMClient:
    """An extractive reader: answers verbatim from the most relevant source.

    ``generate`` parses the numbered sources out of the grounded prompt, picks
    the ones sharing terms with the question, and streams back the best source's
    text followed by ``[n]`` markers for every relevant source — giving a
    grounded, citable answer with zero model calls. ``transcribe`` decodes the
    image bytes as the "vision" reading of a scanned page.

    It also answers the small-talk prompt, which carries a ``Message:`` line
    instead of sources: a fixed friendly sentence, so the conversational path is
    exercisable offline exactly like the grounded one.
    """

    _SOURCE = re.compile(r"^\[(\d+)\]\s+(.*)$")

    SMALL_TALK_REPLY = (
        "Hello! Ask me anything about your uploaded documents and I'll answer with citations."
    )
    """What the fake replies to a conversational message. Fixed, so tests can
    assert on it without pinning the wording of a real model."""

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        if "Sources:" not in prompt and "\nMessage: " in prompt:
            for word in self.SMALL_TALK_REPLY.split():
                yield word + " "
            return

        sources, question = self._parse_prompt(prompt)
        q = _content_tokens(question)
        relevant = [(m, text) for m, text in sources if q & _content_tokens(text)]
        if not relevant:
            for tok in NOT_FOUND_MESSAGE.split():
                yield tok + " "
            return
        _, best_text = max(relevant, key=lambda mt: len(q & _content_tokens(mt[1])))
        markers = "".join(f"[{m}]" for m, _ in sorted(relevant))
        for word in best_text.split():
            yield word + " "
        yield markers

    async def transcribe(self, image: PageImage) -> str:
        if image.data is not None:
            return image.data.decode("utf-8", "replace")
        return f"[vision transcription of {image.ref}]"

    def _parse_prompt(self, prompt: str) -> tuple[list[tuple[int, str]], str]:
        # Mirrors the layout produced by QueryEngine._build_prompt: a "Sources:"
        # block of "[n] text" lines followed by a "Question:" line. If that
        # prompt format changes, this reader must change with it.
        sources: list[tuple[int, str]] = []
        question = ""
        in_sources = False
        for line in prompt.splitlines():
            if line == "Sources:":
                in_sources = True
                continue
            if line.startswith("Question:"):
                question = line[len("Question:") :].strip()
                in_sources = False
                continue
            if in_sources:
                match = self._SOURCE.match(line)
                if match:
                    sources.append((int(match.group(1)), match.group(2)))
        return sources, question


class FakeQueryRewriter:
    """Condenses a follow-up by folding in the content words of prior turns.

    A real rewriter would ask a model to resolve "its" to "the Acme plan"; this
    deterministic stand-in appends the content words said earlier (that the
    follow-up doesn't already carry) to the question. That is enough for the
    overlap-based retrieval and reranker fakes to match the passage the pronoun
    pointed at — so the multi-turn path is exercisable offline, exactly like the
    rest of the engine.
    """

    async def rewrite(self, question: str, history: Sequence[Turn]) -> str:
        if not history:
            return question
        carried: list[str] = []
        seen = _content_tokens(question)
        for turn in history:
            for token in _tokens(turn.question) + _tokens(turn.answer):
                if token in _STOPWORDS or token in seen:
                    continue
                seen.add(token)
                carried.append(token)
        if not carried:
            return question
        return f"{question} {' '.join(carried)}"


# --------------------------------------------------------------------------- #
# Storage & infrastructure fakes
# --------------------------------------------------------------------------- #


class FakeVectorStore:
    """In-memory dense store; cosine search, scoped and filterable."""

    def __init__(self) -> None:
        self._by_ns: dict[str, list[EmbeddedChunk]] = defaultdict(list)

    async def upsert(self, scope: Scope, records: Sequence[EmbeddedChunk]) -> None:
        ns = self._by_ns[scope.namespace]
        incoming = {r.chunk.id for r in records}
        ns[:] = [r for r in ns if r.chunk.id not in incoming]
        ns.extend(records)

    async def search(self, scope: Scope, query_vector: Vector, *, k: int) -> Sequence[ScoredChunk]:
        scored: list[ScoredChunk] = []
        for record in self._by_ns.get(scope.namespace, []):
            if not _matches_filters(record.chunk.document_id, scope):
                continue
            sim = _cosine(query_vector, record.vector)
            if sim > 0.0:
                scored.append(ScoredChunk(chunk=record.chunk, score=sim))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, scope: Scope, document_id: str) -> None:
        ns = self._by_ns.get(scope.namespace)
        if ns is not None:
            ns[:] = [r for r in ns if r.chunk.document_id != document_id]

    def snapshot(self) -> dict[str, Any]:
        return {
            ns: [{"chunk": _chunk_to_dict(r.chunk), "vector": list(r.vector)} for r in records]
            for ns, records in self._by_ns.items()
        }

    def restore(self, data: dict[str, Any]) -> None:
        self._by_ns = defaultdict(list)
        for ns, records in data.items():
            self._by_ns[ns] = [
                EmbeddedChunk(chunk=_chunk_from_dict(r["chunk"]), vector=tuple(r["vector"]))
                for r in records
            ]


class FakeLexicalStore:
    """In-memory keyword store; overlap search, scoped and filterable."""

    def __init__(self) -> None:
        self._by_ns: dict[str, list[Chunk]] = defaultdict(list)

    async def index(self, scope: Scope, chunks: Sequence[Chunk]) -> None:
        ns = self._by_ns[scope.namespace]
        incoming = {c.id for c in chunks}
        ns[:] = [c for c in ns if c.id not in incoming]
        ns.extend(chunks)

    async def search(self, scope: Scope, query: str, *, k: int) -> Sequence[ScoredChunk]:
        q = _content_tokens(query)
        if not q:
            return []
        scored: list[ScoredChunk] = []
        for chunk in self._by_ns.get(scope.namespace, []):
            if not _matches_filters(chunk.document_id, scope):
                continue
            overlap = q & _content_tokens(chunk.text)
            if overlap:
                scored.append(ScoredChunk(chunk=chunk, score=len(overlap)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def delete(self, scope: Scope, document_id: str) -> None:
        ns = self._by_ns.get(scope.namespace)
        if ns is not None:
            ns[:] = [c for c in ns if c.document_id != document_id]

    def snapshot(self) -> dict[str, Any]:
        return {ns: [_chunk_to_dict(c) for c in chunks] for ns, chunks in self._by_ns.items()}

    def restore(self, data: dict[str, Any]) -> None:
        self._by_ns = defaultdict(list)
        for ns, chunks in data.items():
            self._by_ns[ns] = [_chunk_from_dict(c) for c in chunks]


class FakeDocumentStore:
    """In-memory document registry: dedup, listing, and purge."""

    def __init__(self) -> None:
        self._by_ns: dict[str, dict[str, Document]] = defaultdict(dict)

    async def save(self, scope: Scope, document: Document, chunks: Sequence[Chunk]) -> None:
        self._by_ns[scope.namespace][document.id] = document

    async def find_by_hash(self, scope: Scope, content_hash: str) -> Document | None:
        for doc in self._by_ns.get(scope.namespace, {}).values():
            if doc.content_hash == content_hash:
                return doc
        return None

    async def get(self, scope: Scope, document_id: str) -> Document | None:
        return self._by_ns.get(scope.namespace, {}).get(document_id)

    async def list(self, scope: Scope) -> Sequence[Document]:
        return list(self._by_ns.get(scope.namespace, {}).values())

    async def delete(self, scope: Scope, document_id: str) -> None:
        self._by_ns.get(scope.namespace, {}).pop(document_id, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            ns: [
                {
                    "id": d.id,
                    "source": d.source,
                    "content_hash": d.content_hash,
                    "metadata": dict(d.metadata),
                }
                for d in docs.values()
            ]
            for ns, docs in self._by_ns.items()
        }

    def restore(self, data: dict[str, Any]) -> None:
        self._by_ns = defaultdict(dict)
        for ns, docs in data.items():
            self._by_ns[ns] = {
                d["id"]: Document(
                    id=d["id"],
                    source=d["source"],
                    content_hash=d["content_hash"],
                    metadata=d.get("metadata", {}),
                )
                for d in docs
            }


class FakeCache:
    """A plain in-memory dict cache."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value


@dataclass
class RecordedSpan:
    """One captured trace node — a stage's name, tenant tag, and attributes.

    Mutable on purpose: :meth:`FakeSpan.set` updates ``attributes`` and
    :meth:`FakeSpan.child` appends to ``children`` as a stage runs, so after a
    query the tree mirrors exactly what the engine emitted.
    """

    name: str
    tenant: str | None
    attributes: dict[str, object] = field(default_factory=dict)
    children: list[RecordedSpan] = field(default_factory=list)
    error: str | None = None


class FakeSpan:
    """A :class:`~ragsage.ports.Span` that writes into a :class:`RecordedSpan`.

    Children inherit the parent's tenant tag, so the captured tree is proof that
    every node of a trace is tenant-scoped — the guarantee the observability tests
    assert against.
    """

    def __init__(self, record: RecordedSpan) -> None:
        self._record = record

    def set(self, **attributes: object) -> None:
        self._record.attributes.update(attributes)

    def child(self, name: str, **attributes: object) -> FakeSpan:
        record = RecordedSpan(name=name, tenant=self._record.tenant, attributes=dict(attributes))
        self._record.children.append(record)
        return FakeSpan(record)

    def __enter__(self) -> FakeSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        if exc is not None:
            self._record.error = repr(exc)
        return None


class FakeTracer:
    """Records flat events *and* the span tree so tests can assert what ran.

    ``events``/``names`` are the point-event breadcrumbs (unchanged); ``spans``
    holds each request's root :class:`RecordedSpan` with its nested stages, and
    :meth:`all_spans`/:meth:`find_span` walk that tree.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.spans: list[RecordedSpan] = []

    def event(self, name: str, **fields: object) -> None:
        self.events.append((name, fields))

    def span(self, name: str, **attributes: object) -> FakeSpan:
        tenant = attributes.get("tenant")
        record = RecordedSpan(
            name=name,
            tenant=tenant if isinstance(tenant, str) else None,
            attributes=dict(attributes),
        )
        self.spans.append(record)
        return FakeSpan(record)

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def all_spans(self) -> list[RecordedSpan]:
        """Every recorded span, roots and descendants, in pre-order."""
        out: list[RecordedSpan] = []

        def walk(span: RecordedSpan) -> None:
            out.append(span)
            for child in span.children:
                walk(child)

        for root in self.spans:
            walk(root)
        return out

    def find_span(self, name: str) -> RecordedSpan | None:
        """The first recorded span with ``name``, searched depth-first, or ``None``."""
        for span in self.all_spans():
            if span.name == name:
                return span
        return None


# --------------------------------------------------------------------------- #
# Convenience bundle
# --------------------------------------------------------------------------- #


class FakeEngineKit:
    """Every fake, pre-instantiated and shared, ready to wire a pipeline.

    A single object holding one instance of each adapter so a caller (a test or
    the CLI) can construct an :class:`~ragsage.ingestion.IngestionPipeline` and a
    :class:`~ragsage.query.QueryEngine` that read and write the *same* in-memory
    stores. :meth:`snapshot`/:meth:`restore` persist the whole corpus as one
    JSON-able dict.
    """

    def __init__(self) -> None:
        self.parser = FakeDocumentParser()
        self.classifier = FakePageClassifier()
        self.chunker = FakeChunker()
        self.contextualizer = FakeContextualizer()
        self.embedder = FakeEmbedder()
        self.reranker = FakeReranker()
        self.llm = FakeLLMClient()
        self.vector_store = FakeVectorStore()
        self.lexical_store = FakeLexicalStore()
        self.document_store = FakeDocumentStore()
        self.cache = FakeCache()
        self.tracer = FakeTracer()

    def snapshot(self) -> dict[str, Any]:
        return {
            "vector": self.vector_store.snapshot(),
            "lexical": self.lexical_store.snapshot(),
            "documents": self.document_store.snapshot(),
        }

    def restore(self, data: dict[str, Any]) -> None:
        self.vector_store.restore(data.get("vector", {}))
        self.lexical_store.restore(data.get("lexical", {}))
        self.document_store.restore(data.get("documents", {}))


def all_fakes() -> Iterable[object]:
    """Every fake adapter type — used to assert one exists per port."""
    return (
        FakeDocumentParser,
        FakePageClassifier,
        FakeChunker,
        FakeContextualizer,
        FakeEmbedder,
        FakeReranker,
        FakeLLMClient,
        FakeQueryRewriter,
        FakeVectorStore,
        FakeLexicalStore,
        FakeDocumentStore,
        FakeCache,
        FakeTracer,
    )
