"""Arm (a): collapse the typed items into Markdown at the boundary.

Two variants, because "collapse to Markdown" turns out to be two different
decisions that the ticket's phrasing runs together:

* **a1 — page collapse (the naive reading).** Join every item on a page into one
  Markdown blob, build a :class:`~ragsage.models.Page`, and hand it to the
  existing chunker exactly as a parser would. This is what "collapse into
  Markdown at the boundary" means if you take it literally, and it is the arm
  that genuinely throws the type away.

* **a2 — item-attributed collapse.** Still collapse to Markdown (the domain model
  is untouched), but chunk *item by item* and stamp what the item was onto
  ``Chunk.metadata``, which is already a free-form ``Mapping[str, object]``
  (``models.py:171``) and is already persisted as ``metadata jsonb``
  (ADR-0002 §1). The Markdown is the payload; the type rides alongside it.

Both variants reuse the *real* production chunker (``ragsage.parsing.chunking``)
and the real pipeline collaborators, so what they lose is what the shipped code
would lose. Nothing here is production code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from items import ContentItem

from ragsage.config import IngestionConfig
from ragsage.models import Chunk, Document, EmbeddedChunk, Page
from ragsage.parsing.chunking import chunk_markdown, make_token_counter, normalize_newlines

# RAG-Anything's page_idx is 0-based; ragsage's Page.number / Chunk.page are
# 1-based (models.py:106, :168). The seam owns this conversion.
PAGE_BASE = 1


@dataclass
class ItemIngestionResult:
    """What the prototype seam reports back, plus the chunks for inspection."""

    document: Document
    chunks: list[Chunk]
    pages: list[Page]


class ItemIngestionPrototype:
    """A stand-in for ``IngestionPipeline.ingest_items``.

    It takes the same collaborators ``IngestionPipeline`` does and replays the
    *post-parse tail* of ``ingest`` (chunk -> contextualise -> embed -> store).
    That duplication is itself a finding: shipping ``ingest_items`` means
    extracting that tail out of ``IngestionPipeline.ingest`` (ingestion.py:112-137)
    into a shared private method — the same ~25 lines, for both arms.
    """

    def __init__(
        self,
        *,
        contextualizer: object,
        embedder: object,
        vector_store: object,
        lexical_store: object,
        document_store: object,
        cache: object,
    ) -> None:
        self._contextualizer = contextualizer
        self._embedder = embedder
        self._vectors = vector_store
        self._lexical = lexical_store
        self._documents = document_store
        self._cache = cache
        self._count_tokens = make_token_counter()

    # -- a1: page collapse --------------------------------------------------

    def to_pages(self, items: Sequence[ContentItem]) -> list[Page]:
        """Join every item on a page into one Markdown page. Type is gone here."""
        by_page: dict[int, list[str]] = {}
        for item in items:
            rendered = item.to_markdown()
            if rendered:
                by_page.setdefault(item.page_idx, []).append(rendered)
        return [
            Page(number=idx + PAGE_BASE, text=normalize_newlines("\n\n".join(blocks)), layout=None)
            for idx, blocks in sorted(by_page.items())
        ]

    def chunk_collapsed(
        self, document: Document, pages: Sequence[Page], *, size: int, overlap: int
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(
                chunk_markdown(
                    page.text,
                    document_id=document.id,
                    page=page.number,
                    size=size,
                    overlap=overlap,
                    count_tokens=self._count_tokens,
                    start_ordinal=len(chunks),
                )
            )
        return chunks

    # -- a2: item-attributed collapse ---------------------------------------

    def chunk_per_item(
        self, document: Document, items: Sequence[ContentItem], *, size: int, overlap: int
    ) -> list[Chunk]:
        """Chunk each item separately and stamp its type onto chunk metadata.

        Note what this cannot use: the ``Chunker`` *port* (ports.py:75) takes
        ``(document, pages, size, overlap)`` and has no ``start_ordinal``, so
        calling it once per item yields colliding ids (``{doc}:0`` from every
        call). The free function ``chunk_markdown`` does take ``start_ordinal``,
        so the prototype calls it directly — but that means a2 through the
        *declared port* is not possible without either widening the port or
        re-id-ing chunks after the fact.
        """
        chunks: list[Chunk] = []
        headings: list[str] = []
        for item in items:
            rendered = item.to_markdown()
            if not rendered:
                continue
            _track_headings(headings, rendered)
            metadata: dict[str, object] = {"block_type": item.type}
            # Chunking per item makes the structural pass item-scoped, so a table
            # item never sees the heading a preceding text item carried. The seam
            # restores the full document-scoped path from a running stack — a fix
            # that lives at the boundary, needing no model change.
            if headings:
                metadata["headings"] = list(headings)
            if item.caption():
                metadata["caption"] = item.caption()
            if item.footnote():
                metadata["footnote"] = item.footnote()
            produced = chunk_markdown(
                normalize_newlines(rendered),
                document_id=document.id,
                page=item.page_idx + PAGE_BASE,
                size=size,
                overlap=overlap,
                count_tokens=self._count_tokens,
                start_ordinal=len(chunks),
            )
            for chunk in produced:
                merged: dict[str, object] = dict(chunk.metadata)
                merged.update(metadata)
                chunks.append(replace(chunk, metadata=merged))
        return chunks

    # -- the shared post-chunk tail (copied from IngestionPipeline.ingest) ----

    async def store(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        pages: Sequence[Page],
        scope: object,
        config: IngestionConfig,
    ) -> list[Chunk]:
        final = list(chunks)
        if config.contextualize:
            full_text = "\n".join(p.text for p in pages)
            out: list[Chunk] = []
            for chunk in final:
                key = f"ctx:{document.content_hash}:{chunk.id}"
                cached = await self._cache.get(key)  # type: ignore[attr-defined]
                if cached is not None:
                    embed_text = cached
                else:
                    embed_text = await self._contextualizer.contextualize(  # type: ignore[attr-defined]
                        document, chunk, full_text=full_text
                    )
                    await self._cache.set(key, embed_text)  # type: ignore[attr-defined]
                out.append(replace(chunk, embed_text=embed_text))
            final = out

        vectors = await self._embedder.embed([c.embed_text for c in final])  # type: ignore[attr-defined]
        embedded = [
            EmbeddedChunk(chunk=c, vector=tuple(v)) for c, v in zip(final, vectors, strict=True)
        ]
        await self._vectors.upsert(scope, embedded)  # type: ignore[attr-defined]
        await self._lexical.index(scope, final)  # type: ignore[attr-defined]
        await self._documents.save(scope, document, final)  # type: ignore[attr-defined]
        return final


def _track_headings(stack: list[str], markdown: str) -> None:
    """Advance a document-scoped ATX heading stack over one item's Markdown."""
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        hashes = len(stripped) - len(stripped.lstrip("#"))
        title = stripped[hashes:].strip()
        if not title or hashes > 6:
            continue
        del stack[hashes - 1 :]
        stack.append(title)


def metadata_keys(chunks: Sequence[Chunk]) -> set[str]:
    keys: set[str] = set()
    for chunk in chunks:
        keys.update(chunk.metadata)
    return keys


def looks_like_a_table(text: str) -> bool:
    """The only way a downstream consumer can guess 'this is a table' under a1."""
    lines = [line for line in text.splitlines() if line.strip()]
    return any(
        "|" in lines[i] and "|" in lines[i + 1] and set(lines[i + 1].strip()) <= set("|-: ")
        for i in range(len(lines) - 1)
    )


def chunk_summary(chunk: Chunk) -> Mapping[str, object]:
    return {
        "id": chunk.id,
        "page": chunk.page,
        "metadata": dict(chunk.metadata),
        "table_shaped_text": looks_like_a_table(chunk.text),
        "chars": len(chunk.text),
    }
