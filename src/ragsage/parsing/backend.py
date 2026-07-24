"""The heuristic parse/chunk backend — ragsage's first real parser.

:class:`HeuristicBackend` is a pure-Python, model-free document backend that
satisfies *both* the :class:`~ragsage.ports.DocumentParser` and
:class:`~ragsage.ports.Chunker` ports by shape — the same dual-port,
in-flight-stash arrangement the Docling backend uses. ``parse`` reads the source,
routes it to a per-format path that serialises it to Markdown, and stashes that
Markdown keyed by the content-derived document id; ``chunk`` pops the stash and
runs the shared two-pass chunker over it. The two calls happen within one
``IngestionPipeline.ingest`` for a single document, so the hand-off is one
in-flight entry, consumed once.

This ticket implements the **plain-text / Markdown** path: the bytes are read
as-is and emitted as a single :class:`~ragsage.models.Page` with no measured
layout, so the page classifier's text-presence fallback routes it to ``TEXT``.
The remaining formats route (see :mod:`ragsage.parsing.router`) but are not yet
parsed; later issues fill each one in behind the same stash.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragsage.models import Chunk, Document, Page, ParsedDocument, RawSource
from ragsage.parsing.chunking import (
    TokenCounter,
    chunk_markdown,
    make_token_counter,
    normalize_newlines,
)
from ragsage.parsing.identity import document_for, read_bytes
from ragsage.parsing.router import DocumentFormat, route

# Flow formats (plain text, Markdown) have no page grid, so every chunk of one
# document cites the same single source page — mirroring how the Docling adapter
# collapses a page-less document onto page 1.
_FLOW_FORMAT_PAGE = 1


class HeuristicBackend:
    """Both the ``DocumentParser`` and the ``Chunker`` port, no ML dependency.

    ``encoding_name`` names the ``tiktoken`` encoding the chunker bounds token
    counts against; it is loaded lazily on first chunk so importing this module
    (which the composition root does) costs nothing.
    """

    def __init__(self, *, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._count_tokens: TokenCounter | None = None
        # rag document id -> the parsed Markdown awaiting chunking.
        self._parsed: dict[str, str] = {}

    # -- DocumentParser -----------------------------------------------------

    def parse(self, source: RawSource) -> ParsedDocument:
        """Read a source into a Markdown document and stash it for the chunker."""
        content = read_bytes(source)
        document = document_for(source, content)
        markdown = self._to_markdown(source, content)
        self._parsed[document.id] = markdown
        page = Page(number=_FLOW_FORMAT_PAGE, text=markdown, layout=None)
        return ParsedDocument(document=document, pages=[page])

    def _to_markdown(self, source: RawSource, content: bytes) -> str:
        """Serialise a source to Markdown by its routed format.

        Only the text/Markdown path is implemented here: the bytes are decoded and
        newline-normalised as-is (Markdown is already Markdown, plain text is valid
        Markdown). The structured formats route but raise until their issue lands,
        so a mis-routed binary fails loudly rather than ingesting garbage.
        """
        fmt = route(source, content)
        if fmt is DocumentFormat.TEXT:
            return normalize_newlines(content.decode("utf-8", "replace"))
        raise NotImplementedError(
            f"the {fmt.value!r} parse path is not implemented yet (source {source.name!r})"
        )

    def _take_parsed(self, document_id: str) -> str:
        """Pop the stashed Markdown for one document — consumed once, by ``chunk``."""
        try:
            return self._parsed.pop(document_id)
        except KeyError as exc:  # pragma: no cover - defensive; parse precedes chunk
            raise RuntimeError(f"no parsed document stashed for {document_id}") from exc

    # -- Chunker ------------------------------------------------------------

    def chunk(
        self, document: Document, pages: Sequence[Page], *, size: int, overlap: int
    ) -> Sequence[Chunk]:
        """Structure-aware chunks over the stashed Markdown.

        ``size`` is the per-chunk token budget and ``overlap`` the token overlap
        for oversized prose; the two-pass chunker keeps tables whole and carries
        each section's heading path onto its chunks. ``pages`` is unused for the
        text path (its resolved text equals the stashed Markdown) but is part of
        the port so vision-filled pages flow through unchanged for later formats.
        """
        markdown = self._take_parsed(document.id)
        return chunk_markdown(
            markdown,
            document_id=document.id,
            page=_FLOW_FORMAT_PAGE,
            size=size,
            overlap=overlap,
            count_tokens=self._token_counter(),
        )

    def _token_counter(self) -> TokenCounter:
        if self._count_tokens is None:
            self._count_tokens = make_token_counter(self._encoding_name)
        return self._count_tokens
