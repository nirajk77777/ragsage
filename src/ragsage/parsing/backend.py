"""The heuristic parse/chunk backend — ragsage's first real parser.

:class:`HeuristicBackend` is a pure-Python, model-free document backend that
satisfies *both* the :class:`~ragsage.ports.DocumentParser` and
:class:`~ragsage.ports.Chunker` ports by shape — the same dual-port,
in-flight-stash arrangement the Docling backend uses. ``parse`` reads the source,
routes it to a per-format path that turns it into one or more
:class:`~ragsage.models.Page`\\s of Markdown, and stashes those pages keyed by the
content-derived document id; ``chunk`` pops the stash (enforcing parse-before-chunk,
once) and runs the shared two-pass chunker over the *resolved* pages the pipeline
hands back. The two calls happen within one ``IngestionPipeline.ingest`` for a
single document, so the hand-off is one in-flight entry, consumed once.

Each format has its own parse path (see :mod:`ragsage.parsing.router` for how a
source is routed to one). This module owns only the **plain-text / Markdown** path
inline — the bytes are read as-is and emitted as a single
:class:`~ragsage.models.Page` with no measured layout, so the page classifier's
text-presence fallback routes it to ``TEXT``. Every structured format lives in its
own sibling module (``pdf``, ``docx``, ``pptx``, ``html``) and is imported *lazily*,
only when a source actually routes to it, so importing this module — which the
composition root does — never drags in a parser library (and never trips the
CPU-compatibility dependency guard).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

from ragsage.models import Chunk, Document, Page, ParsedDocument, RawSource
from ragsage.parsing.chunking import (
    TokenCounter,
    chunk_markdown,
    make_token_counter,
    normalize_newlines,
)
from ragsage.parsing.identity import document_for, read_bytes
from ragsage.parsing.router import DocumentFormat, route

# Flow formats (plain text, Markdown, and the office/web formats with no page
# grid) have no page grid, so the whole document collapses onto page 1 —
# mirroring how the Docling adapter cites a page-less document.
_FLOW_FORMAT_PAGE = 1

# A per-format parse path: raw source + its bytes -> the ordered pages it yields.
# Paged formats (PDF) return one entry per page with a measured layout and,
# where a page must be read by vision, a rendered image; flow formats return a
# single layout-free page.
FormatParser = Callable[[RawSource, bytes], list[Page]]

# The structured formats each live in a sibling module, imported lazily only when
# a source routes to them (keeping the import graph of this module free of any
# parser library). ``TEXT`` is handled inline; ``UNKNOWN`` never reaches here.
_FORMAT_MODULES: dict[DocumentFormat, tuple[str, str]] = {
    DocumentFormat.PDF: ("ragsage.parsing.pdf", "parse_pdf"),
    DocumentFormat.DOCX: ("ragsage.parsing.docx", "parse_docx"),
    DocumentFormat.PPTX: ("ragsage.parsing.pptx", "parse_pptx"),
    DocumentFormat.HTML: ("ragsage.parsing.html", "parse_html"),
}


class HeuristicBackend:
    """Both the ``DocumentParser`` and the ``Chunker`` port, no ML dependency.

    ``encoding_name`` names the ``tiktoken`` encoding the chunker bounds token
    counts against; it is loaded lazily on first chunk so importing this module
    (which the composition root does) costs nothing.
    """

    def __init__(self, *, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._count_tokens: TokenCounter | None = None
        # rag document id -> the parsed pages awaiting chunking (the in-flight
        # stash; consumed once, by ``chunk``, within the same ingest call).
        self._parsed: dict[str, list[Page]] = {}

    # -- DocumentParser -----------------------------------------------------

    def parse(self, source: RawSource) -> ParsedDocument:
        """Read a source into per-page Markdown and stash it for the chunker."""
        content = read_bytes(source)
        document = document_for(source, content)
        pages = self._parse_pages(source, content)
        self._parsed[document.id] = pages
        return ParsedDocument(document=document, pages=pages)

    def _parse_pages(self, source: RawSource, content: bytes) -> list[Page]:
        """Route a source to its format path and return the pages it yields.

        The text/Markdown path is inline; every structured format is delegated to
        its sibling module, imported lazily. A source that routes to a format
        whose module isn't installed/implemented fails loudly rather than
        ingesting garbage; an unrecognised container is rejected outright.
        """
        fmt = route(source, content)
        if fmt is DocumentFormat.TEXT:
            markdown = normalize_newlines(content.decode("utf-8", "replace"))
            return [Page(number=_FLOW_FORMAT_PAGE, text=markdown, layout=None)]
        if fmt is DocumentFormat.UNKNOWN:
            raise ValueError(f"unrecognised document format for source {source.name!r}")
        return _format_parser(fmt)(source, content)

    def _take_parsed(self, document_id: str) -> list[Page]:
        """Pop the stashed pages for one document — consumed once, by ``chunk``."""
        try:
            return self._parsed.pop(document_id)
        except KeyError as exc:  # pragma: no cover - defensive; parse precedes chunk
            raise RuntimeError(f"no parsed document stashed for {document_id}") from exc

    # -- Chunker ------------------------------------------------------------

    def chunk(
        self, document: Document, pages: Sequence[Page], *, size: int, overlap: int
    ) -> Sequence[Chunk]:
        """Structure-aware chunks over the resolved pages, page by page.

        ``size`` is the per-chunk token budget and ``overlap`` the token overlap
        for oversized prose; the two-pass chunker keeps tables whole and carries
        each section's heading path onto its chunks. ``pages`` is the *resolved*
        page list the pipeline built — a page that routed to vision arrives here
        with its transcription filled into ``text`` — so chunking reads from it
        (not the stash) and a scanned PDF page ends up as retrievable as a typed
        one. The stash is popped purely to enforce parse-before-chunk, once.
        Ordinals run densely across every page so chunk ids stay unique.
        """
        self._take_parsed(document.id)
        count_tokens = self._token_counter()
        chunks: list[Chunk] = []
        for page in pages:
            page_chunks = chunk_markdown(
                normalize_newlines(page.text),
                document_id=document.id,
                page=page.number,
                size=size,
                overlap=overlap,
                count_tokens=count_tokens,
                start_ordinal=len(chunks),
            )
            chunks.extend(page_chunks)
        return chunks

    def _token_counter(self) -> TokenCounter:
        if self._count_tokens is None:
            self._count_tokens = make_token_counter(self._encoding_name)
        return self._count_tokens


def _format_parser(fmt: DocumentFormat) -> FormatParser:
    """Import a structured format's parse function, lazily and by name.

    Kept out of :meth:`HeuristicBackend._parse_pages` so the import happens only
    when a source of that format is actually parsed. A missing module (the format
    isn't implemented yet, or its library isn't installed) surfaces as a
    ``NotImplementedError`` naming the format, not a bare ``ImportError``.
    """
    module_name, func_name = _FORMAT_MODULES[fmt]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only before a format lands
        raise NotImplementedError(
            f"the {fmt.value!r} parse path is not available ({module_name} missing)"
        ) from exc
    parser: FormatParser = getattr(module, func_name)
    return parser
