"""The two-pass, structure-aware Markdown chunker shared by every format path.

Every real parse path serialises its document to Markdown; this module turns that
Markdown into token-bounded :class:`~ragsage.models.Chunk`s that keep tables whole
and carry their heading context — the same properties Docling's ``HybridChunker``
gave us, done in pure Python with no ML dependency.

Two passes:

1. **Structure.** :class:`~langchain_text_splitters.MarkdownHeaderTextSplitter`
   splits the Markdown at heading boundaries and hands back each section with its
   heading path (``# > ## > ###``). That path is carried onto every chunk of the
   section as ``metadata["headings"]``, so a citation can show which section a
   passage came from.
2. **Size.** Each section is packed into chunks bounded by a token budget measured
   against a pinned tokenizer (``tiktoken`` — never ``transformers``). Prose that
   overflows the budget is split recursively; a table is kept whole, and only if a
   single table exceeds the budget is it split by rows with its header repeated so
   every piece stays a readable table.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from ragsage.models import Chunk

TokenCounter = Callable[[str], int]
"""Counts the tokens in a string — the length function the size pass bounds against."""

_DEFAULT_ENCODING = "cl100k_base"

# Heading levels the structural pass splits on, deepest last so the metadata dict
# keeps its H1 > H2 > … order. All six ATX levels split, so a format that emits
# deep headings (HTML/DOCX carry the full <h1>…<h6> hierarchy) has every level's
# context reach chunk metadata rather than being swallowed into its parent section.
_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]
_HEADER_KEYS = [key for _, key in _HEADERS_TO_SPLIT_ON]

# A Markdown table delimiter row is nothing but pipes, dashes, colons and spaces,
# and contains at least one of each of a pipe and a dash (e.g. ``| --- | :--: |``).
_TABLE_DELIM_CHARS = frozenset("|-: ")


def make_token_counter(encoding_name: str = _DEFAULT_ENCODING) -> TokenCounter:
    """A token counter backed by ``tiktoken``, or an approximation if unavailable.

    Loads the pinned ``tiktoken`` encoding (pure Rust/Python — no ``transformers``,
    no torch). Should the encoding be unreachable (e.g. an offline box with a cold
    cache), it degrades to a character-based estimate rather than failing ingest;
    the size pass then bounds chunks by that estimate. Callers that need exact,
    reproducible counts should keep the encoding cached.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:  # pragma: no cover - only when tiktoken can't load offline
        return _approx_token_count
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def _approx_token_count(text: str) -> int:
    """A rough token estimate (~4 chars/token) for when no tokenizer is available."""
    return max(1, len(text) // 4)


def chunk_markdown(
    markdown: str,
    *,
    document_id: str,
    page: int,
    size: int,
    overlap: int,
    count_tokens: TokenCounter,
    start_ordinal: int = 0,
) -> list[Chunk]:
    """Split ``markdown`` into token-bounded chunks, section by section.

    ``size`` is the per-chunk token budget and ``overlap`` the token overlap used
    only when a single prose block overflows and must be recursively split. Each
    returned chunk records ``page`` as its source page and, when the section sat
    under one or more headings, ``metadata["headings"]`` as that heading path.

    ``start_ordinal`` is the ordinal (and id suffix) the first emitted chunk takes,
    so a multi-page parser can chunk page by page yet keep one dense, ordered
    ordinal sequence across the whole document — page 2's chunks continue where
    page 1's left off rather than colliding on ``…:0``.
    """
    chunks: list[Chunk] = []
    ordinal = start_ordinal
    for headings, body in _sections(markdown):
        units = _size_bounded_units(body, size=size, overlap=overlap, count_tokens=count_tokens)
        metadata = {"headings": headings} if headings else {}
        for text in _pack(units, size=size, count_tokens=count_tokens):
            chunks.append(
                Chunk(
                    id=f"{document_id}:{ordinal}",
                    document_id=document_id,
                    text=text,
                    page=page,
                    ordinal=ordinal,
                    metadata=metadata,
                )
            )
            ordinal += 1
    return chunks


def _sections(markdown: str) -> list[tuple[list[str], str]]:
    """Pass 1: split on headings into ``(heading_path, section_body)`` pairs.

    Markdown before any heading yields a section with an empty heading path.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(_HEADERS_TO_SPLIT_ON, strip_headers=True)
    sections: list[tuple[list[str], str]] = []
    for doc in splitter.split_text(markdown):
        headings = [doc.metadata[key] for key in _HEADER_KEYS if key in doc.metadata]
        body = doc.page_content.strip()
        if body:
            sections.append((headings, body))
    if not sections and markdown.strip():
        # No headings at all (plain text): the whole document is one section.
        sections.append(([], markdown.strip()))
    return sections


def _size_bounded_units(
    body: str, *, size: int, overlap: int, count_tokens: TokenCounter
) -> list[str]:
    """Break a section body into ordered units none larger than ``size`` tokens.

    Prose and tables are separated so a table is never cut mid-row: a table within
    budget stays one unit, an oversized table is split by rows (header repeated),
    and oversized prose is split recursively on natural boundaries.
    """
    units: list[str] = []
    for kind, text in _blocks(body):
        if count_tokens(text) <= size:
            units.append(text)
        elif kind == "table":
            units.extend(_split_table_rows(text, size=size, count_tokens=count_tokens))
        else:
            units.extend(_split_prose(text, size=size, overlap=overlap, count_tokens=count_tokens))
    return units


def _blocks(body: str) -> list[tuple[str, str]]:
    """Split a section body into ordered ``("table" | "prose", text)`` blocks."""
    lines = body.split("\n")
    blocks: list[tuple[str, str]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        text = "\n".join(prose).strip()
        if text:
            blocks.append(("prose", text))
        prose.clear()

    i = 0
    n = len(lines)
    while i < n:
        if _is_table_row(lines[i]) and i + 1 < n and _is_table_delimiter(lines[i + 1]):
            flush_prose()
            start = i
            i += 2  # header row + delimiter row
            while i < n and _is_table_row(lines[i]):
                i += 1
            blocks.append(("table", "\n".join(lines[start:i]).strip()))
        else:
            prose.append(lines[i])
            i += 1
    flush_prose()
    return blocks


def _is_table_row(line: str) -> bool:
    return "|" in line.strip()


def _is_table_delimiter(line: str) -> bool:
    stripped = line.strip()
    if "-" not in stripped or "|" not in stripped:
        return False
    return set(stripped) <= _TABLE_DELIM_CHARS


def _split_table_rows(table: str, *, size: int, count_tokens: TokenCounter) -> list[str]:
    """Split an oversized table by rows, repeating the header row in each piece."""
    rows = [line for line in table.split("\n") if line.strip()]
    if len(rows) < 3:
        # Header + delimiter with no data rows (or a degenerate table): nothing to
        # split — keep it whole even if it nominally overflows.
        return [table]
    header, delimiter, data = rows[0], rows[1], rows[2:]

    pieces: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append("\n".join([header, delimiter, *current]))

    for row in data:
        candidate = "\n".join([header, delimiter, *current, row])
        if current and count_tokens(candidate) > size:
            flush()
            current = [row]
        else:
            current.append(row)
    flush()
    return pieces or [table]


def _split_prose(text: str, *, size: int, overlap: int, count_tokens: TokenCounter) -> list[str]:
    """Recursively split oversized prose on natural boundaries, bounded by tokens."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=min(overlap, max(size - 1, 0)),
        length_function=count_tokens,
    )
    parts = [part.strip() for part in splitter.split_text(text)]
    return [part for part in parts if part]


def _pack(units: Sequence[str], *, size: int, count_tokens: TokenCounter) -> list[str]:
    """Greedily pack ordered units into chunks up to ``size`` tokens, order preserved.

    Each unit is already within budget, so a unit is never itself split here; this
    only merges small neighbouring units (a heading's stub paragraph and its
    table, say) so the corpus isn't shredded into one-line chunks. The budget is
    checked against the *joined* text — separators included — so a packed chunk
    never overflows; only a lone unit that is itself irreducible (a single wide
    table row) can exceed ``size``.
    """
    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and count_tokens("\n\n".join([*current, unit])) > size:
            chunks.append("\n\n".join(current))
            current = []
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


_NEWLINES = re.compile(r"\r\n?")


def normalize_newlines(text: str) -> str:
    """Collapse CRLF/CR to LF so heading and table detection see uniform lines.

    Applied by every format path before chunking (the text path decodes raw bytes
    that may carry Windows line endings), so the line-based heading and table
    scanners never miss a boundary that ends in ``\\r\\n``.
    """
    return _NEWLINES.sub("\n", text)
