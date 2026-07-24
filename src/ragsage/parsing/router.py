"""Format routing — pick a per-format parse path without any ML model.

The heuristic backend handles several source formats, each with its own parse
path. This router decides which, using only signals that are cheap and
dependency-light: the caller-supplied ``media_type`` first (most authoritative),
then the filename extension, then — as a last resort — pure-Python content
sniffing over the leading bytes.

Content sniffing uses :mod:`puremagic` (pure Python). It deliberately does **not**
use ``magika``: that classifier pulls in ``onnxruntime``, whose prebuilt wheels
are AVX2-compiled and crash on the x86-64-v1 CPU this parser must run on. No code
path here may import it.
"""

from __future__ import annotations

import enum

from ragsage.models import RawSource

# Only the leading bytes are needed to sniff a container's magic number.
_SNIFF_BYTES = 4096


class DocumentFormat(enum.StrEnum):
    """The parse path a source is routed to.

    ``TEXT`` covers both plain text and Markdown — they share one read-as-is
    path. The remaining members name the structured formats later issues fill
    in; until then the backend rejects them explicitly rather than mis-parsing.
    """

    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    UNKNOWN = "unknown"


# media_type (lower-cased, parameters stripped) -> format.
_MEDIA_TYPES: dict[str, DocumentFormat] = {
    "text/plain": DocumentFormat.TEXT,
    "text/markdown": DocumentFormat.TEXT,
    "text/x-markdown": DocumentFormat.TEXT,
    "application/pdf": DocumentFormat.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        DocumentFormat.DOCX
    ),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        DocumentFormat.PPTX
    ),
    "text/html": DocumentFormat.HTML,
    "application/xhtml+xml": DocumentFormat.HTML,
}

# filename suffix (with leading dot, lower-cased) -> format.
_EXTENSIONS: dict[str, DocumentFormat] = {
    ".txt": DocumentFormat.TEXT,
    ".text": DocumentFormat.TEXT,
    ".md": DocumentFormat.TEXT,
    ".markdown": DocumentFormat.TEXT,
    ".mdown": DocumentFormat.TEXT,
    ".mkd": DocumentFormat.TEXT,
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".pptx": DocumentFormat.PPTX,
    ".html": DocumentFormat.HTML,
    ".htm": DocumentFormat.HTML,
}

# puremagic-reported extension (with leading dot) -> format, for content sniffing.
_SNIFF_EXTENSIONS: dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".pptx": DocumentFormat.PPTX,
    ".html": DocumentFormat.HTML,
    ".htm": DocumentFormat.HTML,
}


def route(source: RawSource, content: bytes) -> DocumentFormat:
    """Resolve ``source`` to a :class:`DocumentFormat`.

    Tries, in order of trustworthiness: the declared ``media_type``, the filename
    extension, then pure-Python magic-byte sniffing. If none identifies a known
    container but the bytes decode as text, it is treated as ``TEXT`` (a bare
    ``.txt``/``.md`` upload with no media type and an unknown extension still
    ingests); otherwise the format is ``UNKNOWN``.
    """
    by_media = _from_media_type(source.media_type)
    if by_media is not None:
        return by_media

    by_ext = _from_extension(source.name)
    if by_ext is not None:
        return by_ext

    by_sniff = _from_content(content)
    if by_sniff is not None:
        return by_sniff

    if _looks_like_text(content):
        return DocumentFormat.TEXT
    return DocumentFormat.UNKNOWN


def _from_media_type(media_type: str | None) -> DocumentFormat | None:
    if not media_type:
        return None
    # Strip any "; charset=..." parameters and normalise case.
    base = media_type.split(";", 1)[0].strip().lower()
    return _MEDIA_TYPES.get(base)


def _from_extension(name: str) -> DocumentFormat | None:
    lowered = name.lower()
    for suffix, fmt in _EXTENSIONS.items():
        if lowered.endswith(suffix):
            return fmt
    return None


def _from_content(content: bytes) -> DocumentFormat | None:
    """Sniff a known container by magic bytes with :mod:`puremagic` (never magika)."""
    if not content:
        return None
    import puremagic

    try:
        matches = puremagic.magic_string(content[:_SNIFF_BYTES])
    except puremagic.PureError:
        return None
    for match in matches:
        fmt = _SNIFF_EXTENSIONS.get((match.extension or "").lower())
        if fmt is not None:
            return fmt
    return None


def _looks_like_text(content: bytes) -> bool:
    """Whether the leading bytes decode cleanly as UTF-8 (so we can read them as text)."""
    if not content:
        return True  # an empty document is a (trivially) valid text document
    try:
        content[:_SNIFF_BYTES].decode("utf-8")
    except UnicodeDecodeError:
        # A truncated multi-byte sequence at the sniff boundary isn't proof of
        # binary content; only a hard failure on the whole prefix is.
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True
