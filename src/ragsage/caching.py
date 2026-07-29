"""Keys and wire format for caching parse output, over the existing ``Cache`` port.

Parsing is the most expensive deterministic step in ingestion — the whole PDF
stack runs, per page — and its result is a pure function of the document's bytes
and the parser that read them. That makes it exactly the thing worth caching, and
re-indexing is what makes it matter: the embedding model is pinned per corpus, so
changing it (or changing chunk size) currently means re-parsing every document to
get chunks to re-embed. A parse cache is what makes that affordable.

Two design decisions are load-bearing.

**Keyed by content hash, never by path or filename.** The same bytes moved,
renamed, or re-uploaded under a different name must hit. RAG-Anything normalises
image paths to basenames for precisely this reason; we already compute a
content-derived id in :mod:`ragsage.parsing.identity`, so the hash is free.

**The parser is part of the key, the chunking parameters are not.** The key
carries a fingerprint of the parser that produced the entry plus a format version
for this module, so swapping parsers or changing the wire format misses rather
than deserialising something stale into a shape that no longer matches. Chunk
size and overlap are deliberately *absent*: what is cached is the parse output —
pages — and chunking runs on every ingest regardless. A caller who changes
``chunk_size`` gets new chunks immediately, so there is no staleness for the key
to guard against. Including them would only lower the hit rate for nothing.

The :class:`~ragsage.ports.Cache` port is unchanged — still ``str -> str | None``.
That is why this module exists: pages have to survive a round trip through a
single string, and every decode failure has to degrade to a miss rather than an
exception, because the port's stated contract is that the engine is correct with
the cache absent, wrong, or empty.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from ragsage.models import Document, Page, PageImage, PageLayout, ParsedDocument

#: Bumped when the encoding below changes shape. An old entry then misses rather
#: than decoding into something subtly wrong — cheaper than a migration, and the
#: only safe option for a cache we do not enumerate or clear.
PARSE_CACHE_VERSION = 1


def parser_fingerprint(parser: object) -> str:
    """A stable-enough identity for the parser that produced a cache entry.

    The :class:`~ragsage.ports.DocumentParser` port has no version field — adding
    one would burden every adapter for this feature's benefit — so the import path
    of the parser's class stands in. It distinguishes the cases that matter in
    practice: a different parser, a fake versus the real backend, a consumer's own
    implementation.

    What it does **not** catch is the same class changing its output, e.g. a
    heuristic tuned between releases. :data:`PARSE_CACHE_VERSION` covers this
    module's own format; a parser whose behaviour changes needs its consumer to
    accept one stale generation or key on something more specific. Documented
    rather than papered over.
    """
    cls = type(parser)
    return f"{cls.__module__}.{cls.__qualname__}"


def parse_cache_key(content_hash: str, parser: object) -> str:
    """The cache key for one document's parse output."""
    return f"parse:v{PARSE_CACHE_VERSION}:{parser_fingerprint(parser)}:{content_hash}"


def encode_parsed(parsed: ParsedDocument) -> str:
    """Serialise parse output to a single JSON string for the cache.

    Image bytes are base64-encoded rather than dropped: a scanned page's
    :class:`~ragsage.models.PageImage` is what the vision route needs, and a cache
    entry that silently lost it would turn a hit into an empty document. It does
    make entries for image-heavy documents large — sizing the cache is the
    deployment's call, and a value too big for the backing store simply fails to
    set, which degrades to a miss.
    """
    return json.dumps(
        {
            "document": {
                "id": parsed.document.id,
                "source": parsed.document.source,
                "content_hash": parsed.document.content_hash,
                "metadata": dict(parsed.document.metadata),
            },
            "pages": [_encode_page(page) for page in parsed.pages],
        },
        separators=(",", ":"),
    )


def decode_parsed(payload: str) -> ParsedDocument | None:
    """Deserialise a cache entry, or return ``None`` if it cannot be trusted.

    Every malformed, truncated, or foreign payload becomes a miss. A cache is
    shared, long-lived, and outside our control; treating a bad entry as absent is
    the only behaviour that keeps the port's "correct without the cache" contract
    true. Deliberately broad: there is no failure here worth propagating over
    simply parsing the document again.
    """
    try:
        raw = json.loads(payload)
        document = raw["document"]
        return ParsedDocument(
            document=Document(
                id=document["id"],
                source=document["source"],
                content_hash=document["content_hash"],
                metadata=document.get("metadata") or {},
            ),
            pages=tuple(_decode_page(page) for page in raw["pages"]),
        )
    except Exception:
        return None


def _encode_page(page: Page) -> dict[str, Any]:
    encoded: dict[str, Any] = {"number": page.number, "text": page.text}
    if page.image is not None:
        encoded["image"] = {
            "ref": page.image.ref,
            "data": None if page.image.data is None else base64.b64encode(page.image.data).decode(),
        }
    if page.layout is not None:
        encoded["layout"] = {
            "text_chars": page.layout.text_chars,
            "image_area_ratio": page.layout.image_area_ratio,
            "bitmap_area": page.layout.bitmap_area,
        }
    return encoded


def _decode_page(encoded: dict[str, Any]) -> Page:
    image = encoded.get("image")
    layout = encoded.get("layout")
    return Page(
        number=encoded["number"],
        text=encoded.get("text", ""),
        image=None
        if image is None
        else PageImage(
            ref=image["ref"],
            data=None if image.get("data") is None else base64.b64decode(image["data"]),
        ),
        layout=None
        if layout is None
        else PageLayout(
            text_chars=layout["text_chars"],
            image_area_ratio=layout["image_area_ratio"],
            bitmap_area=layout["bitmap_area"],
        ),
    )
