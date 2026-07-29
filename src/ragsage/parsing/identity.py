"""Content-derived document identity, shared by every real parser path.

A document's id must be *byte-for-byte* reproducible from its content so that
re-uploading the same file dedups and the ``rag_document_id`` key stays stable
whichever parser produced it. The convention is: the sha256 of the raw bytes is
the content hash and its 16-character prefix is the id.

Every parser *inside this library* derives identity through this one helper — the
heuristic backend and the in-package fakes — so a new format path can't drift
from the convention by accident. The Docling adapter lives in the backend repo
and cannot import this, so it deliberately mirrors the same computation there.
"""

from __future__ import annotations

import hashlib

from ragsage.models import Document, RawSource

_ID_PREFIX_LEN = 16


def read_bytes(source: RawSource) -> bytes:
    """The raw bytes of a source, from memory or its path.

    Delegates to :meth:`RawSource.read`, which is where this lives now that the
    ingestion pipeline needs the same bytes to key its parse cache. Kept as a
    function because every parser path in this package already calls it by name.
    """
    return source.read()


def document_for(source: RawSource, content: bytes) -> Document:
    """A content-derived :class:`Document` identity (stable across re-ingest).

    The sha256 of ``content`` is the content hash and its 16-char prefix the id,
    matching the fakes and the Docling adapter so dedup and ``rag_document_id``
    keying are unchanged no matter which parser ran.
    """
    digest = hashlib.sha256(content).hexdigest()
    return Document(id=digest[:_ID_PREFIX_LEN], source=source.name, content_hash=digest)
