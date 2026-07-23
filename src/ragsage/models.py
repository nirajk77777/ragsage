"""The domain vocabulary the whole engine speaks.

Every value that crosses a port boundary is one of these frozen dataclasses.
They are deliberately plain data — no behaviour that touches a model, a store,
or the network — so the same objects flow unchanged whether the engine is
driven from a CLI with fakes or from the SaaS backend with real adapters.

The pipeline reads bottom-to-top of this file:

    RawSource -> (parser) -> Document + Page[]  -> (route + chunk) -> Chunk[]
    Chunk[]   -> (embed)  -> EmbeddedChunk[]     -> (store)
    question  -> (retrieve/rerank) -> ScoredChunk[] -> (generate) -> Answer

None of these types know about tenants: isolation is carried by the opaque
:class:`~ragsage.scope.Scope`, which the caller supplies alongside them.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

Vector = tuple[float, ...]
"""A dense embedding. A plain tuple keeps domain values hashable and immutable;
adapters convert to/from numpy or ``pgvector`` at their own edge."""


def _freeze(metadata: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return a read-only copy so a stored value can't be mutated afterwards."""
    return MappingProxyType(dict(metadata or {}))


# --------------------------------------------------------------------------- #
# Source & parsed representation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RawSource:
    """An opaque document to ingest, as handed to :meth:`IngestionPipeline.ingest`.

    A source is either in-memory (``content``) or a filesystem ``path``; the
    :class:`~ragsage.ports.DocumentParser` decides how to read it. Keeping the
    engine agnostic to *where* bytes come from is what lets the CLI feed local
    files and the backend feed object-storage blobs through the same port.
    """

    name: str
    content: bytes | None = None
    path: str | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        if self.content is None and self.path is None:
            raise ValueError("RawSource needs either content or a path")


@dataclass(frozen=True)
class PageImage:
    """A pointer to a page rendered as an image, for the vision/OCR route.

    ``ref`` is an opaque handle the parser understands (a filesystem path, an
    object key); ``data`` is the optional raw image when it is already in hand.
    """

    ref: str
    data: bytes | None = None


@dataclass(frozen=True)
class PageLayout:
    """The geometric signals a parser measures about a page, for route classification.

    A :class:`~ragsage.ports.PageClassifier` weighs these to tell a page whose
    born-digital text layer is usable from a scanned or photographed one that
    only a vision model can read — without re-parsing the page itself.

    ``text_chars`` is the length of the extractable text layer (whether that text
    is rendered or a hidden OCR layer — both count as "there is text to trust").
    ``image_area_ratio`` is the fraction ``[0, 1]`` of the page's area covered by
    raster images; ``bitmap_area`` is the absolute area, in the page's own units,
    of the largest embedded bitmap — a page-filling scan reads high on both while
    a small inline figure on a typed page reads low.
    """

    text_chars: int = 0
    image_area_ratio: float = 0.0
    bitmap_area: float = 0.0


@dataclass(frozen=True)
class Page:
    """One page as the parser found it, before routing decides how to read it.

    A born-digital page arrives with a usable ``text`` layer; a scanned or
    photographed page arrives with an empty ``text`` and an ``image`` to be
    transcribed by the vision model. Most real pages have both. ``layout`` carries
    the measured signals a :class:`~ragsage.ports.PageClassifier` routes on; a
    parser that can't measure them (a flow format with no page grid) leaves it
    ``None`` and the classifier falls back to whether a text layer is present.
    """

    number: int
    text: str = ""
    image: PageImage | None = None
    layout: PageLayout | None = None


class PageRoute(enum.StrEnum):
    """How a page's text gets extracted — the two-path routing decision.

    ``TEXT`` uses the born-digital text layer directly. ``VISION`` sends the
    page image to a premium vision model, skipping any traditional-OCR tier.
    """

    TEXT = "text"
    VISION = "vision"


@dataclass(frozen=True)
class Document:
    """A source document once parsed: identity plus its provenance metadata.

    ``id`` is stable and content-derived (see ``content_hash``) so re-ingesting
    the same bytes is recognisably a duplicate. ``source`` is the human label
    (original filename); ``metadata`` carries anything the caller wants to
    filter on later (author, tags), never anything tenant-shaped.
    """

    id: str
    source: str
    content_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class ParsedDocument:
    """A :class:`Document` together with its ordered pages, as a parser returns it."""

    document: Document
    pages: Sequence[Page]


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of a document, carrying everything a citation needs.

    ``text`` is the verbatim passage shown to the user. ``embed_text`` is what
    actually gets embedded — identical to ``text`` unless contextual retrieval
    has prepended a document-level context sentence. Splitting the two is what
    lets an answer quote the clean passage while retrieval matches on the
    enriched one.
    """

    id: str
    document_id: str
    text: str
    page: int
    ordinal: int
    embed_text: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if not self.embed_text:
            object.__setattr__(self, "embed_text", self.text)


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk paired with its dense embedding, as written to the vector store."""

    chunk: Chunk
    vector: Vector


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with a relevance score attached by retrieval or reranking.

    The same shape flows out of dense search, lexical search, fusion, and the
    reranker; ``score`` is only meaningful relative to its siblings in one list.
    """

    chunk: Chunk
    score: float


# --------------------------------------------------------------------------- #
# Answers & citations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Citation:
    """Binds a marker in the answer text to the exact passage it came from.

    ``marker`` is the ``[n]`` the reader sees; the rest points at the source so
    the UI can open the precise page/passage. A grounded answer's every marker
    resolves to one of these.
    """

    marker: int
    chunk_id: str
    document_id: str
    page: int
    quote: str


@dataclass(frozen=True)
class Turn:
    """One prior exchange in a conversation — a question and the answer it got.

    A sequence of these is the conversation history handed to a query so a
    follow-up can be resolved against it: ``QueryEngine`` condenses the latest
    question plus the turns before it into a standalone question before
    retrieving. Deliberately plain — no citations or usage — because rewriting a
    pronoun only needs the words that were said, not how they were sourced.
    """

    question: str
    answer: str


@dataclass(frozen=True)
class Answer:
    """The result of a query: grounded prose plus verifiable citations.

    When the corpus can't support an answer, ``grounded`` is ``False``, ``text``
    is the honest not-found message, and ``citations`` is empty — the engine
    never fabricates a confident answer out of thin retrieval.
    """

    text: str
    citations: Sequence[Citation] = ()
    grounded: bool = True


# --------------------------------------------------------------------------- #
# Streamed answer events
# --------------------------------------------------------------------------- #
#
# A streamed query yields these one at a time, in this order: zero or more
# :class:`AnswerToken`, then zero or more :class:`Citation` (only for a grounded
# answer), then one :class:`Usage`, then a terminal :class:`AnswerComplete`. The
# backend maps each to an SSE event; the same objects assemble into an
# :class:`Answer` for the non-streaming path, so the two never diverge.


@dataclass(frozen=True)
class AnswerToken:
    """One streamed fragment of the answer text, as the model emits it."""

    text: str


@dataclass(frozen=True)
class Usage:
    """A best-effort accounting of one answer's shape.

    ``sources`` is how many chunks were placed in the model's context;
    ``completion_tokens`` counts the fragments streamed back. These are what the
    engine can observe without a tokenizer — provider-reported token counts can
    layer in later through the gateway without changing this shape.
    """

    sources: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class AnswerComplete:
    """The terminal streamed event: the fully-assembled, citation-bound answer.

    Carries the same fields as :class:`Answer`, so a consumer that only wants the
    final result can ignore the incremental events and read this one.
    """

    text: str
    citations: Sequence[Citation] = ()
    grounded: bool = True


AnswerEvent = AnswerToken | Citation | Usage | AnswerComplete
"""The union a streamed query yields; consumers dispatch on the concrete type."""
