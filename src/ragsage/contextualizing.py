"""Contextual retrieval without a model call — the structure is already in hand.

Anthropic-style contextual retrieval improves a chunk's embedding by prefixing a
sentence that says where the chunk sits in its document. The obvious way to
write that sentence is to ask an LLM, which is what the :class:`Contextualizer`
port was shaped for — and what the backend's adapter does. But that costs one
model call per chunk, adds latency to every ingest, and cannot run in an offline
test: today the only :class:`Contextualizer` shipped *in the library* is
:class:`~ragsage.fakes.FakeContextualizer`, which is a stand-in, so
``IngestionConfig.contextualize=True`` buys a standalone CLI user nothing real.

:class:`HeadingWindowContextualizer` fills that hole with string assembly rather
than inference. The two-pass chunker (:mod:`ragsage.parsing.chunking`) already
carries each section's heading path onto every chunk it emits, and the ingestion
pipeline already hands the contextualizer the document's ``full_text``. Between
them that is enough to say, deterministically, *which document*, *which section*,
and *what was written immediately either side* — most of what an LLM-written
context sentence conveys, at zero model cost, zero latency, and with the same
output for the same input every time. That determinism is the point: it is
testable in the offline CI an LLM contextualizer can never run in.

The window is bounded by a token budget counted with
:func:`~ragsage.parsing.chunking.make_token_counter` — deliberately the *same*
``tiktoken`` encoding the chunker bounds chunks against, so "512-token chunks
plus 128 tokens of context" means one thing across the pipeline rather than two.
When the budget bites, text is dropped at whole paragraph boundaries and, only if
a single paragraph still overflows, at whole sentence boundaries. Nothing is ever
cut mid-word: a half-word contributes a garbage token to the embedding, which is
the opposite of what contextualisation is for.

It is *weaker* than an LLM-written context sentence — it repeats source prose
instead of summarising it, and it adds tokens to ``embed_text``. It is offered as
the honest default for a caller with no model budget, not as a replacement for
the LLM adapter where one is available.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragsage.models import Chunk, Document
from ragsage.parsing.chunking import TokenCounter, make_token_counter

_DEFAULT_ENCODING = "cl100k_base"
"""The chunker's encoding. Named here too so the two budgets are the same unit."""

# A paragraph break is a blank line (with any incidental trailing whitespace on
# it); a sentence break is end punctuation followed by space, *or* a bare line
# break. Counting a line break as a sentence boundary is what keeps the finest
# truncation granularity usable on Markdown that is mostly list items and table
# rows, where a whole block can otherwise be one unpunctuated "sentence".
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def _paragraphs(text: str) -> list[str]:
    """Blank-line-separated paragraphs, stripped, empties dropped."""
    return [part.strip() for part in _PARAGRAPH_BREAK.split(text) if part.strip()]


def _sentences(text: str) -> list[str]:
    """Sentence-ish units within one paragraph — the finest boundary we cut on."""
    return [part.strip() for part in _SENTENCE_BREAK.split(text) if part.strip()]


def _locate(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find where a chunk sits in its document's text, or ``None`` if it can't be.

    An exact substring match is the common case for the fake chunker and for
    prose the real chunker passed through untouched. It is *not* guaranteed: the
    two-pass chunker strips heading lines and re-joins packed units with a fixed
    ``\\n\\n``, so a chunk spanning several source blocks is not verbatim in the
    page text. The fallback anchors on the chunk's first and last non-blank lines
    — those survive repacking — and treats the span between them as the chunk.
    Failing that we return ``None`` and the caller simply omits the window; a
    missing neighbourhood is worth far less than a wrong one.
    """
    stripped = needle.strip()
    if not stripped:
        return None

    exact = haystack.find(stripped)
    if exact >= 0:
        return (exact, exact + len(stripped))

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return None
    start = haystack.find(lines[0])
    if start < 0:
        return None
    end = haystack.find(lines[-1], start)
    if end < 0:
        return (start, start + len(lines[0]))
    return (start, end + len(lines[-1]))


class HeadingWindowContextualizer:
    """A :class:`~ragsage.ports.Contextualizer` built from structure, not inference.

    ``window`` is how many neighbouring paragraphs may be drawn from each side of
    the chunk (0 disables the window entirely, leaving heading context only).
    ``max_context_tokens`` is the hard ceiling on everything prepended to the
    chunk — the heading header *and* both windows — so a caller can reason about
    what contextualisation costs their embedder. ``encoding_name`` names the
    ``tiktoken`` encoding those tokens are counted with; it defaults to the
    chunker's, and there is rarely a reason to change one without the other.

    The encoding is loaded lazily on first use, so constructing this at a
    composition root costs nothing.
    """

    def __init__(
        self,
        *,
        window: int = 2,
        max_context_tokens: int = 128,
        encoding_name: str = _DEFAULT_ENCODING,
    ) -> None:
        if window < 0:
            raise ValueError("window must be non-negative")
        if max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative")
        self._window = window
        self._max_context_tokens = max_context_tokens
        self._encoding_name = encoding_name
        self._count_tokens: TokenCounter | None = None

    async def contextualize(self, document: Document, chunk: Chunk, *, full_text: str) -> str:
        """Return the text to embed: a bounded context header, then the chunk verbatim.

        ``async`` only because the port is — nothing here awaits, touches the
        network, or reads a clock, so the same arguments always produce the same
        string.
        """
        header = self._header(document, chunk)
        before, after = self._neighbours(chunk, full_text)

        # The header is short and the highest-value part of the context, so it is
        # charged against the budget but never itself truncated. What remains is
        # offered to the preceding window first (it usually carries the referents
        # a passage leans on) and whatever that leaves goes to the following one.
        budget = max(0, self._max_context_tokens - self._count(header))
        before_text = self._fit(before, budget=budget // 2, from_end=True)
        after_text = self._fit(after, budget=budget - self._count(before_text), from_end=False)

        parts = [header]
        if before_text:
            parts.append(f"Preceding: {before_text}")
        if after_text:
            parts.append(f"Following: {after_text}")
        parts.append(chunk.text)
        return "\n\n".join(parts)

    # -- context assembly ---------------------------------------------------

    def _header(self, document: Document, chunk: Chunk) -> str:
        """The one- or two-line preamble naming the document and the section."""
        lines = [f"Document: {document.source}"]
        path = _heading_path(chunk)
        if path:
            lines.append(f"Section: {path}")
        return "\n".join(lines)

    def _neighbours(self, chunk: Chunk, full_text: str) -> tuple[str, str]:
        """The raw paragraphs either side of the chunk, before the budget applies."""
        if self._window == 0:
            return ("", "")
        span = _locate(full_text, chunk.text)
        if span is None:
            return ("", "")
        start, end = span
        before = _paragraphs(full_text[:start])[-self._window :]
        after = _paragraphs(full_text[end:])[: self._window]
        return ("\n\n".join(before), "\n\n".join(after))

    # -- budgeting ----------------------------------------------------------

    def _fit(self, text: str, *, budget: int, from_end: bool) -> str:
        """Trim ``text`` to ``budget`` tokens, keeping the end (or the start).

        ``from_end`` keeps the tail — what the preceding window wants, since the
        text nearest the chunk situates it best; the following window keeps the
        head for the mirror-image reason. Whole paragraphs go first; only when not
        even the single nearest paragraph fits do we descend to sentences within
        it. If not even one sentence fits, the window is dropped: an empty window
        is honest, a word-sliced one is noise.
        """
        if budget <= 0 or not text.strip():
            return ""
        if self._count(text) <= budget:
            return text

        paragraphs = _paragraphs(text)
        kept = self._fit_pieces(paragraphs, joiner="\n\n", budget=budget, from_end=from_end)
        if kept:
            return "\n\n".join(kept)

        nearest = paragraphs[-1] if from_end else paragraphs[0]
        sentences = self._fit_pieces(
            _sentences(nearest), joiner=" ", budget=budget, from_end=from_end
        )
        return " ".join(sentences)

    def _fit_pieces(
        self, pieces: Sequence[str], *, joiner: str, budget: int, from_end: bool
    ) -> list[str]:
        """Greedily take whole pieces from the near end while the join stays in budget.

        The budget is checked against the *joined* candidate, separators included,
        so the assembled window never overflows by the cost of its own glue.
        """
        kept: list[str] = []
        for piece in reversed(pieces) if from_end else pieces:
            candidate = joiner.join([piece, *kept] if from_end else [*kept, piece])
            if self._count(candidate) > budget:
                break
            if from_end:
                kept.insert(0, piece)
            else:
                kept.append(piece)
        return kept

    def _count(self, text: str) -> int:
        """Token count under the chunker's encoding; blank text costs nothing.

        The blank-text short-circuit matters because the counter degrades to a
        character estimate when ``tiktoken`` can't load, and that estimate has a
        floor of one token — which would quietly charge an absent window.
        """
        if not text:
            return 0
        if self._count_tokens is None:
            self._count_tokens = make_token_counter(self._encoding_name)
        return self._count_tokens(text)


def _heading_path(chunk: Chunk) -> str:
    """The chunk's ``H1 > H2 > …`` path, as the chunker recorded it, or ``""``.

    Read defensively: ``metadata`` is an open mapping any adapter may populate,
    so a chunker that never wrote ``headings`` (the word-window fake, a caller's
    own) yields no section line rather than an error.
    """
    headings = chunk.metadata.get("headings")
    if not isinstance(headings, (list, tuple)):
        return ""
    parts = [str(heading).strip() for heading in headings if str(heading).strip()]
    return " > ".join(parts)
