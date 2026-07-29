"""Unit coverage for :class:`ragsage.parsing.HeuristicBackend` over real bytes.

Unlike the Docling adapter — whose heavy parse/chunk runs only on the integration
path — the heuristic backend is pure Python, so its whole contract is exercised
here with hand-authored Markdown fixtures: content-derived identity and the
dual-port stash, the format router's precedence (and its refusal to touch
``magika``), the plain-text/Markdown parse, and every branch of the two-pass
chunker (heading metadata, token bounding, tables kept whole, oversized tables
row-split with the header repeated). A final case drives a full ingestion through
the pipeline to prove real text flows end-to-end.
"""

from __future__ import annotations

import sys

from ragsage.fakes import FakeEngineKit
from ragsage.ingestion import IngestionPipeline
from ragsage.models import Document, RawSource
from ragsage.parsing import DocumentFormat, HeuristicBackend, route
from ragsage.parsing.chunking import chunk_markdown, make_token_counter
from ragsage.ports import Chunker, DocumentParser
from ragsage.scope import Scope

# --------------------------------------------------------------------------- #
# Fixtures — hand-authored Markdown
# --------------------------------------------------------------------------- #

_HEADINGS_DOC = """\
# Quarterly Report

Opening remarks for the quarter.

## Revenue

Revenue rose across every region.

### North America

Strong growth in enterprise accounts.

## Costs

Costs were held flat.
"""

_TABLE_DOC = """\
## Inventory

A short note before the table.

| Item | Qty | Notes |
| --- | --- | --- |
| Apple | 3 | fresh |
| Pear | 5 | ripe |

A short note after the table.
"""


def _wide_table_doc(rows: int) -> str:
    body = "\n".join(
        f"| Item {i} | Category number {i} | A fuller description of row {i} goes here |"
        for i in range(1, rows + 1)
    )
    return "## Catalog\n\n| Item | Category | Description |\n| --- | --- | --- |\n" + body + "\n"


def _md_source(
    text: str, *, name: str = "doc.md", media_type: str | None = "text/markdown"
) -> RawSource:
    return RawSource(name=name, content=text.encode("utf-8"), media_type=media_type)


# --------------------------------------------------------------------------- #
# Dual-port shape and the in-flight stash
# --------------------------------------------------------------------------- #


def test_backend_satisfies_both_ports_by_shape() -> None:
    backend = HeuristicBackend()
    assert isinstance(backend, DocumentParser)
    assert isinstance(backend, Chunker)


def test_stash_is_released_on_chunk() -> None:
    # parse stashes the intermediate keyed by document id so it is not retained for
    # the backend's lifetime; chunk releases it. Chunking itself reads the resolved
    # pages it is passed, so releasing the stash must not make chunking fail.
    backend = HeuristicBackend()
    parsed = backend.parse(_md_source(_TABLE_DOC))

    first = backend.chunk(parsed.document, parsed.pages, size=512, overlap=64)
    assert first
    assert not backend._parsed, "the stash should not outlive chunking"

    # Chunking again is fine and gives the same answer. It has to be: once the
    # pipeline caches parse output, a re-index chunks documents this backend never
    # parsed, so an absent stash is a normal state rather than an error.
    again = backend.chunk(parsed.document, parsed.pages, size=512, overlap=64)
    assert [c.text for c in again] == [c.text for c in first]


def test_chunking_a_document_this_backend_never_parsed_is_allowed() -> None:
    # The parse-cache path: pages arrive from ragsage.caching, not from parse.
    parsed = HeuristicBackend().parse(_md_source(_TABLE_DOC))

    fresh = HeuristicBackend()
    chunks = fresh.chunk(parsed.document, parsed.pages, size=512, overlap=64)

    assert chunks


# --------------------------------------------------------------------------- #
# Content-derived document identity (byte-for-byte compatible with the fakes)
# --------------------------------------------------------------------------- #


def test_document_identity_is_content_derived_and_stable() -> None:
    backend = HeuristicBackend()
    source = _md_source("# Title\n\nbody\n")

    first = backend.parse(source).document
    again = HeuristicBackend().parse(source).document

    assert first.id == again.id  # stable across a re-parse
    assert first.content_hash == again.content_hash
    assert first.id == first.content_hash[:16]  # id is the 16-char hash prefix
    assert first.source == "doc.md"


def test_different_bytes_get_a_different_identity() -> None:
    backend = HeuristicBackend()
    one = backend.parse(_md_source("# A\n\none\n")).document
    two = backend.parse(_md_source("# A\n\ntwo\n")).document
    assert one.content_hash != two.content_hash
    assert one.id != two.id


# --------------------------------------------------------------------------- #
# Format router: media_type -> extension -> content sniff, and never magika
# --------------------------------------------------------------------------- #


def test_router_prefers_media_type_then_extension_then_sniff() -> None:
    text = b"# just markdown\n\nwith a paragraph\n"
    # 1. media_type wins even when the extension disagrees.
    assert route(RawSource(name="mystery.bin", content=text, media_type="text/markdown"), text) is (
        DocumentFormat.TEXT
    )
    # 2. no media_type -> the extension decides.
    assert route(RawSource(name="notes.md", content=text), text) is DocumentFormat.TEXT
    # 3. no media_type and an unknown extension -> decodable bytes sniff as text.
    assert route(RawSource(name="notes.unknown", content=text), text) is DocumentFormat.TEXT
    # A real PDF's magic bytes are recognised by content alone.
    pdf = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n"
    assert route(RawSource(name="scan", content=pdf), pdf) is DocumentFormat.PDF


def test_router_does_not_import_magika() -> None:
    # Sniffing binary content must go through puremagic, never magika (onnxruntime).
    pdf = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
    route(RawSource(name="x", content=pdf), pdf)
    assert "magika" not in sys.modules


# --------------------------------------------------------------------------- #
# Plain-text / Markdown parse -> a single layout-free page
# --------------------------------------------------------------------------- #


def test_text_parses_to_single_page_with_no_layout() -> None:
    backend = HeuristicBackend()
    parsed = backend.parse(_md_source(_HEADINGS_DOC))

    assert len(parsed.pages) == 1
    page = parsed.pages[0]
    assert page.number == 1
    assert page.layout is None  # flow format: nothing to measure, classifier falls back to text
    assert page.image is None
    assert "Quarterly Report" in page.text


def test_plain_text_without_media_type_or_markdown_extension_parses() -> None:
    backend = HeuristicBackend()
    parsed = backend.parse(RawSource(name="README", content=b"plain text, no headings at all"))
    assert len(parsed.pages) == 1
    chunks = backend.chunk(parsed.document, parsed.pages, size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "plain text, no headings at all"
    assert chunks[0].metadata.get("headings") in (None, [])


# --------------------------------------------------------------------------- #
# Two-pass chunker: heading metadata, token bounding, tables
# --------------------------------------------------------------------------- #


def test_chunks_carry_their_heading_path() -> None:
    backend = HeuristicBackend()
    parsed = backend.parse(_md_source(_HEADINGS_DOC))
    chunks = backend.chunk(parsed.document, parsed.pages, size=512, overlap=64)

    by_heading = {tuple(c.metadata.get("headings", ())): c for c in chunks}
    # The nested section carries its full H1 > H2 > H3 path.
    assert ("Quarterly Report", "Revenue", "North America") in by_heading
    assert ("Quarterly Report", "Costs") in by_heading
    # Every chunk records its source page.
    assert all(c.page == 1 for c in chunks)
    # Ordinals are dense and ordered.
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_small_table_is_kept_whole() -> None:
    backend = HeuristicBackend()
    parsed = backend.parse(_md_source(_TABLE_DOC))
    chunks = backend.chunk(parsed.document, parsed.pages, size=512, overlap=64)

    # The whole table survives verbatim inside a single chunk — no row was split off.
    table_lines = ["| Apple | 3 | fresh |", "| Pear | 5 | ripe |", "| Item | Qty | Notes |"]
    holders = [c for c in chunks if all(line in c.text for line in table_lines)]
    assert len(holders) == 1


def test_oversized_table_is_split_by_rows_with_header_repeated() -> None:
    backend = HeuristicBackend()
    parsed = backend.parse(_md_source(_wide_table_doc(rows=14)))
    # A tight budget forces the single table to be split across several chunks.
    chunks = backend.chunk(parsed.document, parsed.pages, size=48, overlap=8)

    assert len(chunks) > 1
    header = "| Item | Category | Description |"
    delimiter = "| --- | --- | --- |"
    # Every piece repeats the header + delimiter so each stays a readable table.
    assert all(header in c.text and delimiter in c.text for c in chunks)
    # No data row is lost across the split.
    for i in range(1, 15):
        assert any(f"| Item {i} |" in c.text for c in chunks)
    # The heading path rides along on every piece.
    assert all(c.metadata.get("headings") == ["Catalog"] for c in chunks)


def test_oversized_prose_is_token_bounded() -> None:
    count = make_token_counter()
    sentence = "The quick brown fox jumps over the lazy dog and then keeps on running. "
    long_body = "# Log\n\n" + sentence * 60
    chunks = chunk_markdown(
        long_body, document_id="deadbeefdeadbeef", page=1, size=64, overlap=8, count_tokens=count
    )
    assert len(chunks) > 1
    # Each chunk is bounded near the budget. The recursive splitter's token count
    # is approximate at a boundary (separator accounting can nudge it a token or
    # two over), so allow a small slack rather than a hard equality — production
    # pins ``size`` with headroom for exactly this reason.
    assert all(count(c.text) <= 64 + 4 for c in chunks)
    # Without bounding, this prose would be one oversized chunk many times the budget.
    assert max(count(c.text) for c in chunks) < count(long_body)


def test_deep_h5_h6_headings_reach_chunk_metadata() -> None:
    # HTML and DOCX carry the full <h1>…<h6> hierarchy, so the structural pass
    # must split on all six levels — a deep h5/h6 section's heading context must
    # reach metadata rather than being swallowed into its parent section.
    count = make_token_counter()
    doc = "# Top\n\nintro\n\n##### Deep Five\n\nfive body\n\n###### Deeper Six\n\nsix body\n"
    chunks = chunk_markdown(
        doc, document_id="deadbeefdeadbeef", page=1, size=512, overlap=64, count_tokens=count
    )
    paths = {tuple(c.metadata.get("headings", ())) for c in chunks}
    assert ("Top", "Deep Five") in paths
    assert ("Top", "Deep Five", "Deeper Six") in paths


def test_token_counting_needs_no_transformers() -> None:
    # The pinned tokenizer is tiktoken; the heavy transformers/torch stack must
    # never be dragged in just to count tokens.
    count = make_token_counter()
    assert count("hello world") >= 1
    assert "transformers" not in sys.modules
    assert "torch" not in sys.modules


# --------------------------------------------------------------------------- #
# End-to-end: real text bytes flow through the ingestion pipeline
# --------------------------------------------------------------------------- #


async def test_text_ingests_end_to_end_through_the_pipeline() -> None:
    kit = FakeEngineKit()
    backend = HeuristicBackend()
    # The heuristic backend is both the parser and the chunker; every other port
    # is a fake, exactly as the real composition root wires the two halves.
    pipeline = IngestionPipeline(
        parser=backend,
        classifier=kit.classifier,
        chunker=backend,
        contextualizer=kit.contextualizer,
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        llm=kit.llm,
        cache=kit.cache,
        tracer=kit.tracer,
    )
    scope = Scope(namespace="local")

    result = await pipeline.ingest(_md_source(_HEADINGS_DOC, name="report.md"), scope)

    assert not result.deduplicated
    assert result.chunk_count > 0
    assert result.document.id == result.document.content_hash[:16]
    # The document and its chunks actually landed in the stores.
    stored = await kit.document_store.get(scope, result.document.id)
    assert isinstance(stored, Document)
    assert kit.vector_store.snapshot()["local"]

    # Re-ingesting the identical bytes dedups instead of reprocessing.
    again = await pipeline.ingest(_md_source(_HEADINGS_DOC, name="report.md"), scope)
    assert again.deduplicated
