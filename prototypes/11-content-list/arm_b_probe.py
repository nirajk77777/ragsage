"""Arm (b): probe what a block-typed domain model breaks. Not production code.

This script only runs with ``arm-b.patch`` applied to ``src/`` — that patch is
the arm-(b) widening itself (``BlockType``, ``Block``, ``Page.blocks``,
``Chunk.block_type/caption/footnote``, a ``chunk_blocks`` pass, and the
plain-text format path emitting blocks):

    git apply prototypes/11-content-list/arm-b.patch
    uv run python prototypes/11-content-list/arm_b_probe.py
    uv run pytest -q                       # count the failures
    git checkout -- src/                   # put production code back

Its measured output is recorded verbatim in ``arm-b-measured.txt`` so the numbers
survive the revert.
"""

from __future__ import annotations

from ragsage.fakes import (
    FakeChunker,
    FakeDocumentParser,
    _chunk_from_dict,
    _chunk_to_dict,
)
from ragsage.models import BlockType, Chunk, RawSource
from ragsage.parsing.backend import HeuristicBackend

MARKDOWN = b"## Catalog\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    rule("B1 — fakes.py silently drops the new typed fields")
    chunk = Chunk(
        id="d:0",
        document_id="d",
        text="| a | b |",
        page=1,
        ordinal=0,
        block_type=BlockType.TABLE,
        caption="Table 1",
        footnote="note",
    )
    as_dict = _chunk_to_dict(chunk)
    revived = _chunk_from_dict(as_dict)
    print(f"  _chunk_to_dict keys           : {sorted(as_dict)}")
    print(f"  block_type after round trip   : {revived.block_type!r}")
    print(f"  caption after round trip      : {revived.caption!r}")
    print(
        "  -> SILENT loss, not a crash. Two functions (fakes.py:103, :115) must gain\n"
        "     three fields each, and until they do the CLI's ingest -> JSON -> query\n"
        "     round trip erases the type with no error. Chunk.metadata has no such\n"
        "     problem: it is already serialised."
    )

    rule("B2 — the block path loses metadata['headings']")
    source = RawSource(name="d.md", content=MARKDOWN, media_type="text/markdown")
    backend = HeuristicBackend()
    parsed = backend.parse(source)
    print(f"  page.blocks                   : {[(str(b.type), b.markdown) for b in parsed.pages[0].blocks]}")
    chunks = backend.chunk(parsed.document, parsed.pages, size=512, overlap=64)
    for produced in chunks:
        print(
            f"  chunk {produced.id}: block_type={str(produced.block_type)!r} "
            f"metadata={dict(produced.metadata)} text={produced.text!r}"
        )
    print(
        "  -> The two-pass chunker's structural pass is DOCUMENT-scoped\n"
        "     (chunking.py:_sections). Chunking block-by-block runs it per block, so\n"
        "     the heading never reaches the table's chunk, and the heading itself\n"
        "     becomes an orphan chunk. metadata['headings'] is now {} where it was\n"
        "     ['Catalog'] — which is exactly what src/ragsage/contextualizing.py:246\n"
        "     (HeadingWindowContextualizer, ticket 07) reads."
    )

    rule("B3 — the shipped fakes produce no blocks and no block types")
    fake_parsed = FakeDocumentParser().parse(source)
    print(f"  FakeDocumentParser page.blocks: {list(fake_parsed.pages[0].blocks)}")
    fake_chunks = FakeChunker().chunk(fake_parsed.document, fake_parsed.pages, size=20, overlap=4)
    print(f"  FakeChunker block_types       : {[c.block_type for c in fake_chunks]}")
    print(
        "  -> The CLI, examples/, and every Seam-1 test run entirely on the fakes, so\n"
        "     under (b) the whole typed-block model is invisible and untested from the\n"
        "     library's own shipped entry points until both fakes are widened too."
    )


if __name__ == "__main__":
    main()
