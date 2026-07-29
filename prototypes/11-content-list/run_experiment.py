"""Run both arms end to end and print the evidence. Not production code.

    uv run python prototypes/11-content-list/run_experiment.py

Ingests the same typed content list (text, table, image, equation, custom) twice
— once collapsed per page (a1), once attributed per item (a2) — through the real
chunker, the real fakes, and the real ``QueryEngine``, then reports what each arm
can still say about the table at retrieval time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from arm_a import (  # noqa: E402
    ItemIngestionPrototype,
    chunk_summary,
    looks_like_a_table,
    metadata_keys,
)
from items import SAMPLE_CONTENT_LIST, normalise  # noqa: E402
from ragsage import IngestionConfig, QueryEngine, Scope  # noqa: E402
from ragsage.fakes import FakeEngineKit  # noqa: E402
from ragsage.models import Document  # noqa: E402
from ragsage.parsing.backend import HeuristicBackend  # noqa: E402

CONFIG = IngestionConfig(chunk_size=512, chunk_overlap=64, contextualize=False)
QUESTION = "What is the tariff rate for lunar injection?"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def document_for_items(items: list[object], name: str) -> Document:
    """Identity for an item list. There are no source bytes to hash — see §finding."""
    payload = json.dumps(SAMPLE_CONTENT_LIST, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return Document(id=digest[:16], source=name, content_hash=digest)


def build(kit: FakeEngineKit) -> ItemIngestionPrototype:
    return ItemIngestionPrototype(
        contextualizer=kit.contextualizer,
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        cache=kit.cache,
    )


async def main() -> None:
    items = normalise(SAMPLE_CONTENT_LIST)
    scope = Scope(namespace="proto")

    # ------------------------------------------------------------------ #
    rule("FINDING 0 — the Chunker port cannot be used at all without a parse")
    # ------------------------------------------------------------------ #
    backend = HeuristicBackend()
    proto = build(FakeEngineKit())
    doc = document_for_items(items, "tariffs.content_list.json")
    pages = proto.to_pages(items)
    try:
        backend.chunk(doc, pages, size=512, overlap=64)
        print("  (no error — unexpected)")
    except Exception as exc:  # noqa: BLE001
        print(f"  HeuristicBackend.chunk(...) without parse() -> {type(exc).__name__}: {exc}")
    print(
        "  The shipped Chunker (backend.py:128) pops an in-flight stash keyed by\n"
        "  document id, filled only by parse(). A seam that bypasses parsing cannot\n"
        "  call the shipped chunker through its own port. Both arms hit this."
    )

    # ------------------------------------------------------------------ #
    rule("ARM a1 — collapse every item on a page into one Markdown page")
    # ------------------------------------------------------------------ #
    kit_a1 = FakeEngineKit()
    proto_a1 = build(kit_a1)
    pages_a1 = proto_a1.to_pages(items)
    chunks_a1 = proto_a1.chunk_collapsed(
        doc, pages_a1, size=CONFIG.chunk_size, overlap=CONFIG.chunk_overlap
    )
    await proto_a1.store(doc, chunks_a1, pages_a1, scope, CONFIG)

    print(f"  items in           : {len(items)}")
    print(f"  pages built        : {len(pages_a1)}")
    print(f"  chunks produced    : {len(chunks_a1)}")
    print(f"  metadata keys seen : {sorted(metadata_keys(chunks_a1))}")
    for chunk in chunks_a1:
        print(f"    {chunk_summary(chunk)}")
    table_chunks_a1 = [c for c in chunks_a1 if looks_like_a_table(c.text)]
    print(f"  chunks whose text contains table syntax: {len(table_chunks_a1)}")
    for chunk in table_chunks_a1:
        pipe_lines = sum(1 for line in chunk.text.splitlines() if "|" in line)
        total_lines = len([line for line in chunk.text.splitlines() if line.strip()])
        print(
            f"    {chunk.id}: {pipe_lines}/{total_lines} non-blank lines are table rows"
            f" -> the chunk is {'PURE table' if pipe_lines == total_lines else 'MIXED prose+table'}"
        )
        print("    ---- verbatim chunk text ----")
        for line in chunk.text.splitlines():
            print(f"    | {line}")
        print("    -----------------------------")

    # ------------------------------------------------------------------ #
    rule("ARM a2 — same Markdown collapse, but chunked per item with metadata")
    # ------------------------------------------------------------------ #
    kit_a2 = FakeEngineKit()
    proto_a2 = build(kit_a2)
    chunks_a2 = proto_a2.chunk_per_item(
        doc, items, size=CONFIG.chunk_size, overlap=CONFIG.chunk_overlap
    )
    await proto_a2.store(doc, chunks_a2, pages_a1, scope, CONFIG)

    print(f"  chunks produced    : {len(chunks_a2)}")
    print(f"  metadata keys seen : {sorted(metadata_keys(chunks_a2))}")
    for chunk in chunks_a2:
        print(f"    {chunk_summary(chunk)}")
    by_type: dict[str, int] = {}
    for chunk in chunks_a2:
        key = str(chunk.metadata.get("block_type", "?"))
        by_type[key] = by_type.get(key, 0) + 1
    print(f"  selectable by type : {by_type}")

    # ------------------------------------------------------------------ #
    rule("FINDING 1 — item -> chunk attribution: which items stay addressable?")
    # ------------------------------------------------------------------ #
    # A per-item fingerprint: its longest rendered line. Robust to the heading
    # stripping the structural pass does, unlike matching the whole rendering.
    probes = [
        (item.type, max(item.to_markdown().splitlines(), key=len).strip())
        for item in items
        if item.to_markdown().strip()
    ]
    for label, chunks in (("a1", chunks_a1), ("a2", chunks_a2)):
        addressable = 0
        for chunk in chunks:
            contained = [t for t, probe in probes if probe and probe in chunk.text]
            marker = "1:1" if len(contained) == 1 else f"{len(contained)} items fused"
            print(f"  [{label}] {chunk.id}: {contained} -> {marker}")
            if len(contained) == 1:
                addressable += 1
        print(f"  [{label}] chunks that are exactly one item: {addressable}/{len(chunks)}\n")

    # ------------------------------------------------------------------ #
    rule("FINDING 2 — does the type survive persistence? (JSON snapshot round-trip)")
    # ------------------------------------------------------------------ #
    snapshot = json.loads(json.dumps(kit_a2.snapshot()))
    revived = FakeEngineKit()
    revived.restore(snapshot)
    restored = await revived.lexical_store.search(scope, "tariff rate lunar injection", k=10)
    print(f"  chunks restored from JSON: {sum(len(v) for v in snapshot['lexical'].values())}")
    for scored in restored:
        print(f"    {scored.chunk.id} metadata={dict(scored.chunk.metadata)}")
    survived = any("block_type" in s.chunk.metadata for s in restored)
    print(f"  block_type survived snapshot -> restore -> search: {survived}")
    print(
        "  Chunk.metadata is already a free-form Mapping (models.py:171), already\n"
        "  serialised by the fakes (fakes.py:103) and already persisted as\n"
        "  `metadata jsonb` by ADR-0002 §1. No model change was needed for this."
    )

    # ------------------------------------------------------------------ #
    rule("FINDING 3 — what each arm can say about the table at RETRIEVAL time")
    # ------------------------------------------------------------------ #
    for label, kit in (("a1", kit_a1), ("a2", kit_a2)):
        engine = QueryEngine(
            embedder=kit.embedder,
            vector_store=kit.vector_store,
            lexical_store=kit.lexical_store,
            reranker=kit.reranker,
            llm=kit.llm,
        )
        answer = await engine.query(QUESTION, scope)
        print(f"\n  [{label}] answer  : {answer.text[:110]}")
        print(f"  [{label}] grounded: {answer.grounded}, citations: {len(answer.citations)}")
        hits = await kit.lexical_store.search(scope, QUESTION, k=5)
        for scored in hits[:3]:
            md = dict(scored.chunk.metadata)
            print(
                f"  [{label}] retrieved {scored.chunk.id} "
                f"block_type={md.get('block_type', '<absent>')} "
                f"caption={md.get('caption', '<absent>')!r} "
                f"text-looks-like-a-table={looks_like_a_table(scored.chunk.text)}"
            )

    # ------------------------------------------------------------------ #
    rule("FINDING 4 — could a BlockEnricher (report #6) target tables under each arm?")
    # ------------------------------------------------------------------ #
    a1_targets = [c for c in chunks_a1 if looks_like_a_table(c.text)]
    a2_targets = [c for c in chunks_a2 if c.metadata.get("block_type") == "table"]
    print(f"  a1: selectable only by re-sniffing Markdown -> {len(a1_targets)} candidate(s)")
    for chunk in a1_targets:
        prose = [ln for ln in chunk.text.splitlines() if ln.strip() and "|" not in ln]
        print(f"      {chunk.id} carries {len(prose)} non-table line(s) an enricher would")
        print("      have to describe as if they were part of the table:")
        for line in prose:
            print(f"        {line!r}")
    print(f"  a2: selectable by metadata -> {len(a2_targets)} candidate(s)")
    for chunk in a2_targets:
        print(
            f"      {chunk.id} caption={chunk.metadata.get('caption')!r} "
            f"footnote={chunk.metadata.get('footnote')!r} (separable, not inlined prose)"
        )

    # ------------------------------------------------------------------ #
    rule("FINDING 5 — what a2 costs: packing across items is gone")
    # ------------------------------------------------------------------ #
    print(f"  a1 chunks: {len(chunks_a1)}   a2 chunks: {len(chunks_a2)}")
    print(
        "  a1's _pack() merges small neighbouring units into one chunk\n"
        "  (chunking.py:238); a2 chunks each item in isolation, so a one-line\n"
        "  custom item becomes its own chunk. Same corpus, more and smaller chunks."
    )
    smallest_a1 = min(len(c.text) for c in chunks_a1)
    smallest_a2 = min(len(c.text) for c in chunks_a2)
    print(f"  smallest chunk: a1={smallest_a1} chars, a2={smallest_a2} chars")


if __name__ == "__main__":
    asyncio.run(main())
