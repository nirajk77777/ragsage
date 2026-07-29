# 0004 — Typed blocks ride in `Chunk.metadata`, not in the domain model

- Status: **Proposed — drafted from the ticket-11 prototype, unmerged**
- Date: 2026-07-30

> Drafted on the throwaway branch `11-content-list-proto` alongside the prototype
> that produced its evidence (`prototypes/11-content-list/`). Nothing from that
> branch merges. Promote this ADR only when `ingest_items` is actually scheduled.

## Context

Report item #5 proposes a public, parser-agnostic insertion seam,
`IngestionPipeline.ingest_items(items, scope)`, taking an already-parsed typed
sequence (`{type, page_idx, …}` for text, image, table, equation and custom
types) and bypassing parsing entirely — mirroring RAG-Anything's
`insert_content_list`. It buys bring-your-own-parser (including a cloud parser we
do not want as a dependency) and, with ticket 09's parse cache, the whole
re-index story.

The seam's input is *typed*; our domain model is not. `Page` is a page of
Markdown (`models.py:94`) and `Chunk` is a passage of text with a page number
(`models.py:155`). So the seam must either collapse the typed items into Markdown
at its boundary, or the domain model must learn what a block is — which would be
the first real widening since ADR-0001.

Ticket 11 prototyped both. Evidence and reproduction steps are in
`prototypes/11-content-list/README.md`; the numbers below come from
`arm-a-measured.txt` and `arm-b-measured.txt`.

## Decision

**`ingest_items` collapses each item to Markdown at the boundary and carries the
item's type, caption and footnote in `Chunk.metadata`. The domain model does not
gain a block type.**

Concretely:

1. **One chunking call per item, not per page.** The seam chunks each item's
   Markdown separately so every chunk descends from exactly one item. That is
   what makes the type meaningful; collapsing a whole page first destroys the
   item→chunk correspondence irrecoverably.
2. **`metadata["block_type"]` is the type**, alongside `metadata["caption"]` and
   `metadata["footnote"]` when the item carried them. `Chunk.metadata` is already
   a free-form `Mapping[str, object]` (`models.py:171`), already serialised by
   the fakes (`fakes.py:103`), and already persisted as `metadata jsonb` by
   ADR-0002 §1 — which that ADR defines as *payload the consumer can attach and
   filter on*. A block type is exactly that payload. Nothing new is required for
   it to reach retrieval.
3. **Unknown types pass through verbatim.** No enum, no allow-list. A foreign
   parser's custom type is stored as it arrived, matching the seam's whole reason
   to exist.
4. **The seam owns the heading path.** Chunking per item makes the chunker's
   structural pass item-scoped, so a table item cannot see the heading a
   preceding text item carried. The seam maintains a document-scoped ATX heading
   stack and writes the full path into `metadata["headings"]`, so
   `HeadingWindowContextualizer` (`contextualizing.py:246`) and citations behave
   as they do on the parsed path.
5. **`ingest_items` needs two refactors of existing code, and no new port.** The
   post-parse tail of `IngestionPipeline.ingest` (`ingestion.py:112-137` — chunk,
   contextualise, embed, store) is extracted into a shared private method; and
   the seam must be able to chunk without a parse, which today it cannot
   (`HeuristicBackend.chunk` pops an in-flight stash keyed by document id,
   `backend.py:128`, and raises `RuntimeError: no parsed document stashed for …`).
   Either the stash becomes optional, or the seam calls `chunk_markdown`
   directly. This is a cost of the *seam*, under either option.
6. **Document identity for an item list is the seam's to define.** There are no
   source bytes to hash, so `ingest_items` takes a caller-supplied `Document`, or
   hashes a canonical serialisation of the item list. Content-hash dedup
   (`ports.py:194`) keeps working either way; ADR-0001's identity convention
   (`identity.py:32`) is about *bytes* and does not reach here.

## Considered options

### (a1) Collapse each page's items into one Markdown page — rejected

The literal reading of "collapse at the boundary", and it does throw the type
away. Measured on a 7-item fixture (text ×3, table, image, equation, custom):

- 7 items became **3 chunks**; only **1 of 3** descends from a single item. One
  chunk fuses **four** items — text, image, equation and the custom type.
- The only metadata key produced is `headings`. There is no type anywhere.
- The chunk holding the table is **mixed prose + table** — 5 of 8 non-blank lines
  are table rows, the rest being the section's lead sentence, the caption and the
  footnote, now indistinguishable prose. So the type is not merely absent, it is
  *unrecoverable*: re-sniffing the Markdown finds table syntax in that chunk but
  cannot say the chunk *is* a table, and an enricher pointed at it would be
  describing three lines of prose as if they were part of the table.

The type does survive in one degraded form: the table's *text* is still
retrievable by the tokens printed in it (the fixture's table chunk is retrieved
and cited for "what is the tariff rate for lunar injection?"). That is exactly
today's behaviour and exactly the failure mode report #6 exists to fix — a table
findable by its cell contents and by nothing else.

### (b) Introduce a `Block` type into the domain model — rejected for now

Prototyped as `arm-b.patch`: `BlockType`, `Block`, `Page.blocks`,
`Chunk.block_type/caption/footnote`, a `chunk_blocks` pass, and the plain-text
format path emitting blocks — 130 insertions across `models.py`,
`parsing/chunking.py`, `parsing/backend.py`.

It fails on three counts, all measured:

- **It is inert until propagated.** The models-only widening passes **all 207
  tests and mypy**, because every new field has a default and nothing reads it.
  A domain change that no test can feel is not yet a domain change.
- **It regresses the structural metadata we already ship.** Making it
  load-bearing for **one** of five format paths already fails
  `test_oversized_table_is_split_by_rows_with_header_repeated`, and empties
  `metadata["headings"]` (`['Catalog']` → `{}`) because the two-pass chunker's
  structural pass is document-scoped while blocks are not. **13 test functions
  across 6 files** assert on `metadata["headings"]`, and
  `contextualizing.py` (ticket 07) reads it. The heading path can be rebuilt —
  but at the boundary, with the same heading stack option (a) uses, so it is not
  a reason to prefer (b).
- **It fails silently in the one place that matters.** `fakes.py`'s
  `_chunk_to_dict` / `_chunk_from_dict` (`:103`, `:115`) drop the three new
  fields with no error, so the CLI's ingest → JSON → query round trip erases the
  type. `metadata` has no such failure mode: it is already serialised, on both
  sides, everywhere.

And it buys nothing option (a) does not. Both arms chunk per item; both lose
`_pack`'s merging of small neighbouring units (`chunking.py:238`); both must
rebuild the heading path. The only difference is whether the type lands in a
typed field or in a mapping that is already plumbed end to end.

### (b′) A block type reached by widening the storage schema — rejected

`ragsage_chunks`' column list is fixed by ADR-0002 §1. Promoting `block_type`,
`caption` and `footnote` to real columns reopens that ADR for three fields that
are never a `WHERE` predicate on the hot ANN path. ADR-0002 §2 already draws this
line: `namespace` is a column because it is the isolation mechanism; everything
else is payload in `metadata jsonb`.

## Consequences

**Positive**

- `ingest_items` ships without touching `models.py`, `ports.py`, `fakes.py`, the
  CLI, `examples/`, or any format path. ADR-0001's model and ADR-0002's schema
  both stand unamended.
- Report item #6 (`BlockEnricher`) is **unblocked and needs no domain-model
  ADR of its own.** It gets the selector it needs —
  `metadata["block_type"] == "table"` — plus a caption and footnote kept
  separable rather than inlined, which is what
  `_apply_chunk_template`-style "raw artifact + description" enrichment requires.
  It remains gated on its own retrieval evaluation, not on a model change.
- Unknown item types are supported for free, which a `BlockType` enum would have
  had to special-case.

**Negative / accepted trade-offs**

- **The type is untyped.** `metadata["block_type"]` is `object`, unvalidated and
  invisible to mypy. Mitigation: the seam writes it through one module-level
  constant and one small normalising function, and readers (`BlockEnricher`)
  read it through the same helper — the same defensive-read discipline
  `contextualizing.py:246` already uses for `headings`.
- **Per-item chunking produces more, smaller chunks.** Measured: 3 chunks (a1) vs
  7 (a2) on the same fixture, smallest chunk 125 chars vs 53. A one-line custom
  item becomes its own chunk. Accepted for now; if it hurts retrieval, the fix is
  to pack *within* a block type at the boundary, still no model change.
- **Two ways to know a chunk is a table.** The parsed path detects tables
  heuristically inside the chunker (`chunking.py:_blocks`) and stamps nothing;
  the item path is told and stamps `block_type`. So a chunk from `ingest` has no
  `block_type` while a chunk from `ingest_items` does. Consumers must treat the
  key as absent-by-default. Closing that gap — having the heuristic parser stamp
  `block_type` too — is a follow-up that is *also* pure metadata, and is the
  natural second step for report #6.
- **This ADR is a deferral, not a refutation.** If a future need genuinely
  requires per-block *behaviour* rather than per-chunk *labelling* — retaining
  image bytes for VLM re-entry (report #8), or a block-level enrichment cache —
  reopen it. The condition for revisiting is a reader that needs to dispatch on
  the type, not merely record it.
