# Prototype — typed content-list insertion seam (ticket 11)

> **THROWAWAY. NOT PRODUCTION CODE. NOTHING HERE MERGES TO `main`.**
> This directory exists only to answer one design question with evidence. It is
> not on any import path the library uses, has no tests in `tests/`, and its
> `arm-b.patch` deliberately modifies production files — apply it only to measure,
> never to keep. The answer it produced lives in the ticket
> (`.scratch/ragsage-owns-storage/issues/11-content-list-insertion-seam.md`) and,
> as a decision, in `docs/adr/0004-typed-blocks-ride-in-chunk-metadata.md`.

## The question

Should a public, parser-agnostic `IngestionPipeline.ingest_items(items, scope)` —
taking an already-parsed typed sequence (`{type, page_idx, …}` for text, image,
table, equation, custom) and bypassing parsing entirely —

- **(a)** collapse the typed items into Markdown at the boundary, which is cheap
  but throws away the type information the seam exists to carry; or
- **(b)** introduce a block type into the domain model, the first real widening
  since ADR-0001?

## What is here

| File | What it is |
|---|---|
| `items.py` | The typed item shapes, mirroring RAG-Anything's `insert_content_list` (`processor.py:2104`), plus the 7-item fixture both arms ingest (text ×3, table, image, equation, one custom type) |
| `arm_a.py` | Arm (a) in two variants — **a1** page collapse, **a2** item-attributed collapse — over the real chunker and the real pipeline collaborators |
| `run_experiment.py` | Runs both variants end to end (chunk → contextualise → embed → store → retrieve → cite) and prints the evidence |
| `arm-a-measured.txt` | Recorded output of the above |
| `arm-b.patch` | Arm (b) itself: the domain-model widening, as a patch against `src/` (130 insertions across 3 files). **Not applied.** |
| `arm_b_probe.py` | Measures what (b) breaks. Requires the patch applied |
| `arm-b-measured.txt` | Recorded output of the probe, plus the test/mypy result under (b) |

`arm_a.py` splits arm (a) in two because "collapse into Markdown at the boundary"
turns out to name two different decisions:

- **a1 — page collapse.** Join every item on a page into one Markdown blob, build
  a `Page`, hand it to the chunker exactly as a parser would. This is the literal
  reading, and it is the arm that really does throw the type away.
- **a2 — item-attributed collapse.** Still collapse to Markdown (the domain model
  is untouched), but chunk *item by item* and stamp the item's type, caption and
  footnote onto `Chunk.metadata`, which is already a free-form
  `Mapping[str, object]` (`models.py:171`) and is already persisted as
  `metadata jsonb` (ADR-0002 §1).

## Reproducing

```bash
# Arm (a), both variants — safe, touches nothing in src/
uv run python prototypes/11-content-list/run_experiment.py

# Arm (b) — modifies production code; revert when done
git apply prototypes/11-content-list/arm-b.patch
uv run python prototypes/11-content-list/arm_b_probe.py
uv run pytest -q
git checkout -- src/
```

Baseline for comparison: `207 passed`, `mypy src examples` clean.

## Headline results

- **a1 loses the type completely.** Of 3 chunks produced from 7 items, only 1 is
  a single item; one chunk fuses text + image + equation + custom. The only
  metadata key that exists is `headings`. The chunk holding the table is
  **mixed prose + table** (5 of 8 non-blank lines are table rows), so even
  re-sniffing the Markdown cannot recover "this chunk *is* a table".
- **a2 keeps the type, with no model change.** All 7 items become 7 chunks,
  selectable by `metadata["block_type"]` (`{text: 3, table: 1, image: 1,
  equation: 1, checklist: 1}`), with caption and footnote kept beside the body
  rather than inlined into it — and it survives a JSON snapshot → restore →
  search round trip because the fakes already serialise `metadata`.
- **(b) is inert until it is propagated, and breaks things once it is.** The
  models-only widening passes all 207 tests and mypy — because every new field
  has a default and nothing reads it. Making it load-bearing for *one* of five
  format paths already fails a test, empties `metadata["headings"]`, and is
  silently dropped by the fakes' serialisation.

Both arms hit one thing first: the shipped `Chunker` cannot be called at all
without a parse. `HeuristicBackend.chunk` pops an in-flight stash keyed by
document id (`backend.py:128`), so a seam that bypasses parsing gets
`RuntimeError: no parsed document stashed for …`. That is a cost of
`ingest_items` under either arm, not of either arm.
