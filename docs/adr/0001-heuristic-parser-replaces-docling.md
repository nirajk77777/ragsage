# 0001 — Heuristic parser replaces Docling

- Status: Accepted
- Date: 2026-07-24

## Context

`ragsage`'s ingestion backend needs to turn raw document bytes (PDF, DOCX, PPTX,
HTML, plain text/Markdown) into structure-aware `Page`s and `Chunk`s. Until now
that job was done by **Docling** (layout + TableFormer models), wired in the
consuming backend behind ragsage's `DocumentParser` / `Chunker` ports.

Docling requires a modern CPU. Its NumPy 2.x / PyTorch / ONNX Runtime stack is
compiled for the **x86-64-v2** instruction baseline (SSE4.2/AVX2). The deployment
VPS exposes only **x86-64-v1** (an old QEMU CPU model): NumPy 2.x *refuses to
import* and PyTorch/Docling never load, so document ingestion is **completely
blocked** on the target box. Fixing the VM's CPU model is a hosting-provider
ticket outside our control, and even once fixed we would rather not carry a heavy
ML parser and its multi-gigabyte models for what is mostly born-digital PDF and
Office content.

Historically ragsage's *core* was deliberately dependency-free — the whole engine
lived behind ports so it pulled no provider SDK and no database driver. That
zero-dependency core is incompatible with owning a real parser.

Full problem analysis, the library CPU-compatibility matrix, and the structure
heuristics are captured in the spec (`.scratch/heuristic-parser/spec.md`) and the
reference research (`backend/docs/research/local-document-parser.md`).

## Decision

**Ship a pure-Python, model-free heuristic parser inside ragsage as the ingestion
backend, and remove Docling.**

- A single `ragsage.parsing.HeuristicBackend` implements *both* the
  `DocumentParser` and `Chunker` ports by shape (same in-flight-stash pattern the
  Docling adapter used). A format router (`media_type` → extension → pure-Python
  content sniff via `puremagic`, **never `magika`**) selects a per-format path;
  each path serialises to Markdown, which a shared two-pass, structure-aware
  chunker (heading split + token bounding, tables kept whole) turns into chunks.
- Structure without ML: PDF headings from modal body font size + font/bold +
  numbered/short-line heuristics, reading order with column-gutter detection,
  ruled + borderless tables to Markdown (`pdfplumber` / `pypdfium2`); Office/HTML
  honour native structure — real heading styles, list levels, table grids,
  `<h1>…<h6>` (`python-docx`, `python-pptx`, `beautifulsoup4`/`lxml`).
- **The library stack becomes ragsage core `dependencies`.** ragsage's previously
  zero-dependency core is **retired** in favour of a batteries-included MIT/BSD
  parser stack (`pdfplumber`, `pypdfium2`, `python-docx`, `python-pptx`,
  `beautifulsoup4`, `lxml`, `langchain-text-splitters`, `tiktoken`, `puremagic`).
  All are permissively licensed (no AGPL/copyleft exposure) and pure-Python /
  pure-Rust / pure-C.
- **Hard dependency constraint (CPU-compatibility invariant).** The parser's
  transitive import graph must not include `numpy>=2`, `torch`, `onnxruntime`,
  `transformers`, or `magika`. This is an acceptance criterion, not a preference,
  and is locked in CI by a guard test that inspects the import graph in a clean
  subprocess. Token counting uses **`tiktoken`** (pure Rust/Python), deliberately
  not `transformers` — installing `transformers` upgrades NumPy to 2.x and
  re-breaks the box. (The spec anticipated the HuggingFace `tokenizers` library as
  the lightweight tokenizer; `tiktoken`, which the spec accepts as an alternative,
  is what shipped.)
- Page routing is unchanged. The backend measures the same `PageLayout` signals
  (`text_chars`, `image_area_ratio`, `bitmap_area`) per page and attaches a
  rendered `PageImage` for text-thin / image-dominated pages; the parser-neutral
  `LayoutPageClassifier` (three tunable thresholds) routes them, and image-only
  pages continue to the premium **VISION** model (`mistral-ocr` via
  `RAG_VISION_MODEL`). There is no local OCR tier.
- Docling is removed, not kept behind a flag. The consuming backend's composition
  root (`build_compute_adapters`) wires the heuristic backend as the sole
  `parser`/`chunker`; the Docling adapter module, its `DOCLING_ARTIFACTS_PATH`
  model-artifacts wiring, the model-download step, and the `transformers` tokenizer
  prefetch are all removed from the worker image.

## Consequences

**Positive**

- Ingestion runs on the x86-64-v1 VPS — uploads process instead of failing at
  parser import. The whole dependency set pulls zero numpy≥2 / torch /
  onnxruntime / transformers.
- The worker image is much smaller and cheaper: no Docling install, no
  multi-gigabyte models, no torch.
- Because the libraries are light, the *real* parsing runs in CI. Unlike the
  Docling adapter (parsing was integration-only; only the classifier was
  unit-tested), heading detection, table-kept-whole, multi-page page attribution,
  `PageLayout` measurement, scan-page image rendering, DOCX/PPTX/HTML structure,
  and the two-pass chunker are all unit-tested with hermetic fixtures.
- Document identity (sha256, 16-char id prefix), the ports, and everything
  downstream are unchanged — the swap is invisible past the parse/chunk seam.

**Negative / accepted trade-offs**

- **Retrieval-quality trade-off.** The heuristic parser does not match Docling's
  ML layout/TableFormer quality on *messy* inputs: multi-column scans, borderless
  or ragged tables, and complex layouts. Some degradation on those is **accepted**.
  The **pressure valve** is the unchanged VISION route — a page the heuristics
  can't read (text-thin, image-dominated) and a hard/unparseable page both degrade
  to `mistral-ocr` rather than failing the document.
- ragsage's core is no longer dependency-free; consumers now inherit the parser
  stack transitively. This is a deliberate reframing of the library as a
  batteries-included, open-source parser — not an optional extra.

**Reserved for the future**

- **Docling is reserved for capable hosts** and future reconsideration once the
  CPU model is fixed (e.g. the VPS moves off x86-64-v1, or on a modern host).
  Docling's design is not lost: the spec (`.scratch/heuristic-parser/spec.md`) and
  the research doc (`backend/docs/research/local-document-parser.md`) capture it if
  it is ever restored. Restoring it would reopen this ADR.
- Out of scope and unaffected: legacy binary Office formats (.doc/.ppt/.xls), a
  local OCR tier, cloud parsing services, and re-indexing already-ingested
  corpora (this changes the parser going forward only).
