# RAG-Anything vs. `ragsage` — what it is, what it does, and what (if anything) we should take

Research report — 2026-07-29.
Subject: [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) (`raganything` on PyPI), investigated
against primary sources only — repo source files, `pyproject.toml`, the PyPI JSON metadata, the GitHub API,
its upstream [LightRAG](https://github.com/HKUDS/LightRAG), and the [arXiv paper](https://arxiv.org/abs/2510.12323).
Compared against our own library at `/Users/niraj/Desktop/Projects/rag-system/ragsage/`.

---

## Bottom line / recommendation

**Do not adopt RAG-Anything, at any level — not as a dependency, not as a vendored subsystem, not as an
optional extra. Steal three ideas from it and leave the rest.**

Three independent reasons, each sufficient on its own:

1. **It cannot physically run on our deployment box.** `raganything`'s *base* (non-optional) dependency set
   is `huggingface_hub`, `lightrag-hku`, `mineru[core]`, `tqdm`
   ([pyproject.toml](https://github.com/HKUDS/RAG-Anything/blob/main/pyproject.toml),
   [PyPI JSON](https://pypi.org/pypi/raganything/json)). `mineru[core]` expands to `mineru[vlm] + mineru[pipeline] + mineru[gradio]`,
   which pulls **`torch>=2.6`, `transformers>=4.57.3`, `onnxruntime>1.17.0`, and `magika>=0.6.2`**
   ([mineru PyPI JSON](https://pypi.org/pypi/mineru/json)). Those are, verbatim and by name, **four of the five
   things ragsage's CI dependency guard exists to forbid** (`ragsage/tests/test_dependency_guard.py:34`,
   `_FORBIDDEN = ("torch", "onnxruntime", "transformers", "magika")`) because the VPS advertises only
   **x86-64-v1** (`ragsage/docs/adr/0001-heuristic-parser-replaces-docling.md`). `pip install raganything` is
   not a heavy install here; it is an install that ends in `Illegal instruction`.
2. **It solves a different problem.** RAG-Anything is a *single-user, single-corpus, filesystem-rooted research
   framework* whose deliverable is a knowledge graph in a `working_dir`
   ([config.py:18](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/config.py)). It contains
   **zero** concepts for tenancy, auth, isolation, per-user scoping, or an async job lifecycle — verified by
   grepping the whole package: the only hits for "namespace" are LightRAG **KV-storage cache partitions**
   (`raganything.py:312`, `namespace="parse_cache"`), not isolation boundaries. Our entire product is the
   isolation boundary (`backend/docs/adr/0001-user-isolation-boundary.md`, Postgres RLS keyed on `app.user_id`).
3. **Its licensing carries a SaaS obligation we'd rather not inherit.** RAG-Anything and LightRAG are both MIT
   ([GitHub API](https://api.github.com/repos/HKUDS/RAG-Anything)), but its mandatory parser MinerU is
   **"Apache-2.0 plus additional terms"** including an explicit **online-service attribution obligation** —
   *"If you provide online services to third parties based on MinerU, you must clearly and prominently indicate…
   that MinerU is used"* — with automatic termination for non-compliance
   ([MinerU LICENSE.md](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)). GitHub cannot classify
   it (`license: NOASSERTION`, [API](https://api.github.com/repos/opendatalab/MinerU)). Not AGPL, but not
   plain Apache-2.0 either, and it is a **mandatory** dependency, not an extra.

**What is genuinely worth taking (ranked in §10):** (a) the **content-list intermediate representation** as a
public insertion seam — a typed, parser-agnostic list of `{type, page_idx, …}` items
([processor.py:2104](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/processor.py)); (b) the
**modal-processor pattern for tables and figures** — enrich a non-prose block with an LLM-written description
*and keep the original*, via a per-type chunk template ([processor.py:1109](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/processor.py));
(c) the **neighbour-window context extractor** as a cheap, deterministic complement to our LLM contextualizer
([modalprocessors.py:55](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/modalprocessors.py)).

**What is emphatically not worth taking:** the knowledge graph, the query-mode taxonomy, MinerU, the mixin
architecture, and the config surface. Reasons in §9.

---

## 1. What RAG-Anything actually is

A **thin multimodal front-end bolted onto LightRAG**. That is not a slight — it is literally the architecture.
`RAGAnything` is a dataclass that inherits three mixins and delegates all storage, retrieval, and generation to
a `LightRAG` instance it holds as a field:

```python
class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin):
    lightrag: Optional[LightRAG] = field(default=None)
```
([raganything.py:51](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/raganything.py))

Its own contribution is: document parsing (MinerU/Docling/PaddleOCR wrappers), turning parsed non-text blocks
into LLM-described text entities, and a VLM re-entry path at query time. Retrieval, the graph, the vector
store, the prompts, the doc-status tracking, and the tokenizer are all LightRAG's.

**Provenance.** Both are from HKUDS (University of Hong Kong Data Science). LightRAG is the EMNLP-2025 paper
project (38,310 stars, [API](https://api.github.com/repos/HKUDS/LightRAG)); RAG-Anything is the multimodal
follow-up, [arXiv:2510.12323](https://arxiv.org/abs/2510.12323), submitted 2025-10-14, by Zirui Guo, Xubin Ren,
Lingrui Xu, Jiahao Zhang, Chao Huang. The paper's claimed contributions are "dual-graph construction to capture
both cross-modal relationships and textual semantics" and "cross-modal hybrid retrieval"; it asserts "superior
performance on challenging multimodal benchmarks" but **the abstract names no benchmark, dataset, or number** —
I could not extract quantitative results from the arXiv abstract page, so treat the performance claim as
unverified here.

**Scale signals** ([GitHub API](https://api.github.com/repos/HKUDS/RAG-Anything), fetched 2026-07-29):
22,466 stars, 2,617 forks, **107 open issues**, MIT, created 2025-06-06, last push 2026-07-20.

---

## 2. Concrete architecture

### 2a. The five-stage pipeline

Per the README, the pipeline is: **document parsing → multi-modal content understanding → multimodal analysis
engine → multimodal knowledge-graph index → modality-aware retrieval**
([README](https://github.com/HKUDS/RAG-Anything/blob/main/README.md)). In code this maps onto:

| Stage | Where it lives | What it does |
|---|---|---|
| Parse | `parser.py` | Subprocess/library call to MinerU, Docling, or PaddleOCR → Markdown + a **`content_list`** JSON |
| Split by modality | `processor.py` | Text items → LightRAG's normal insert; non-text items → the modal processors |
| Describe | `modalprocessors.py` | Per-type LLM/VLM call producing a description + an entity |
| Index | `modalprocessors.py:475` | Write chunk → `text_chunks` + `chunks_vdb`; write entity node → graph + `entities_vdb`; then run LightRAG-style entity/relation extraction over the chunk |
| Retrieve | `query.py` | `lightrag.aquery(...)`, optionally re-entered with images as base64 for a VLM |

### 2b. Parsers ("parser" here means *external document converter*)

Three concrete subclasses of a `Parser` base
([parser.py](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/parser.py)):

- **`MineruParser`** (`parser.py:739`) — the default (`parser: str = "mineru"`, `config.py:29`). Invoked as a
  subprocess (`_run_mineru_command`, `parser.py:833`), output tree read back from disk (`_read_output_files`, `:1079`).
- **`DoclingParser`** (`parser.py:1589`) — runs Docling in-process (`_run_docling_python`, `:1837`), recursively
  walking its document blocks (`read_from_block_recursive`, `:1931`). Docling is **not** a declared dependency;
  it is a soft/optional import.
- **`PaddleOCRParser`** (`parser.py:2186`) — OCR-only, behind the `[paddleocr]` extra.
- Plus a **parser plugin registry**: `register_parser` / `unregister_parser` / `list_parsers` /
  `get_supported_parsers` (`parser.py:2527–2622`), feature-gated in `__init__.py` behind a `try/except ImportError`.
- Office formats are handled by **shelling out to LibreOffice** to convert to PDF first
  (`convert_office_to_pdf`, `parser.py:239`; `_libreoffice_command_candidates`, `:194`). The `office` extra is
  literally `office = []` with the comment "Requires LibreOffice (external program)" (`pyproject.toml`).

### 2c. Modal processors — the actual good idea

Four processors, all subclassing `BaseModalProcessor`
([modalprocessors.py](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/modalprocessors.py)):

- `ImageModalProcessor` (`:836`) — base64-encodes the image (`_encode_image_to_base64`, `:854`) and calls the
  **vision** function.
- `TableModalProcessor` (`:1080`) — text LLM over the table body.
- `EquationModalProcessor` (`:1275`) — text LLM over LaTeX.
- `GenericModalProcessor` (`:1458`) — always registered as the fallback (`raganything.py:238`).

Which are constructed is toggled by `enable_image_processing` / `enable_table_processing` /
`enable_equation_processing` (`raganything.py:204–248`); the generic one is unconditional.

Each does two things worth noting:

1. **`generate_description_only`** — produces a description *plus* an `entity_info` dict
   (`entity_name`, `entity_type`, `summary`).
2. **`_create_entity_and_chunk`** (`:475`) — writes the chunk to `text_chunks_db` **and** `chunks_vdb`, upserts
   an **entity node** into the graph with `source_id = chunk_id`, upserts that entity into `entities_vdb`, then
   runs `_process_chunk_for_extraction` (`:733`) to pull further entities/relations out of the enriched chunk.
   Relations back to the owning document are added in a batch pass,
   `_batch_add_belongs_to_relations_type_aware` (`processor.py:1397`).

3. **`_apply_chunk_template`** (`processor.py:1109`) — the enriched chunk is not just the description. Per type,
   it renders a template (`PROMPTS["image_chunk"]`, `["table_chunk"]`, `["equation_chunk"]`, `["generic_chunk"]`)
   that keeps **`section_path`, `neighbor_text`, `image_path`, `captions`, `footnotes`, and the original
   `table_body`** alongside the `enhanced_caption`. So the embedded text carries both the raw artifact and the
   model's reading of it, with a `try/except` fallback to bare description if the template fails.

### 2d. Context-aware processing

`ContextConfig` + `ContextExtractor` (`modalprocessors.py:40`, `:55`) pull a **window of neighbouring items**
around the one being described, so the VLM describing a figure sees the prose around it. Modes: `page` or
`chunk` (`_extract_page_context` `:139`, `_extract_chunk_context` `:179`), bounded by `max_context_tokens` using
**LightRAG's tokenizer**, truncated at sentence/paragraph boundaries (`_truncate_context`, `:314`).
Documented at [docs/context_aware_processing.md](https://github.com/HKUDS/RAG-Anything/blob/main/docs/context_aware_processing.md).

**This is deterministic and costs nothing** — it is string assembly, not an LLM call. Contrast our
`Contextualizer`, which is an LLM call per chunk (`ragsage/src/ragsage/ports.py:81`).

### 2e. Insert / query APIs

**Insert:**
- `process_document_complete(file_path, …)` — `processor.py:1660`
- `process_document_complete_lightrag_api(…)` — `processor.py:1826`
- `insert_content_list(content_list, file_path, …)` — `processor.py:2104`. **Bypasses parsing entirely.**
  The documented item shapes are:
  `{"type": "text", "text": …, "page_idx": 0}`,
  `{"type": "image", "img_path": <absolute>, "image_caption": [...], "image_footnote": [...], "page_idx": 1}`,
  `{"type": "table", "table_body": <markdown>, "table_caption": [...], "table_footnote": [...], "page_idx": 2}`,
  `{"type": "equation", "latex": …, "text": …, "page_idx": 3}`,
  `{"type": <custom>, "content": …, "page_idx": 4}`.
- **Batch** (`batch.py`): `process_folder_complete` (`:34`), `process_documents_batch` (`:183`),
  `process_documents_batch_async` (`:238`), `filter_supported_files` (`:302`),
  `process_documents_with_rag_batch` (`:321`). Concurrency is `max_concurrent_files`, **default 1**
  (`config.py:56`).

**Query** (`query.py`):
- `aquery(query, mode="mix", system_prompt=None, **kwargs)` (`:102`) — thin pass-through to
  `lightrag.aquery` via `QueryParam`. Notably it **auto-upgrades to VLM-enhanced** whenever
  `vision_model_func` is set unless you pass `vlm_enhanced=False`.
- `aquery_with_multimodal(query, multimodal_content, …)` (`:195`) — attach tables/equations to the question itself.
- `aquery_vlm_enhanced(…)` (`:349`) — the interesting one: it calls LightRAG with
  `QueryParam(mode=mode, only_need_prompt=True)` to get the **assembled retrieval prompt without generation**,
  scans it for image paths, replaces them with base64 (`_process_image_paths_for_vlm`, `:589`, with an
  `extra_safe_dirs` allow-list), rebuilds messages (`_build_vlm_messages_with_images`, `:708`), and calls the
  VLM. If no images are found it **falls back to the normal query**.
- Sync wrappers `query` (`:826`) and `query_with_multimodal` (`:844`).

**Query modes are LightRAG's, not RAG-Anything's:**
`Literal["local", "global", "hybrid", "naive", "mix", "bypass"]`, default `"mix"`
([lightrag/base.py:93](https://github.com/HKUDS/LightRAG/blob/main/lightrag/base.py)), documented in-place as:
local = context-dependent, global = global knowledge, hybrid = both, naive = basic search, mix = KG + vector.

### 2f. Caching and storage

- **Parse cache** and **multimodal-status cache** are LightRAG KV storages in dedicated namespaces
  (`raganything.py:312`, `:326`, `:387`, `:396`); keys via `_generate_cache_key` (`processor.py:50`),
  read/write via `_get_cached_result` / `_store_cached_result` (`:241`, `:320`).
- **Multimodal query cache** keyed by an MD5 of normalized query + content + mode, with image paths reduced to
  basenames and large table bodies hashed (`query.py:26`).
- **Doc status**: its own `DocStatus` enum — `READY / HANDLING / PENDING / PROCESSING / PROCESSED / FAILED`
  ([base.py](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/base.py)) — with
  `_ensure_doc_status_record` / `_upsert_doc_status` (`processor.py:105`, `:138`),
  `is_document_fully_processed` (`:1580`), `get_document_processing_status` (`:1608`).
- **Storage backends are entirely LightRAG's**, four abstractions with these implementations
  ([lightrag/kg/__init__.py](https://github.com/HKUDS/LightRAG/blob/main/lightrag/kg/__init__.py)):
  - **KV**: `JsonKVStorage`, `RedisKVStorage`, `PGKVStorage`, `MongoKVStorage`, `OpenSearchKVStorage`
  - **Graph**: `NetworkXStorage`, `Neo4JStorage`, `PGGraphStorage`, `MongoGraphStorage`, `MemgraphStorage`, `OpenSearchGraphStorage`
  - **Vector**: `NanoVectorDBStorage`, `MilvusVectorDBStorage`, `PGVectorStorage`, `FaissVectorDBStorage`, `QdrantVectorDBStorage`, `MongoVectorDBStorage`, `OpenSearchVectorDBStorage`
  - **Doc-status**: `JsonDocStatusStorage`, `RedisDocStatusStorage`, `PGDocStatusStorage`, `MongoDocStatusStorage`, `OpenSearchDocStatusStorage`

  Selection is by env var per backend (`STORAGE_ENV_REQUIREMENTS`), and defaults are the JSON/NetworkX/
  NanoVectorDB **local-file** implementations.

### 2g. `RAGAnythingConfig` — the full surface

All 17 fields, every one env-var-defaulted via `get_env_value`
([config.py](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/config.py)):

`working_dir` (`./rag_storage`), `parse_method` (`auto`|`ocr`|`txt`), `parser_output_dir` (`./output`),
`parser` (`mineru`|`docling`|`paddleocr`), `display_content_stats`, `enable_image_processing`,
`enable_table_processing`, `enable_equation_processing`, `max_concurrent_files` (1),
`supported_file_extensions` (17 extensions incl. `.xls/.xlsx`), `recursive_folder_processing`,
`context_window` (1), `context_mode` (`page`), `max_context_tokens` (2000), `include_headers`,
`include_captions`, `context_filter_content_types` (`["text"]`), `content_format` (`minerU`), `use_full_path`.

Plus a free-form `lightrag_kwargs: Dict[str, Any]` escape hatch documented as accepting ~25 LightRAG knobs
(`raganything.py:71`). It is a **mutable** dataclass with `update_config(**kwargs)` doing `setattr` on
whatever matches (`raganything.py:249`).

---

## 3. Dependencies, versions, license — the hard facts

| | RAG-Anything |
|---|---|
| Latest release | **1.3.1**, 2026-05-21 ([PyPI JSON](https://pypi.org/pypi/raganything/json)) |
| Release count | 19, first `0.0.1` on 2025-06-09 |
| Base deps | `huggingface_hub`, `lightrag-hku`, `mineru[core]`, `tqdm` |
| Version pin drift | main `pyproject.toml` says **`lightrag-hku<1.5`**; published 1.3.1 metadata says **unpinned `lightrag-hku`**. Current LightRAG is **1.5.4** ([PyPI](https://pypi.org/pypi/lightrag-hku/json)) — so main is pinned *below* the current upstream, and the published wheel will happily resolve to a version main excludes. |
| Python | `pyproject.toml`: **`>=3.10`**; published 1.3.1 metadata: **`>=3.9`**. (`mineru` itself demands `>=3.10,<3.14`.) |
| Extras | `image` (Pillow), `text` (reportlab), `office` (**empty** — external LibreOffice), `paddleocr` (paddleocr, pypdfium2), `markdown` (markdown, weasyprint, pygments), `all` |
| License | MIT — but see the MinerU rider below |
| Status | `Development Status :: 4 - Beta` |

**The transitive reality of `mineru[core]`** ([mineru PyPI JSON](https://pypi.org/pypi/mineru/json)):

```
mineru[core] = mineru[vlm] + mineru[pipeline] + mineru[gradio]
  vlm      → torch>=2.6,<3 ; transformers>=4.57.3,<5 ; accelerate>=1.5.1
  pipeline → torch>=2.6,<3 ; torchvision ; transformers>=4.57.3,<5 ;
             onnxruntime>1.17.0 ; safetensors ; shapely ; pyclipper ; ftfy
  gradio   → gradio>=5.49.1 ; gradio-pdf
mineru (base) also requires: magika>=0.6.2,<1.1.0 ; opencv-python ; numpy>=1.21.6 ; fastapi ; uvicorn
```

Cross-referenced against `ragsage/tests/test_dependency_guard.py:34` — `_FORBIDDEN = ("torch", "onnxruntime",
"transformers", "magika")`, plus `numpy>=2` handled separately at `:22` — **every single forbidden name is
present**, and `magika` is the exact package our router's docstring calls out by name as banned
(`ragsage/src/ragsage/parsing/router.py:9-12`). A base `pip install raganything` also drags in Gradio, FastAPI,
Uvicorn, and OpenCV into a library install.

**Licensing.** `raganything` MIT, `lightrag-hku` MIT, `paddleocr` Apache-2.0. But **MinerU is
Apache-2.0-with-additional-terms** ([LICENSE.md](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)):
a commercial license is required above 100M MAU or USD 20M monthly revenue (irrelevant to us), **and** §2
imposes an unconditional attribution obligation on anyone "providing online services to third parties based on
MinerU", with §3 automatic termination for non-compliance. Also note `[markdown]` pulls **WeasyPrint**, which is
BSD-3 but is a heavyweight rendering stack. No AGPL/GPL exposure was found in the declared graph — the risk here
is a **non-standard Apache rider**, not copyleft.

---

## 4. Honest weaknesses, and maturity signals

**What it does not do at all** (verified by reading the package, not by absence in the README):

- **No multi-tenancy, no auth, no isolation.** A grep for `tenant|multi.tenant|row.level|auth|jwt|rbac` across
  the entire `raganything/` package returns nothing but the author's name and four LightRAG KV-cache
  `namespace=` arguments. One `working_dir`, one graph, one corpus.
- **No async job lifecycle.** `DocStatus` is a status *field* on a record; there is no queue, no worker, no
  retry/claim semantics, no redelivery-idempotency. `process_document_complete` is a long-running coroutine you
  await. Compare `backend/app/ingestion/pipeline.py:161` (`_begin` claims a job only if still `QUEUED`,
  making the worker idempotent under redelivery).
- **No production serving surface.** No HTTP API in this package; it defers to LightRAG's `[api]` extra.
- **No evaluation harness.** There is no eval module, no golden set, no metrics. (LightRAG has an
  `[evaluation]` extra pulling `ragas` — RAG-Anything does not.)
- **No streaming answer surface** in RAG-Anything's own query path; `aquery` returns `str`.
- **No reranking of its own** — it is whatever LightRAG's optional `rerank_model_func` does.
- **Global mutable process state.** Prompt language is set process-wide (`set_prompt_language`,
  `prompt_manager.py`), and `__post_init__` registers an `atexit` hook (`raganything.py:118`) — both hostile in
  a multi-worker server.

**Fragility signals in the code itself.** `BaseModalProcessor` carries a *seven-method ladder* purely for
salvaging malformed LLM JSON: `_robust_json_parse` → `_extract_all_json_candidates` → `_try_parse_json` →
`_basic_json_cleanup` → `_progressive_quote_fix` → `_extract_fields_with_regex` → `_fix_json_escapes`
(`modalprocessors.py:581–732`), plus `_strip_thinking_tags` for reasoning models (`:558`). That is an honest
signal about how reliable "ask the model for structured JSON per figure" is at scale.

**Their own documented failure modes** are worth reading and are creditably candid
([docs/multimodal_rag_failure_modes.md](https://github.com/HKUDS/RAG-Anything/blob/main/docs/multimodal_rag_failure_modes.md)):
OCR/layout silently corrupting text; table structure lost; image↔caption misalignment; **retrieval biased
toward text so image/table chunks never rank**; and "slow vs stuck" on large PDFs.

**Maturity.** 107 open issues; `Development Status :: 4 - Beta`; 19 releases in ~11 months, but the cadence has
thinned markedly (13 releases in Jul 2025 alone, then 1.2.8 Sep-2025 → 1.2.9 Jan-2026 → 1.2.10 Mar-2026 → 1.3.0
May-2026 → 1.3.1 May-2026). **Tests exist and run in CI** — 25 test files and a
[test.yaml](https://github.com/HKUDS/RAG-Anything/blob/main/.github/workflows/test.yaml) matrix over Python
3.10/3.11/3.12 doing `pip install -e ".[all]"` then `pytest tests/ -v`. There is **no coverage gate and no
type-check step**; `mypy` is listed as a dev tool in `pyproject.toml` but the only other workflows are
`linting.yaml` and `pypi-publish.yml`. The package's own `__init__.py` wraps four separate import blocks in
`try/except ImportError` with comments like *"only present in newer versions / when feature PR is merged"* —
i.e. the public API is not stable across installs.

---

## 5. `ragsage` as it stands today (and what's in the backend instead)

**In the library** (`/Users/niraj/Desktop/Projects/rag-system/ragsage/`), ~4,700 lines of `src/`, MIT,
`requires-python >=3.12` (`pyproject.toml`):

- **12 ports**, all `typing.Protocol`, conformance by shape (`src/ragsage/ports.py`): `DocumentParser` (`:48`),
  `PageClassifier` (`:61`), `Chunker` (`:68`), `Contextualizer` (`:81`), `Embedder` (`:98`), `Reranker` (`:105`),
  `LLMClient` (`:114`), `QueryRewriter` (`:130`), `VectorStore` (`:152`), `LexicalStore` (`:170`),
  `DocumentStore` (`:184`), `Cache` (`:204`), plus `Tracer`/`Span` (`:247`, `:217`).
- **Three façades**: `IngestionPipeline.ingest` (`src/ragsage/ingestion.py:91`), `QueryEngine`
  (`src/ragsage/query.py:145`), `Evaluator` (`src/ragsage/evaluation.py:215`).
- **Frozen-dataclass domain models** (`src/ragsage/models.py`) — `RawSource`, `PageImage`, `PageLayout`, `Page`,
  `PageRoute`, `Document`, `ParsedDocument`, `Chunk` (with the `text` / `embed_text` split at `:154`),
  `EmbeddedChunk`, `ScoredChunk`, `Citation`, `Turn`, `Outcome`, `Answer`, and a streamed-event union
  `AnswerEvent` (`:326`).
- **`Scope`** (`src/ragsage/scope.py:19`) — the single tenancy seam: an opaque `namespace` + `filters`. Its
  docstring states the litmus test: a tenancy change must never require editing that file.
- **A real, model-free parser** (`src/ragsage/parsing/`, ADR-0001): `HeuristicBackend` implements *both*
  `DocumentParser` and `Chunker` via an in-flight stash (`parsing/backend.py:62`); a format router using
  media-type → extension → `puremagic` sniff (`parsing/router.py:82`); per-format paths for PDF
  (`parsing/pdf.py:82`, 550 lines: modal-font-size heading detection `:366`/`:385`, column-band reading order
  `:310`, ruled `:427` and borderless `:444` tables to Markdown, page render for the vision route `:538`),
  DOCX (`parsing/docx.py:60`, real `Heading N` styles and numbering levels), PPTX (`parsing/pptx.py:44`,
  including speaker notes `:132`), HTML (`parsing/html.py:71`); a shared two-pass chunker
  (`parsing/chunking.py:76` — heading split then token-bounded packing, tables kept whole via
  `_split_table_rows` `:198`); and `LayoutPageClassifier` (`parsing/classifier.py:20`).
- **Hybrid retrieval with RRF** — `reciprocal_rank_fusion` at `src/ragsage/query.py:73` (K=60), plus
  edge-weighted context arrangement (`:133`), citation binding, an honest `NOT_FOUND` outcome, and a small-talk
  short-circuit (`src/ragsage/smalltalk.py`).
- **A full fake for every port** (`src/ragsage/fakes.py`, 670 lines, `FakeEngineKit` at `:617`) and a **CLI**
  (`src/ragsage/cli.py`) that runs the whole loop offline — deliberately, as "architectural proof" per its
  module docstring.
- **A deterministic, offline evaluator** — faithfulness, answer relevancy, context precision, context recall,
  by content-token overlap, explicitly *not* an LLM judge (`src/ragsage/evaluation.py:1-19`), with a built-in
  golden set (`src/ragsage/goldens.py`) so `ragsage eval` runs in CI with no network.
- **132 tests** across 18 files, including a CI dependency guard that inspects the import graph in a clean
  subprocess (`tests/test_dependency_guard.py`).

**In the backend, not the library** (`/Users/niraj/Desktop/Projects/rag-system/backend/`): the composition root
`build_compute_adapters` (`app/ingestion/pipeline.py:89`); Postgres adapters `PgVectorStore` / `PgLexicalStore` /
`PgDocumentStore` (`app/ingestion/stores.py:96`, `:162`, `:205`) over a `chunks` table with a
`Vector(EMBEDDING_DIM)` HNSW cosine index and a **generated `tsvector` + GIN** index searched with
`websearch_to_tsquery` + `ts_rank` (`app/models.py:209-246`, `app/ingestion/stores.py:189`); the async job
lifecycle `queued → parsing → embedding → ready|failed` (`app/ingestion/pipeline.py:228`); object storage
(`app/ingestion/storage.py`); Taskiq enqueue (`app/ingestion/enqueue.py`); Langfuse tracing
(`app/observability/tracer.py`); and RLS keyed on `app.user_id`.

**Corrections to stale internal notes:** there is **no** `app/gateway/` and no LiteLLM — models are called
directly through LangChain, Voyage for embed/rerank and OpenAI for generate/contextualize/vision
(`backend/docs/adr/0002-direct-provider-access-langchain.md`, `app/providers/`). There is **no** multi-tenancy —
the isolation boundary is the **User** (`backend/docs/adr/0001-user-isolation-boundary.md`). Docling is gone
(`ragsage/docs/adr/0001-heuristic-parser-replaces-docling.md`).

**What exists in neither:** a knowledge graph, entity/relation extraction, multimodal indexing of figures as
first-class retrievable objects, a re-index path, or a cross-process cache (the backend's `Cache` is an
in-process dict, `app/ingestion/pipeline.py:52`).

---

## 6. Feature comparison

| Capability | RAG-Anything | ragsage | Note |
|---|---|---|---|
| **Document parsing** | Has — via MinerU / Docling / PaddleOCR subprocess or lib call ([parser.py](https://github.com/HKUDS/RAG-Anything/blob/main/raganything/parser.py)) | Has — own pure-Python heuristic parser, PDF/DOCX/PPTX/HTML/TXT (`src/ragsage/parsing/`) | Same word, opposite strategy: theirs delegates to ML models, ours *is* the parser |
| **OCR / scanned pages** | Has — MinerU `parse_method="ocr"`, PaddleOCR (`config.py:23`) | Partial — no local OCR tier by design; image-dominated pages route to a cloud VLM (`models.py:111` `PageRoute.VISION`) | Deliberate: ADR-0001 |
| **Office formats** | Partial — shells out to **LibreOffice** → PDF (`parser.py:239`) | Has — native OOXML via `python-docx`/`python-pptx`, real heading styles and table grids | Ours is structurally better here; theirs covers legacy `.doc/.xls`, ours does not |
| **Spreadsheets (.xls/.xlsx)** | Has (`config.py:66`) | **Absent** | Real gap |
| **Equations / LaTeX** | Has — `EquationModalProcessor` + `omml_extractor.py` for Word math | **Absent** | Real gap for technical corpora |
| **Figures/images as retrievable objects** | Has — described by VLM, stored as chunk **and** graph entity (`modalprocessors.py:475`) | **Absent** — images only matter as a page-level route to transcription | The most substantive capability gap |
| **Tables** | Has — described by LLM, template keeps `table_body` + captions + footnotes (`processor.py:1109`) | Partial — extracted to Markdown, kept whole in one chunk (`parsing/chunking.py:198`), never described | Adoptable (§8c-1) |
| **Chunking** | Delegated to LightRAG (token size/overlap, optional split-by-character) | Has — own two-pass structure-aware chunker, heading-path metadata | "Chunk" differs: theirs is a KV record with `tokens`/`full_doc_id`; ours is a frozen dataclass with `page`, `ordinal`, `embed_text` |
| **Context enrichment before embedding** | Has — deterministic **neighbour-window** extractor (`modalprocessors.py:55`), applied to *multimodal* items only | Has — **LLM-written** contextualization for *every* chunk (`ports.py:81`, `ingestion.py:156`) | Different mechanisms; complementary, see §8c-3 |
| **Knowledge graph** | Has — entities + relations + `entities_vdb` + `relationships_vdb` (LightRAG) | **Absent** | Deliberate non-goal, see §9 |
| **Query modes** | Has — `local/global/hybrid/naive/mix/bypass` ([lightrag/base.py:93](https://github.com/HKUDS/LightRAG/blob/main/lightrag/base.py)) | Partial — one mode: dense + lexical fused by RRF (`query.py:73`) | Their taxonomy is graph-vs-vector; ours is dense-vs-lexical. Not comparable |
| **Reranking** | Partial — optional LightRAG `rerank_model_func` | Has — first-class `Reranker` port (`ports.py:105`), `rerank_k` in `QueryOptions` | |
| **VLM-enhanced query** | Has — `only_need_prompt` re-entry with base64 images (`query.py:349`) | **Absent** | Adoptable in principle (§8c-2) |
| **Citations** | **Absent** as a typed contract — `file_path` strings inside chunks | Has — `Citation` model, marker binding, `Outcome.ANSWERED` gating (`models.py:204`, `query.py`) | Our core product guarantee |
| **Honest not-found** | **Absent** | Has — `NOT_FOUND_MESSAGE`, `min_score` floor (`query.py:63`, `config.py:48`) | |
| **Streaming answers** | **Absent** in its own API (`aquery -> str`) | Has — `AnswerEvent` union, `QueryEngine.stream` | |
| **Multi-turn / query rewriting** | Absent | Has — `QueryRewriter` port (`ports.py:130`) | |
| **Direct pre-parsed insertion** | Has — `insert_content_list` (`processor.py:2104`) | **Absent** | Adoptable (§8c-1) |
| **Batch / folder processing** | Has (`batch.py:34`), `max_concurrent_files` default 1 | Partial — CLI ingests a folder sequentially (`cli.py`) | Low value: the backend has a real queue |
| **Async job lifecycle** | **Absent** — status field only, no queue/claim/retry | Out of scope for the library; **backend has it** (`backend/app/ingestion/pipeline.py:228`) | |
| **Multi-tenancy / isolation** | **Absent entirely** | By design: `Scope` seam (`scope.py:19`); enforced by RLS in the backend | The defining difference |
| **Auth** | Absent | Deliberately out of scope (`ragsage/README.md`) | Backend owns it |
| **Storage backends** | Has — 5 KV / 6 graph / 7 vector / 5 doc-status via LightRAG | Has — ports + in-memory fakes; real pgvector/tsvector adapters live in the backend | Theirs ships adapters; ours ships *interfaces* |
| **Caching** | Has — parse cache, multimodal-status cache, multimodal query cache (LightRAG KV) | Partial — `Cache` port used only for contextualization (`ingestion.py:167`); in-process dict in prod | Adoptable (§8c-4) |
| **Observability** | Partial — `CallbackManager` / `MetricsCallback` (`callbacks.py`) | Has — `Tracer`/`Span` ports that may never raise (`ports.py:247`), Langfuse adapter in backend | |
| **Evaluation** | **Absent** | Has — offline deterministic evaluator + golden set (`evaluation.py`, `goldens.py`) | We are ahead |
| **Typed / strict** | Absent — no `py.typed`, mypy not in CI | Has — `py.typed`, `mypy strict` (`pyproject.toml`) | |
| **Runs on x86-64-v1** | **No** — `mineru[core]` → torch/onnxruntime/transformers/magika | **Yes** — enforced by CI guard (`tests/test_dependency_guard.py`) | Decisive |

---

## 7. Architectural differences — and what that implies

**RAG-Anything is a research framework that owns its world.** It is a mutable dataclass with three mixins,
a `working_dir` on local disk, env-var-defaulted config, `atexit` hooks, process-global prompt language, and
soft imports that make the public API vary by install. It is optimised for *one researcher, one corpus, one
machine, maximum capability*. Every model function is a bare `Callable` you pass in — flexible, but with no
contract: nothing declares what `llm_model_func` must accept or return.

**ragsage is a library designed to be consumed.** The dependency arrow points strictly inward
(`ports.py:10-13`): adapters know the engine, the engine never knows adapters. Ports are `Protocol`s so an
adapter need not import ragsage at all. Every domain value is a frozen dataclass. Isolation is one opaque
`Scope`. The CLI and fakes exist as *proof* that no web/DB/tenancy concept leaked in.

Three consequences for "what's worth copying":

1. **Nothing structural transfers.** Their mixins, their config-as-env-defaults, their `working_dir`, their
   global state — all of it is exactly the coupling ragsage was built to avoid. Copying any of it would be a
   regression.
2. **Their *data shapes* transfer well.** The `content_list` item schema and the per-type chunk template are
   plain data, portable across any architecture, and would slot behind our existing ports without changing them.
3. **Their capabilities transfer only if we re-implement them behind our own seams.** Multimodal indexing,
   figure description, KG extraction — each would be a new ragsage port or a new adapter, written by us, with
   `mineru`/`lightrag` nowhere in the import graph. Which is to say: the value here is **as a design reference,
   not as a dependency**.

---

## 8. What ragsage should consider adopting

### 8a. Folder structure / packaging

Our layout is already better factored than theirs (they have a flat `raganything/` with a 2,794-line
`parser.py`, a 2,258-line `processor.py`, and a 1,618-line `modalprocessors.py`; our largest module is
`parsing/pdf.py` at 550). Three narrow ideas, each additive:

1. **Add `src/ragsage/enrichment/` as a sibling of `parsing/`.** Today `parsing/` produces `Page`s and the
   `Contextualizer` port sits alone in `ports.py`. If we take the modal-processor idea (§8c-1), the
   per-block-type enrichers belong in their own package, mirroring `parsing/`'s shape — a small `__init__.py`
   re-export, one module per block type, lazy imports. This keeps `parsing/` about *extraction* and the new
   package about *description*, which is exactly the separation RAG-Anything blurs by putting
   `ContextExtractor` inside `modalprocessors.py`.
2. **Add `docs/` content, not structure.** They ship six topic docs under `docs/` (`batch_processing.md`,
   `context_aware_processing.md`, `multimodal_rag_failure_modes.md`, `offline_setup.md`, `vllm_integration.md`,
   `enhanced_markdown.md`). We have `docs/adr/` and `docs/agents/` and now `docs/research/`. Worth adding: a
   **`docs/failure-modes.md`** in the spirit of theirs — a short, honest "when retrieval looks wrong, check
   these five things" for our heuristic parser (borderless tables, multi-column reading order, VISION-route
   misclassification). Cheap, high user value, and it is the one doc of theirs I'd copy the *format* of.
3. **Add a top-level `examples/` directory.** They have 13 runnable example scripts; we have a CLI and a README
   snippet. Since ragsage is intended for open-sourcing (`README.md`, ADR-0001), two or three runnable
   `examples/*.py` showing (i) fakes end-to-end, (ii) wiring a real embedder behind `Embedder`, (iii) a custom
   `DocumentParser` — would materially lower the adoption bar. **Do not** copy their `[all]`-extras pattern; our
   batteries-included single dependency set is a deliberate ADR-0001 decision.

**Explicitly do not**: split ragsage into extras, adopt mixins, or introduce a `working_dir` concept.

### 8b. Libraries worth evaluating

Filtered hard through (i) the **x86-64-v1** constraint — no `numpy>=2`, `torch`, `onnxruntime`, `transformers`,
`magika` — and (ii) **SaaS licensing**.

| Library | Verdict | Why |
|---|---|---|
| **`mineru` / `mineru[core]`** | ❌ **Blocked** | torch + onnxruntime + transformers + magika ([PyPI](https://pypi.org/pypi/mineru/json)). Also the Apache+rider license with an **online-service attribution obligation** ([LICENSE.md](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md)) |
| **`lightrag-hku`** | ❌ **Reject** | Base deps are individually benign (`networkx`, `numpy<3,>=1.24`, `pandas`, `tiktoken` — [PyPI](https://pypi.org/pypi/lightrag-hku/json)) and it is MIT, so it is *technically* installable. Reject anyway: it is a whole competing engine with its own storage/retrieval/prompt stack. Importing it means importing a second architecture |
| **`paddleocr`** | ❌ **Blocked** | Apache-2.0 (fine) but pulls the Paddle runtime; an OCR tier is explicitly out of scope per ADR-0001 |
| **`weasyprint`** (their `[markdown]` extra) | ❌ **No** | Heavyweight HTML→PDF renderer for a use case we don't have |
| **`json_repair`** (LightRAG dep) | ✅ **Evaluate** | If we ever ask a model for structured JSON (figure/table descriptions, §8c-1), this replaces the seven-method salvage ladder at `modalprocessors.py:581-732` with one small MIT pure-Python dep. Verify `requires_dist` before adopting |
| **`networkx`** | 🟡 Only-if | Pure Python, BSD. Relevant *only* if we ever do a graph experiment. No reason today |
| **`pypdfium2`** | ✅ Already ours | Their `[paddleocr]` extra uses it; we already depend on it (`pyproject.toml`) |

**Net: zero new mandatory dependencies.** The only candidate is `json_repair`, and only conditional on §8c-1.
That is a feature, not a shortfall — ADR-0001's dependency invariant is doing exactly its job.

### 8c. Techniques

**1. Per-type block enrichment + a content-list insertion seam.** *(the strongest idea here)*

- *What it is*: `_apply_chunk_template` (`processor.py:1109`) embeds a table as
  `[caption] + [raw table_body] + [LLM-written description]` rather than either alone; the same for figures,
  with `section_path` and `neighbor_text`. Separately, `insert_content_list` (`processor.py:2104`) exposes a
  typed `{type, page_idx, …}` list as a **public insertion API** that bypasses parsing entirely.
- *What it buys*: (a) tables and figures become retrievable by what they *mean*, not just by the tokens printed
  in them — which directly addresses their own documented failure mode #4 ("retrieval biased toward text");
  (b) a content-list seam lets a caller bring their own parser (or a cloud parser, or a re-index of an existing
  corpus) without touching our `DocumentParser`.
- *What it costs*: one model call per non-prose block at ingest (mitigable by caching on block hash); a wider
  ingest contract; a new failure mode when the description is wrong.
- *Fit with our seams*: **excellent, with a caveat.** The insertion seam is a straightforward new façade method
  on `IngestionPipeline` taking a typed sequence of items — no new port. The enrichment is a new port next to
  `Contextualizer` (call it `BlockEnricher`), which the pipeline invokes between chunk and embed. **The caveat:**
  our `Page`/`Chunk` models are currently *page-of-Markdown* shaped, not *typed-block* shaped
  (`models.py:93`, `:154`). Introducing a block type would be the first real widening of the domain model since
  ADR-0001 and deserves its own ADR. Do not do it casually.

**2. VLM re-entry over assembled context.** `aquery_vlm_enhanced` (`query.py:349`) gets the retrieval prompt
*without generation* (`only_need_prompt=True`), swaps image references for base64, and answers with a VLM.

- *Buys*: answers that actually read the figure, rather than reading someone's description of it.
- *Costs*: VLM tokens on every query that retrieves an image; requires image bytes to still be addressable at
  query time (we don't retain per-page images after ingest — `PageImage` is transient, `models.py:60`); and
  their path-safety `extra_safe_dirs` allow-list (`query.py:589`) is a reminder that "expand a path found in
  retrieved text" is a path-traversal hazard. In a multi-user system that hazard is a cross-user data-leak
  hazard, and would have to be enforced by object-storage key scoping, not a directory allow-list.
- *Fit*: **poor today, good later.** It presumes multimodal indexing (§8c-1) already exists. Revisit only after.

**3. Deterministic neighbour-window context.** `ContextExtractor` (`modalprocessors.py:55`) assembles a bounded
window of surrounding items, truncated at sentence boundaries within a token budget.

- *Buys*: much of contextual retrieval's benefit at **zero model cost and zero latency**, fully deterministic
  (so it is testable in our offline CI, unlike an LLM contextualizer).
- *Costs*: weaker than an LLM-written context sentence; adds tokens to `embed_text`.
- *Fit*: **very good.** It needs no new port at all — it is an alternative `Contextualizer` implementation
  (`ports.py:81` already returns "the text to embed"), so it drops in as `HeadingWindowContextualizer` in
  ragsage itself. And it fills a real hole: today the *only* shipped `Contextualizer` is `FakeContextualizer`
  (`fakes.py:221`); the real one lives in the backend (`OpenAIContextualizer`). A deterministic, dependency-free
  contextualizer would make `IngestionConfig.contextualize=True` meaningful for a standalone CLI user.
  **Highest value-to-effort item in this report.**

**4. Cache what is expensive, keyed by content.** They cache parse output, multimodal-processing state, and
query results in KV namespaces (`raganything.py:312`, `processor.py:241`, `query.py:26`).

- *Buys*: re-ingest and retry become cheap; the parse cache in particular makes re-indexing (changing chunk size,
  changing embedding model) affordable, which is a known hole — `backend/app/models.py:43` notes the embedding
  model is pinned per corpus and changing it is a migration.
- *Costs*: cache invalidation; storage.
- *Fit*: **good, and mostly a backend change.** Our `Cache` port already exists (`ports.py:204`) but is used for
  exactly one thing (`ingestion.py:167`) and is backed by a per-process dict in production
  (`backend/app/ingestion/pipeline.py:52`). Widening its use to parse output, and backing it with Redis, needs
  no library change. The one library-side idea worth taking: **key by content hash, not path** — they normalize
  image paths to basenames precisely so caches survive relocation (`query.py:26`).

**5. Callback/metrics hooks.** `CallbackManager` with `on_query_start` / `on_query_complete` / `on_query_error`
(`query.py:150`, `:180`). **Skip** — our `Tracer`/`Span` ports (`ports.py:217`, `:247`) already cover this with a
stronger contract (never raise, tenant-tagged, nested spans). Theirs is strictly weaker.

**6. Parser plugin registry.** `register_parser(name, cls)` (`parser.py:2527`). **Skip** — our ports are
`Protocol`s and the composition root chooses the adapter (`backend/app/ingestion/pipeline.py:113`). A global
name→class registry is a *worse* version of dependency injection, and it is process-global state.

---

## 9. Explicitly NOT worth adopting

**1. The knowledge graph and entity/relation extraction.** This is the headline feature and we should
consciously decline it. It costs an LLM extraction pass over *every chunk* at ingest, plus entity-merge passes
(`processor.py:1461`), plus two extra vector stores (`entities_vdb`, `relationships_vdb`), plus a graph backend
(Neo4j/Memgraph/PGGraph). In exchange it improves multi-hop and "summarize the whole corpus" questions. Our
product is *"a sage that only speaks from your corpus"* with verbatim citations back to a page — the
`Citation` contract binds a marker to a `chunk_id` + `page` + `quote` (`models.py:204`). A graph-derived answer
is synthesized across entity summaries and is **structurally harder to cite verbatim**. We would be trading our
one differentiating guarantee for a capability our users have not asked for. Revisit only if a user need for
multi-hop retrieval is actually demonstrated.

**2. MinerU.** Blocked on CPU, drags in a Gradio/FastAPI/OpenCV install, and carries a non-standard license
rider with an online-service attribution obligation. Even if the VPS gets host-passthrough (restoring AVX2),
ADR-0001 already reserves that slot for **Docling**, which is plain MIT and better documented.

**3. The query-mode taxonomy (`local`/`global`/`hybrid`/`naive`/`mix`/`bypass`).** These modes only mean
something if you have a knowledge graph — `local` = entity neighbourhood, `global` = community summaries,
`mix` = graph + vector ([lightrag/base.py:94-99](https://github.com/HKUDS/LightRAG/blob/main/lightrag/base.py)).
Without a graph, four of the six collapse to "vector search". Exposing a mode selector we can't back would be
cargo-culting an API. Our single RRF-fused dense+lexical path (`query.py:73`) is the honest surface.

**4. The mixin architecture.** `class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin)` with 2,258 lines of
`ProcessorMixin` reaching into `self.lightrag.text_chunks`, `self.chunks_vdb`, `self.entities_vdb` etc.
(`modalprocessors.py:388-392`) is the opposite of a port. Our façades take their collaborators in `__init__`
(`ingestion.py:64`). Do not regress.

**5. Config-as-env-var-defaults.** Every `RAGAnythingConfig` field defaults from `get_env_value(...)` at *class
definition* time (`config.py:18`), the dataclass is mutable, and `update_config` does blind `setattr`
(`raganything.py:249`). Our `IngestionConfig`/`QueryOptions` are frozen with validating `__post_init__`
(`config.py:28`, `:50`) and env reading lives in the backend's `Settings`. Ours is correct; theirs makes a
library's behaviour depend on ambient process state.

**6. Batch/folder processing in the library.** `max_concurrent_files` defaults to **1** (`config.py:56`), i.e.
sequential. The backend already has Taskiq + a real job lifecycle with idempotent claim
(`backend/app/ingestion/pipeline.py:161`). Adding a second, weaker concurrency mechanism inside the library
would be a liability.

**7. `atexit` hooks and process-global prompt language.** `atexit.register(self.close)` (`raganything.py:118`)
and `set_prompt_language` mutating module-level prompts are actively hostile in a multi-worker server. Non-starters.

**8. LibreOffice-as-a-parser.** Shelling out to a desktop office suite to convert to PDF, then OCR-ing the PDF,
throws away structure we already read natively from OOXML — and their own failure-modes doc admits the DOCX→PDF
path *"can drop inline math"*
([failure modes §3](https://github.com/HKUDS/RAG-Anything/blob/main/docs/multimodal_rag_failure_modes.md)).
Our `python-docx` path is strictly better for `.docx`. (Legacy `.doc`/`.xls` remain unsupported by us — a real
but separate gap, and not one LibreOffice-in-the-request-path should fill.)

---

## 10. Recommendations ranked by value / effort

| # | Recommendation | Value | Effort | Verdict |
|---|---|---|---|---|
| 1 | **Ship a deterministic `HeadingWindowContextualizer` in ragsage** (§8c-3) — neighbour/heading-window `Contextualizer`, no model call, no new dependency, no new port. Fills the "only a fake contextualizer ships" hole and makes the CLI's default config honest | High | Low | **Do it** |
| 2 | **Write `ragsage/docs/failure-modes.md`** (§8a-2) in the style of theirs, for our heuristic parser's known weak spots (borderless tables, multi-column order, VISION misroutes) | Medium | Very low | **Do it** |
| 3 | **Widen the `Cache` port's use to parse output, keyed by content hash; back it with Redis in the backend** (§8c-4). Makes re-index affordable — a known gap given the per-corpus embedding-model pin | High | Medium | **Do it** (mostly backend) |
| 4 | **Add `examples/` to ragsage** (§8a-3) ahead of open-sourcing | Medium | Low | Do it |
| 5 | **Design a typed content-list insertion seam** (`IngestionPipeline.ingest_items`) (§8c-1) — parser-agnostic entry point, enables re-index and BYO-parser | High | Medium-High | **Prototype behind an ADR** |
| 6 | **Table/figure enrichment via a `BlockEnricher` port** (§8c-1) — keep raw + description in `embed_text`; evaluate on the golden set before committing | High | High | Only after #5; needs a domain-model ADR |
| 7 | **Evaluate `json_repair`** if #6 proceeds (§8b) | Low | Very low | Conditional on #6 |
| 8 | **VLM re-entry at query time** (§8c-2) — requires retained page images + object-storage-scoped access | Medium | High | Defer; revisit after #6 |
| 9 | Knowledge graph / entity extraction (§9-1) | — | Very high | **No** |
| 10 | MinerU, LightRAG, query-mode taxonomy, mixins, env-default config, library-side batching (§9) | — | — | **No** |

The honest summary of that table: RAG-Anything's *architecture* has nothing to teach us, and its *dependency
graph* is disqualifying, but its **treatment of non-prose content as first-class retrievable objects** is a real
capability we lack, and items #1, #3, and #5 are the cheap first steps toward it that cost us nothing
architecturally.

---

## 11. Same word, different meaning

| Word | RAG-Anything | ragsage |
|---|---|---|
| **parser** | An external document converter (MinerU/Docling/PaddleOCR), invoked by subprocess or soft import (`parser.py:68`) | A `Protocol` port (`ports.py:48`) whose only shipped implementation is our own pure-Python `HeuristicBackend` |
| **chunk** | A dict in LightRAG KV: `{tokens, content, chunk_order_index, full_doc_id, file_path}` (`modalprocessors.py:490`). A described figure is also a "chunk" | A frozen dataclass with `id`, `document_id`, `text`, **`page`**, `ordinal`, **`embed_text`**, `metadata` (`models.py:154`). Always a passage of prose or a table; never a figure |
| **namespace** | A KV-storage **cache partition** (`raganything.py:312`, `namespace="parse_cache"`) | The **isolation boundary** — `Scope.namespace`, the tenant/user key every store partitions on (`scope.py:33`) |
| **context** | Surrounding document items fed to a VLM *at description time* (`modalprocessors.py:55`) | Two things: the retrieved chunks placed in the prompt (`QueryOptions.context_k`), and the sentence prepended to `embed_text` by `Contextualizer` |
| **adapter** | Not used | An implementation of a port, living outside the library (`ports.py:10-13`) |
| **document status** | `READY/HANDLING/PENDING/PROCESSING/PROCESSED/FAILED` (`base.py`) — a record field | `QUEUED/PARSING/EMBEDDING/READY/FAILED` (`backend/app/models.py:72`) — a job lifecycle with an idempotent claim step |
| **hybrid** | Query mode = graph retrieval + vector retrieval (`lightrag/base.py:97`) | Retrieval = dense vector + lexical BM25/`ts_rank`, fused by RRF (`query.py:73`) |
| **contextualize** | Not a term they use | Anthropic-style contextual retrieval, an LLM call per chunk (`ports.py:81`) |

---

## 12. Claims I could NOT verify from a primary source (flagged)

- **The paper's benchmark results.** [arXiv:2510.12323](https://arxiv.org/abs/2510.12323)'s abstract claims
  "superior performance on challenging multimodal benchmarks" and "significant improvements over
  state-of-the-art methods" but names no benchmark and gives no number on the abstract page. I did not fetch the
  PDF body. **Treat all performance claims as unverified.**
- **Which `lightrag-hku` version actually installs.** Main's `pyproject.toml` pins `lightrag-hku<1.5`; the
  published 1.3.1 metadata on PyPI shows it **unpinned**, and upstream is at 1.5.4. I did not install the
  package to observe the resolved version. The two metadata sources genuinely disagree.
- **The `requires-python` discrepancy.** `pyproject.toml` on main says `>=3.10`; the 1.3.1 PyPI metadata says
  `>=3.9`. Since `mineru` itself requires `>=3.10,<3.14`, the effective floor is 3.10 — but I am inferring that,
  not reading it from a single authoritative field.
- **Whether `raganything` actually fails to import on x86-64-v1.** I did not attempt an install on the VPS. The
  conclusion is derived from declared metadata (`mineru[core]` → torch/onnxruntime/transformers/magika) plus the
  already-established empirical failures documented in
  `backend/docs/research/local-document-parser.md`. It is a very strong inference, not an observation.
- **Docling's exact status as a RAG-Anything parser.** `DoclingParser` exists at `parser.py:1589` and is a valid
  `PARSER` value (`config.py:30`), but Docling is not in any declared extra — so it is an undeclared soft
  dependency the user must install themselves. I did not find documentation stating this explicitly.
- **Test quality.** I confirmed 25 test files and a CI workflow running them on 3.10–3.12, but did not read the
  tests or measure coverage. "Tests exist and run" is verified; "tests are good" is not.
- **`json_repair`'s dependency graph.** Recommended for evaluation in §8b on the basis of it being a LightRAG
  dependency and reputedly pure-Python; I did **not** fetch its PyPI metadata. Verify `requires_dist` before adopting.
- **Star/issue counts** are a single point-in-time read of the GitHub API on 2026-07-29 and will drift.

---

## 13. Sources

**RAG-Anything (primary):**
- Repo & README — https://github.com/HKUDS/RAG-Anything · https://github.com/HKUDS/RAG-Anything/blob/main/README.md
- `pyproject.toml` — https://github.com/HKUDS/RAG-Anything/blob/main/pyproject.toml
- PyPI JSON metadata (versions, dates, `requires_dist`, license, classifiers) — https://pypi.org/pypi/raganything/json
- `raganything/config.py` (RAGAnythingConfig) — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/config.py
- `raganything/raganything.py` (main class, processor init, storage/cache setup) — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/raganything.py
- `raganything/modalprocessors.py` (ContextConfig/ContextExtractor, BaseModalProcessor, Image/Table/Equation/Generic) — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/modalprocessors.py
- `raganything/processor.py` (doc status, caching, chunk templates, `insert_content_list`, entity batching) — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/processor.py
- `raganything/query.py` (`aquery`, `aquery_with_multimodal`, `aquery_vlm_enhanced`, query cache) — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/query.py
- `raganything/parser.py` (Parser base, MineruParser, DoclingParser, PaddleOCRParser, registry, LibreOffice) — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/parser.py
- `raganything/batch.py` — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/batch.py
- `raganything/base.py` (`DocStatus`) — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/base.py
- `raganything/__init__.py` (feature-gated exports, `__version__`) — https://github.com/HKUDS/RAG-Anything/blob/main/raganything/__init__.py
- `docs/context_aware_processing.md` — https://github.com/HKUDS/RAG-Anything/blob/main/docs/context_aware_processing.md
- `docs/multimodal_rag_failure_modes.md` — https://github.com/HKUDS/RAG-Anything/blob/main/docs/multimodal_rag_failure_modes.md
- CI test workflow — https://github.com/HKUDS/RAG-Anything/blob/main/.github/workflows/test.yaml
- Repo metadata (stars, issues, license, dates) — https://api.github.com/repos/HKUDS/RAG-Anything
- Paper — https://arxiv.org/abs/2510.12323

**Upstream / transitive (primary):**
- LightRAG repo & metadata — https://github.com/HKUDS/LightRAG · https://api.github.com/repos/HKUDS/LightRAG
- LightRAG `QueryParam` modes, storage ABCs — https://github.com/HKUDS/LightRAG/blob/main/lightrag/base.py
- LightRAG storage implementations & env requirements — https://github.com/HKUDS/LightRAG/blob/main/lightrag/kg/__init__.py
- `lightrag-hku` PyPI metadata — https://pypi.org/pypi/lightrag-hku/json
- `mineru` PyPI metadata (the `[core]` expansion) — https://pypi.org/pypi/mineru/json
- MinerU license (Apache-2.0 + additional terms) — https://github.com/opendatalab/MinerU/blob/master/LICENSE.md · https://api.github.com/repos/opendatalab/MinerU
- `paddleocr` PyPI metadata — https://pypi.org/pypi/paddleocr/json

**ragsage / backend (local):**
- `ragsage/README.md`, `ragsage/pyproject.toml`, `ragsage/CLAUDE.md`
- `ragsage/docs/adr/0001-heuristic-parser-replaces-docling.md`
- `ragsage/src/ragsage/ports.py`, `models.py`, `scope.py`, `config.py`, `ingestion.py`, `query.py`, `evaluation.py`, `goldens.py`, `fakes.py`, `cli.py`, `smalltalk.py`
- `ragsage/src/ragsage/parsing/` — `backend.py`, `router.py`, `chunking.py`, `classifier.py`, `pdf.py`, `docx.py`, `pptx.py`, `html.py`, `identity.py`
- `ragsage/tests/test_dependency_guard.py` (the CPU-compatibility invariant)
- `backend/CONTEXT.md`, `backend/docs/adr/0001-user-isolation-boundary.md`, `0002-direct-provider-access-langchain.md`, `0003-usage-observability-langfuse.md`
- `backend/app/ingestion/pipeline.py` (composition root, job lifecycle), `stores.py` (pgvector + tsvector), `app/models.py` (chunks table, HNSW + GIN)
- `backend/docs/research/local-document-parser.md` (the x86-64-v1 constraint and library matrix)
