<div align="center">

# ragsage

**A sage that only speaks from your corpus.**

A production-ready Retrieval-Augmented Generation engine for Python — parsing, chunking,
hybrid retrieval, reranking, grounded generation and verifiable citations — behind
swappable ports.

[![PyPI](https://img.shields.io/pypi/v/ragsage.svg)](https://pypi.org/project/ragsage/)
[![Python](https://img.shields.io/pypi/pyversions/ragsage.svg)](https://pypi.org/project/ragsage/)
[![CI](https://github.com/nirajk77777/ragsage/actions/workflows/ci.yml/badge.svg)](https://github.com/nirajk77777/ragsage/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/ragsage/badge/?version=latest)](https://ragsage.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nirajk77777/ragsage/blob/main/LICENSE)
[![Typed](https://img.shields.io/badge/typing-strict-informational.svg)](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/py.typed)

</div>

---

`ragsage` answers questions from **your** documents, and says so honestly when it can't.
Every answer is either grounded prose with citations you can resolve back to a chunk and a
page, or an explicit *not found* — the engine never invents a confident answer out of thin
retrieval.

```bash
pip install ragsage
```

📖 **[Documentation](https://ragsage.readthedocs.io/)** — quickstart, full API reference,
and the failure modes of the heuristic parser.

## 60 seconds, no API key

Every one of the engine's thirteen ports ships with a working in-memory implementation, so
the full ingest-and-query loop runs **offline** — no database, no server, no credentials:

```console
$ ragsage ingest ./docs
  + france.txt: 1 chunk
corpus saved to .ragsage/state.json

$ ragsage query "What is the capital of France?"
Paris is the capital of France. It sits on the Seine river. [1]

Sources:
  [1] 36ad5ede1a879836 (page 1)

$ ragsage query "Who won the 1998 World Cup final?"
I couldn't find an answer to that in your documents.
```

That second answer is the point. The corpus can't support it, so the engine says so.

## The production path

One config object, a Postgres with the `vector` extension, and two API keys:

```python
import asyncio
import os

from ragsage import RawSource, Scope
from ragsage.providers import ProviderConfig
from ragsage.sage import RagSage, RagSageConfig
from ragsage.storage import PostgresConfig


async def main() -> None:
    sage = RagSage.from_config(
        RagSageConfig(
            postgres=PostgresConfig(dsn=os.environ["DATABASE_URL"]),
            providers=ProviderConfig(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                voyage_api_key=os.environ["VOYAGE_API_KEY"],
            ),
        )
    )
    await sage.migrate()  # idempotent: safe on every boot

    scope = Scope(namespace="acme-corp")
    await sage.ingest(RawSource(name="handbook.pdf", path="./handbook.pdf"), scope)

    answer = await sage.query("How much parental leave do we get?", scope)
    print(answer.text, answer.outcome, answer.grounded)
    for citation in answer.citations:
        print(f"[{citation.marker}] {citation.document_id} page {citation.page}")

    await sage.dispose()


asyncio.run(main())
```

`RagSage` also exposes `stream()` (tokens → citations → usage → complete, ready to relay
over SSE), `delete_document()` and `purge()`.

<details>
<summary><b>Streaming</b></summary>

```python
from ragsage import AnswerComplete, AnswerToken, Citation, Usage

async for event in sage.stream("How much parental leave do we get?", scope):
    match event:
        case AnswerToken(text=text):
            print(text, end="", flush=True)
        case Citation() as citation:
            print(f"\n[{citation.marker}] {citation.document_id} p{citation.page}")
        case Usage(sources=sources):
            print(f"\n{sources} sources in context")
        case AnswerComplete() as done:
            print(f"\ndone: {done.outcome}")
```

The scoped database session is held open for the whole stream, and released even if the
consumer abandons it early.

</details>

## Why ragsage

- **Honest by construction.** Three outcomes — `answered`, `not_found`, `conversational` —
  and `Answer.grounded` is derived from the outcome, so the two can never disagree. A
  greeting is never answered with "that isn't in your documents".
- **Citations that resolve.** Each context chunk carries a stable marker; the returned
  citations are exactly the markers the model used, bound back to chunk, document and page.
- **Nothing is welded in.** Thirteen `typing.Protocol` ports. Your adapter satisfies one by
  *shape* — it imports and subclasses nothing from ragsage.
- **Runnable the moment it's installed.** Real in-memory adapters for every port, plus a CLI
  that drives the whole loop with no network. The offline test suite never touches a
  provider.
- **Portable parsing.** The parser's import graph carries **no torch, no numpy ≥ 2, no
  transformers, no onnxruntime, no magika** — a deliberate promise (x86-64-v1 hosts
  included), enforced by a test that probes the graph in a clean subprocess rather than by
  convention.
- **Tenancy-agnostic.** The library never imports web, auth or tenant concepts. Callers pass
  an opaque [`Scope`](https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/scope.py)
  — a namespace plus optional metadata filters — which the engine treats as an untyped
  label. One boundary; the same engine runs from a script or behind a multi-tenant SaaS,
  unchanged.
- **Strictly typed.** `mypy --strict` over the library *and* its examples, in CI.

## How it works

```
ingest   RawSource ─▶ parse ─▶ classify ─▶ chunk ─▶ contextualize ─▶ embed ─┬▶ VectorStore
                                                                           ├▶ LexicalStore
                                                                           └▶ DocumentStore

query    question ─▶ rewrite ─▶ embed ─┬▶ dense ───┐
                                       └▶ lexical ─┴▶ RRF fuse ─▶ rerank ─▶ min_score gate
                                                                                  │
                     Answer + Citations ◀─ LLM ◀─ bounded context ◀───────────────┘
```

Dense and lexical hits combine by Reciprocal Rank Fusion (k=60), so a chunk strong in either
channel surfaces. A cross-encoder reranks the fused candidates, and only the top few reach
the model's context. Retrieval below `min_score` is treated as empty — which is what turns a
weak match into an honest *not found* instead of a hallucination.

### The ports

| Port | Responsibility | Shipped implementations |
|---|---|---|
| `DocumentParser` | bytes → pages | `HeuristicBackend` (PDF, DOCX, PPTX, HTML, Markdown, text) |
| `PageClassifier` | route a page: text or vision | `LayoutPageClassifier` |
| `Chunker` | pages → chunks | `HeuristicBackend` |
| `Contextualizer` | situate a chunk before embedding | `HeadingWindowContextualizer` (model-free), `OpenAIContextualizer` |
| `Embedder` | texts → vectors | `VoyageEmbedder` |
| `Reranker` | reorder candidates | `VoyageReranker` |
| `LLMClient` | generate, transcribe | `OpenAILLMClient` |
| `QueryRewriter` | resolve a follow-up against history | `OpenAIQueryRewriter` |
| `VectorStore` | dense search | `PgVectorStore` — pgvector, HNSW cosine |
| `LexicalStore` | keyword search | `PgLexicalStore` — generated `tsvector` + GIN |
| `DocumentStore` | document rows, dedup by content hash | `PgDocumentStore` |
| `Cache` | optional memoization | no-op default |
| `Tracer` | spans and events | `NullTracer`; bring your own |

Every one also has an in-memory fake in `ragsage.fakes`. The Postgres stores are
Row-Level-Security-scoped, and `migrate()` reads the live column width and refuses to run
onto a table of a different embedding dimension — before any DDL, not after the first insert
fails.

### Swapping one out

The ports are `Protocol`s, so the dependency arrow only ever points inward:

```python
from collections.abc import Sequence
from dataclasses import replace

from ragsage.sage import ComputeKit, RagSage


# Subclasses nothing, registers nowhere: it satisfies the port by shape alone.
class MyEmbedder:
    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]: ...


compute = replace(ComputeKit.from_config(config), embedder=MyEmbedder())
sage = RagSage.from_config(config, compute=compute)
```

`stores=` swaps persistence the same way, and `database=` shares a connection pool with your
own app. Outgrown the assembler entirely? `IngestionPipeline` and `QueryEngine` stay directly
constructible, and `RagSage.pipeline_for()` / `engine_for()` are public so you reach past the
façade instead of forking it.

## Configuration

The library **never reads `os.environ`** — every key, DSN and model id arrives in a frozen
dataclass constructed by you, at your own composition root. A config whose values depend on
the ambient process is untestable, un-injectable and silently different in production.

<details>
<summary><b>The knobs, and their defaults</b></summary>

**`PostgresConfig`** — where ragsage's tables live.

| Field | Default | Notes |
|---|---|---|
| `dsn` | *required* | a bare `postgresql://…` is normalised to `postgresql+asyncpg://` |
| `app_role` | `ragsage_app` | non-privileged role scoped work runs under, so RLS is enforced |
| `isolation_variable` | `ragsage.namespace` | the setting the RLS policy compares rows against |
| `embedding_dim` | `1024` | matches `voyage-3-large`; pinned per corpus, max 2000 |
| `text_search_config` | `english` | the generated `tsvector` column's configuration |
| `pool_size` / `max_overflow` | `5` / `5` | ragsage owns a *second* pool; both must fit `max_connections` |

**`ProviderConfig`** — per-role models, priced and shaped differently on purpose.

| Field | Default |
|---|---|
| `embedding_model` | `voyage-3-large` |
| `rerank_model` | `rerank-2` |
| `generation_model`, `vision_model` | `gpt-4o` |
| `contextualize_model` | `gpt-4o-mini` |
| `timeout_seconds` | `60.0` |

**`IngestionConfig`** — `chunk_size=512`, `chunk_overlap=64`, `contextualize=True`.

**`QueryOptions`** — `retrieve_k=20`, `rerank_k=5`, `context_k=3`, `min_score=0.0`.

The last two carry policy defaults on `RagSageConfig` and stay overridable per call.

</details>

## The CLI

```console
$ ragsage info                     # version and default scope
$ ragsage ingest ./docs            # --namespace, --no-contextualize, --contextualizer
$ ragsage query "…"                # answer from the stored corpus
$ ragsage eval                     # score the built-in golden set
```

It wires nothing but ports and fakes, which makes it the standing proof that the engine is
self-contained: swapping the fakes for real adapters is the only difference between this and
a production deployment.

## Evaluation

`Evaluator` scores a dataset on the standard RAG metrics — faithfulness, answer relevancy,
context precision and context recall — against `EvalThresholds`, so a retrieval regression
fails a build instead of surfacing in production. `ragsage eval` runs it over the built-in
golden set:

```console
$ ragsage eval
ragsage offline eval — golden set
  examples:          5
  answer accuracy:   1.000
  faithfulness:      1.000
  answer relevancy:  1.000
  context precision: 1.000
  context recall:    1.000
  grounded rate:     0.800
PASS — all metrics met their thresholds
```

## Examples

Four scripts, argument-free, type-checked *and executed* in CI:

| Script | What it shows |
|---|---|
| [`fakes_end_to_end.py`](https://github.com/nirajk77777/ragsage/blob/main/examples/fakes_end_to_end.py) | The whole loop against the fakes, including the honest not-found path — about ten lines of wiring. |
| [`custom_embedder.py`](https://github.com/nirajk77777/ragsage/blob/main/examples/custom_embedder.py) | Implementing `Embedder` against something that isn't Voyage. |
| [`custom_parser.py`](https://github.com/nirajk77777/ragsage/blob/main/examples/custom_parser.py) | Implementing `DocumentParser` for a format the built-in backend doesn't understand. |
| [`assembled_engine.py`](https://github.com/nirajk77777/ragsage/blob/main/examples/assembled_engine.py) | `RagSage.from_config(...)` — migrate, ingest, query, purge. Wants a Postgres; skips with a message without one. |

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nirajk77777/ragsage.git
cd ragsage
uv sync
uv run pytest
```

Tests are offline by default — no database, no API keys, no network. The gates CI runs, in
order:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src examples && uv run pytest
```

Issues and pull requests are welcome. [**CONTRIBUTING.md**](CONTRIBUTING.md) has the rest:
the opt-in Postgres and live-provider suites, the import-graph invariants a change must not
break, and the house rules that CI enforces. Found a security problem? Don't open an issue —
see [SECURITY.md](SECURITY.md).

## Releasing

Tag-driven and tokenless: `git tag v0.1.0 && git push origin v0.1.0` re-runs every gate,
builds, and publishes to PyPI via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API token exists
anywhere, and the job holding the OIDC credential runs no project code. See
[`docs/releasing.md`](https://github.com/nirajk77777/ragsage/blob/main/docs/releasing.md).

## License

[MIT](https://github.com/nirajk77777/ragsage/blob/main/LICENSE) — © 2026 Niraj Kumar.
