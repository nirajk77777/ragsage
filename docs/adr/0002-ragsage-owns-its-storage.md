# 0002 — ragsage owns its storage and its model adapters

- Status: Accepted
- Date: 2026-07-29

## Context

Until now `ragsage` shipped *interfaces*: twelve `Protocol` ports, in-memory fakes
for each, and no adapter that talks to a real database or a real model provider.
Every production adapter lived in the consuming backend —
`app/ingestion/stores.py` (pgvector + tsvector), `app/providers/` (Voyage,
OpenAI), and four composition-root functions that assembled them.

That split cost the backend ~750 lines of code whose subject matter is entirely
RAG: SQL for a vector table the library defines the shape of, and provider
adapters for ports the library declares. Every consumer of the open-source
library would have to write the same 750 lines before it did anything. The
library was reusable in principle and unusable in practice.

Two facts made this the right moment to change it:

**The x86-64-v1 constraint is retired.** ADR-0001 assumed the deployment VPS
advertised only x86-64-v1, which made numpy≥2 (and therefore `voyageai`)
unshippable. Re-verified on the box on 2026-07-29: it now reports **x86-64-v4**
(Intel Xeon Gold 6240, AVX2/FMA present) and `numpy==2.5.1` imports and runs
correctly. The provider appears to have switched the VM to host-passthrough. The
dependency invariant survives as a deliberate portability promise, not a hardware
necessity — see the amendment on ADR-0001.

**Nothing is deployed yet.** No backend, worker, or web container runs on the
VPS; only Postgres, Redis, Garage, and Langfuse. There is no production corpus,
so the schema change is greenfield rather than a migration.

## Decision

`ragsage` owns its storage end to end and ships its model adapters in core.

1. **One table, owned by the library.** `ragsage_chunks` — id, namespace,
   document id, chunk ref, text, embed_text, page, ordinal, `vector(dim)` with an
   HNSW cosine index, a generated `tsvector` with a GIN index, the pinned
   `embedding_model`, and a `metadata jsonb` column persisting `Chunk.metadata`.
   Primary key `(namespace, chunk_ref)`. The library owns nothing else — document
   identity, job lifecycle, conversations, and users stay the consumer's.

2. **`namespace` is a real indexed column, not metadata.** It stores
   `Scope.namespace` verbatim as `TEXT`, so `namespace="local"` from the CLI
   still works and `scope.py`'s litmus test stays true. `metadata jsonb` is
   *payload* the consumer can attach and filter on — never the isolation
   mechanism. (LightRAG lands on the same shape independently: `workspace
   VARCHAR(255)` inside `PRIMARY KEY (workspace, id)`, with JSONB kept strictly
   as payload and absent entirely from the table it runs ANN against.)

3. **ragsage owns the connection pool and enforces isolation itself.** Given one
   DSN it connects as owner and, per operation, runs `SET LOCAL ROLE` into a
   non-privileged role and binds its isolation variable from `Scope.namespace` —
   the same posture as the backend's `open_user_session`, moved inward. Row-Level
   Security stays the load-bearing guarantee; it is not downgraded to an
   application-side `WHERE` clause.

4. **ragsage owns its DDL.** `await sage.migrate()` creates the table, its
   indexes, the app role, and the RLS policy, idempotently. The consumer's own
   bootstrap stops managing chunks.

5. **Postgres, Voyage and OpenAI adapters are core dependencies**, not extras.
   `pip install ragsage` gives a working engine.

6. **The ports stay public and the config stays explicit.** `RagSage` is a
   convenience assembler *over* the twelve ports, not a replacement for them: the
   fakes, the CLI, and the Seam-2 harness keep working, and a consumer can still
   inject a custom `Embedder` or `VectorStore`. Configuration is a frozen
   dataclass passed by the caller — never environment variables read at
   class-definition time.

## Considered options

- **Optional adapters behind extras** (`ragsage[postgres,openai]`) with the
  backend keeping the pool and the session — rejected as not going far enough:
  the consumer still hand-assembles seven ports and still owns a session before
  it can call `ingest`.
- **Isolation as a configured metadata key**, with the RLS policy reading
  `metadata->>'user_id'` — rejected. It needs an expression index, makes every
  ANN query a post-filtered scan over a JSONB extraction, and reinvents
  `namespace` with more steps. No reference implementation does this.
- **ragsage owns the pool but drops RLS**, filtering by `namespace` in SQL —
  rejected. `PgVectorStore.search` currently carries *no* owner predicate at all
  and is safe only because the database enforces isolation. Trading a database
  guarantee for code correctness is a downgrade, and it is the one thing
  LightRAG's Postgres backend does that we should not copy (it has no RLS
  anywhere, and resolves its workspace from an env var with a `"default"`
  fallback).

## Consequences

- The backend deletes `app/ingestion/stores.py`, `app/providers/`, the `Chunk`
  model, and `chunks` from `USER_SCOPED_TABLES` — roughly 750 lines. Its
  composition roots collapse to constructing one `RagSage`.
- Two isolation variables exist in one deployment: `app.user_id` for the
  backend's five tables, and ragsage's own for `ragsage_chunks`, on separate
  connections. They must agree in meaning and nothing enforces that they do.
- Two connection pools reach the same Postgres. They must be sized together
  against `max_connections`.
- `chunks.user_id → users.id ON DELETE CASCADE` is gone; ragsage cannot reference
  a `users` table. Purging a deleted user's chunks becomes an explicit
  `await sage.purge(scope)` call. Nothing deletes users today, so this is new
  code for whenever account deletion ships.
- `ragsage` gains sqlalchemy, asyncpg, pgvector, and the provider SDKs as core
  dependencies — including numpy 2.5.1 and `tokenizers` via `voyageai`. The
  parser's import graph stays clean, and `tests/test_dependency_guard.py` keeps
  guarding `ragsage.parsing` specifically.
- Ingestion no longer shares a transaction with the document lifecycle. It never
  did — `_begin` and `_transition` already open their own sessions — so this
  changes nothing in practice.
