# Adapters

What ships in the box. Each of these is one implementation of a port, not a privileged part of
the engine — swapping any of them changes no ragsage code.

## Parsing and chunking

Heuristic and model-free by decision
([ADR-0001](../adr/0001-heuristic-parser-replaces-docling.md)): the import graph carries no
torch, no numpy ≥ 2, no transformers, no onnxruntime and no magika, which is a portability
promise enforced by a test that probes the graph in a clean subprocess. See
[Failure modes](../failure-modes.md) for what that costs.

```{eval-rst}
.. automodule:: ragsage.parsing
   :members:
```

## Contextualizing

```{eval-rst}
.. automodule:: ragsage.contextualizing
   :members:
```

## Models: Voyage and OpenAI

Core dependencies rather than an extra, so `pip install ragsage` plus two API keys is a
working engine. Their configuration is on the [Configuration](configuration.md) page.

```{eval-rst}
.. automodule:: ragsage.providers
   :members: VoyageEmbedder, VoyageReranker, OpenAIContextualizer, OpenAIQueryRewriter,
             OpenAILLMClient, ProviderClients, build_provider_clients
```

## Storage: Postgres

pgvector dense retrieval over an HNSW cosine index, alongside a generated `tsvector` lexical
column with GIN, both Row-Level-Security-scoped by namespace. `PostgresConfig` is on the
[Configuration](configuration.md) page.

```{eval-rst}
.. automodule:: ragsage.storage
   :members: Database, PgVectorStore, PgLexicalStore, PgDocumentStore, SchemaMismatch,
             migrate, migration_statements, purge_namespace, open_scoped_session,
             existing_embedding_dim, isolation_preamble, create_engine, create_sessionmaker
```

## Caching

```{eval-rst}
.. automodule:: ragsage.caching
   :members:
```

## Fakes

A genuine in-memory implementation of every port. Not only a testing convenience: they are
what makes the engine runnable the moment it is installed, and the standing proof that the
library is free of web, tenant and provider concepts.

```{eval-rst}
.. automodule:: ragsage.fakes
   :members:
```
