# Overview

Generated from the docstrings in `src/`, which is where the reasoning lives — most of them
explain *why* a thing is shaped the way it is, not only what it does. Every entry has a
`[source]` link.

The surface divides into five layers, and reading them in this order matches how the engine
is put together:

| Page | What's in it |
|---|---|
| [Façades](site:api/facades) | The four entry points: ingest, query, evaluate, and the assembler over all of them. |
| [Ports](site:api/ports) | The thirteen `Protocol`s every adapter conforms to. |
| [Models](site:api/models) | The domain data — documents, chunks, answers, citations, scope. |
| [Configuration](site:api/configuration) | The frozen dataclasses you construct and pass in. |
| [Adapters](site:api/adapters) | What ships in the box: the heuristic parser, Voyage/OpenAI, Postgres, and the fakes. |

```{toctree}
:maxdepth: 2
:hidden:

facades
ports
models
configuration
adapters
```

## The shape of it

```text
ingest   RawSource ─▶ parse ─▶ classify ─▶ chunk ─▶ contextualize ─▶ embed ─┬▶ VectorStore
                                                                           ├▶ LexicalStore
                                                                           └▶ DocumentStore

query    question ─▶ rewrite ─▶ embed ─┬▶ dense ───┐
                                       └▶ lexical ─┴▶ RRF fuse ─▶ rerank ─▶ min_score gate
                                                                                  │
                     Answer + Citations ◀─ LLM ◀─ bounded context ◀───────────────┘
```

Dense and lexical hits combine by Reciprocal Rank Fusion, so a chunk strong in either channel
surfaces. A cross-encoder reranks the fused candidates and only the top few reach the model's
context. Retrieval below `min_score` is treated as empty — which is what turns a weak match
into an honest *not found*.
