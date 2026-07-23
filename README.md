# ragsage

> A sage that only speaks from your corpus.

`ragsage` is a reusable, **model-agnostic** and **tenancy-agnostic** Retrieval-Augmented
Generation engine. It owns the whole RAG domain — parsing, chunking, embedding, hybrid
retrieval, reranking, grounded generation, and verifiable citations — behind
provider-agnostic ports, so you can swap any model or store by implementing an interface.

The library never imports web, auth, or tenant concepts. Callers pass an opaque
[`Scope`](src/ragsage/scope.py) (a namespace plus optional metadata filters); the engine
treats it as an untyped label. That single boundary is what lets the same engine run
single-tenant from a script or CLI, and multi-tenant behind a SaaS backend, unchanged.

## Try it standalone

The library ships in-memory fake adapters for every port, so the full
ingest-and-query loop runs offline with no web server, database, or network:

```console
$ ragsage ingest ./docs
  + france.txt: 2 chunks
  corpus saved to .ragsage/state.json

$ ragsage query "What is the capital of France?"
Paris is the capital of France. It sits on the Seine river. [1]

Sources:
  [1] f9d8ec95d8be4f2e (page 1)
```

Or drive the façades directly, wiring the fakes (swap in real adapters — Voyage
embeddings, Cohere rerank, pgvector — behind the same ports for production):

```python
from ragsage import IngestionPipeline, QueryEngine, RawSource, Scope
from ragsage.fakes import FakeEngineKit

kit = FakeEngineKit()
scope = Scope(namespace="local")
# ... build IngestionPipeline / QueryEngine from `kit` and run ingest() / query()
```

## Public surface

- **Façades:** `IngestionPipeline.ingest`, `QueryEngine.query`, `Evaluator.evaluate`.
- **Ports:** `DocumentParser`, `PageClassifier`, `Chunker`, `Contextualizer`, `Embedder`,
  `Reranker`, `LLMClient`, `VectorStore`, `LexicalStore`, `DocumentStore`, `Cache`, `Tracer`.
- **Models:** `Document`, `Chunk`, `Citation`, `Answer`, `Scope`, and friends.
- **Fakes:** a working in-memory adapter for every port, in `ragsage.fakes`.

## Status

Early. Ports, pipelines, and the standalone CLI are in place with in-memory fakes;
production adapters and the async streaming surface land as the backend wires them.

## License

Apache-2.0.
