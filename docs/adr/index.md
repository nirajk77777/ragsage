# Architecture decisions

Why the engine is shaped the way it is. Each record states the context, the decision and what
it costs; a change that contradicts one should amend it rather than quietly diverge.

| # | Decision | Status |
|---|---|---|
| [0001](0001-heuristic-parser-replaces-docling.md) | Heuristic parser replaces Docling | Accepted — premise amended 2026-07-29 |
| [0002](0002-ragsage-owns-its-storage.md) | ragsage owns its storage and its model adapters | Accepted |
| [0004](0004-typed-blocks-ride-in-chunk-metadata.md) | Typed blocks ride in `Chunk.metadata`, not in the domain model | Proposed |

```{toctree}
:maxdepth: 1
:hidden:

0001-heuristic-parser-replaces-docling
0002-ragsage-owns-its-storage
0004-typed-blocks-ride-in-chunk-metadata
```
