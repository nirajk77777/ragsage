# Examples

Four scripts, each runnable with no arguments and no setup beyond an installed
package:

```console
$ python examples/fakes_end_to_end.py
$ python examples/custom_embedder.py
$ python examples/custom_parser.py
$ python examples/assembled_engine.py
```

| Script | What it shows |
|---|---|
| [`fakes_end_to_end.py`](fakes_end_to_end.py) | The whole ingest-and-query loop against the in-memory fakes, including the honest not-found path. |
| [`custom_embedder.py`](custom_embedder.py) | Implementing the `Embedder` port against something that isn't Voyage — the adapter imports and subclasses nothing from ragsage. |
| [`custom_parser.py`](custom_parser.py) | Implementing `DocumentParser` to bypass the built-in `HeuristicBackend` for a format it doesn't understand. |
| [`assembled_engine.py`](assembled_engine.py) | `RagSage.from_config(...)` — migrate, ingest, query and purge from one config object, instead of hand-wiring ports. |

The first three need no network, no database and no API key. The fourth wants a
Postgres with the `vector` extension and **skips with a message when there isn't
one**, so all four are safe to run unconditionally:

```console
$ TEST_DATABASE_URL=postgresql://user:pw@localhost:5432/ragsage \
    python examples/assembled_engine.py
```

It also swaps the model adapters for fakes, so it needs no API keys — which
doubles as a demonstration that every adapter `from_config` builds is replaceable.

They all run in CI (`.github/workflows/ci.yml`) so a change to a port signature
breaks the build rather than the examples.
