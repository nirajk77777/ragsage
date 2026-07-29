# Examples

Three scripts, each runnable with no arguments and no setup beyond an installed
package — no network, no database, no API key:

```console
$ python examples/fakes_end_to_end.py
$ python examples/custom_embedder.py
$ python examples/custom_parser.py
```

| Script | What it shows |
|---|---|
| [`fakes_end_to_end.py`](fakes_end_to_end.py) | The whole ingest-and-query loop against the in-memory fakes, including the honest not-found path. |
| [`custom_embedder.py`](custom_embedder.py) | Implementing the `Embedder` port against something that isn't Voyage — the adapter imports and subclasses nothing from ragsage. |
| [`custom_parser.py`](custom_parser.py) | Implementing `DocumentParser` to bypass the built-in `HeuristicBackend` for a format it doesn't understand. |

They run in CI (`.github/workflows/ci.yml`) so a change to a port signature
breaks the build rather than the examples.

A fourth example — `RagSage.from_config(...)` end to end against a real
Postgres — lands with the assembler façade in ticket 05.
