# ragsage

**A sage that only speaks from your corpus.**

`ragsage` answers questions from your documents, and says so honestly when it can't. Every
answer is either grounded prose with citations that resolve back to a chunk and a page, or an
explicit *not found* — the engine never invents a confident answer out of thin retrieval.

```bash
pip install ragsage
```

It owns the whole RAG domain — parsing, chunking, contextualizing, embedding, hybrid
retrieval, reranking, grounded generation and citations — behind thirteen
{mod}`typing.Protocol` ports, so any model or store is swapped by implementing an interface
rather than by editing the engine. It never imports web, auth or tenant concepts: callers
pass an opaque {class}`~ragsage.scope.Scope`, and the engine treats it as an untyped label.

New here? [**Quickstart**](quickstart.md) runs the full ingest-and-query loop offline in
about sixty seconds, with no database and no API key.

```{toctree}
:maxdepth: 2
:caption: Guides
:hidden:

quickstart
failure-modes
```

```{toctree}
:maxdepth: 2
:caption: API reference
:hidden:

api/index
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

releasing
measurements/07-heading-window-contextualizer
```

## Where to go next

| If you want to… | Read |
|---|---|
| run the engine in the next five minutes | [Quickstart](quickstart.md) |
| know what every class and port does | [API reference](api/index.md) |
| work out why an answer looks wrong | [Failure modes of the heuristic parser](failure-modes.md) |
| cut a release | [Releasing](releasing.md) |
| see what changed in a version | [GitHub releases](https://github.com/nirajk77777/ragsage/releases) |

## Status

Alpha. The public surface is stable in shape, but minor versions may still break it before
1.0. Source, issues and pull requests live on
[GitHub](https://github.com/nirajk77777/ragsage); the package is on
[PyPI](https://pypi.org/project/ragsage/). Licensed
[MIT](https://github.com/nirajk77777/ragsage/blob/main/LICENSE).
