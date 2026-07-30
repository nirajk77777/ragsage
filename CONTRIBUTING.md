# Contributing to ragsage

Issues and pull requests are welcome. This file is the short version of what the
project expects; everything in it is enforced by CI, so nothing here is a matter of
taste you have to guess at.

## Getting set up

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nirajk77777/ragsage.git
cd ragsage
uv sync
uv run pytest
```

The tests are offline by default — no database, no API keys, no network. If they
pass, your environment is right.

## The gates

CI runs these, in this order, and a pull request has to clear all of them:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src examples
uv run pytest
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

Two notes on the less obvious ones:

- `ruff format` **also formats the Python blocks inside Markdown**, so a snippet in
  the README or the docs is held to the same style as the library.
- The docs build runs with `-W`, so a broken `:class:` target or a page missing from
  a toctree fails the build rather than shipping a dead link.

Beyond the gates, CI *executes* all four scripts in `examples/`. They are argument-free
and type-checked, and a change that breaks one breaks the build.

### The suites that don't run by default

Two opt in through the environment, because they need something CI can't assume:

```bash
# Integration tests against a real Postgres (needs the `vector` extension).
TEST_DATABASE_URL=postgresql://user:pw@localhost:5432/ragsage uv run pytest -m integration

# The real Voyage and OpenAI APIs. Costs money.
RAGSAGE_LIVE_PROVIDERS=1 OPENAI_API_KEY=… VOYAGE_API_KEY=… uv run pytest -m live_providers
```

Without those variables the tests skip with a message rather than fail.

## House rules

These are the constraints a newcomer is most likely to break by accident. Most have a
test standing behind them, so you'll find out — but knowing first is cheaper.

**The import graph is load-bearing.** Two invariants, both checked in a clean
subprocess (an in-process `sys.modules` check would prove nothing once pytest has
already imported everything):

- `ragsage.parsing` must not pull `torch`, `numpy>=2`, `transformers`, `onnxruntime`
  or `magika`. That is a portability promise to every downstream consumer, not a
  preference — see `tests/test_dependency_guard.py` and
  [ADR-0001](docs/adr/0001-heuristic-parser-replaces-docling.md).
- A bare `import ragsage` must not pull SQLAlchemy, asyncpg or a provider SDK. Storage
  and providers are installed, but reaching them is deliberate. See
  `tests/test_storage_imports.py` and
  [ADR-0002](docs/adr/0002-ragsage-owns-its-storage.md).

If you need something heavy, put it behind a lazy import inside the function that
needs it, the way `ragsage.parsing.pdf` does.

**The library never reads the environment.** No `os.environ`, anywhere in `src/`.
Every key, DSN and model id arrives in a frozen dataclass constructed by the caller.
A config object whose values depend on the ambient process is untestable and silently
different in production.

**Ports are `Protocol`s, and they stay that way.** An adapter conforms by shape and
imports nothing from ragsage. Don't add an abstract base class, a registry, or a
plugin entry point — the dependency arrow points inward, and that is the whole design.

**No web, auth or tenant concepts in the library.** Callers pass an opaque `Scope`.
If a change would make a pricing, auth or tenancy decision visible inside `src/`,
it belongs in the consumer instead.

**Changing a port signature means changing the examples.** They are documentation that
runs, and CI runs them. Update them in the same commit.

**Docstrings are reStructuredText.** They use Sphinx roles — ``` :class:`Scope` ```,
``` :mod:`ragsage.ports` ``` — which the docs site turns into links. One trap worth
naming, because it fails silently rather than loudly: a role or literal closed
immediately by a letter does not close at all, and docutils swallows everything up to
the next backtick.

```
:class:`Chunk`s          ← wrong: eats the rest of the paragraph
:class:`Chunk` objects   ← right
```

**ADRs are the record.** Design decisions live in
[`docs/adr/`](https://github.com/nirajk77777/ragsage/tree/main/docs/adr). A change that
contradicts one should amend that ADR in the same pull request rather than quietly
diverge from it.

## Style

Ruff and mypy decide, not review comments. Line length is 100, the target is `py312`,
and mypy runs `--strict` over both `src` and `examples`. Run `uv run ruff format .`
before pushing and the formatting question never comes up.

The one thing the tools can't check: comments and docstrings here explain **why**, not
what. A docstring that restates the signature isn't worth the line. Match the density
and voice of the file you're editing.

## Pull requests

- One concern per pull request. A formatting sweep mixed into a behaviour change is
  two reviews wearing one hat.
- Write the subject in the imperative and the body about *why* — the existing history
  is the reference (`git log` is the style guide).
- Say what you ran. If you couldn't run the Postgres or live-provider suites, say that
  too; it's useful, not embarrassing.
- New behaviour needs a test. Bug fixes need the test that would have caught it.

## Reporting things

- **Bugs and feature requests:** [open an issue](https://github.com/nirajk77777/ragsage/issues).
  For a retrieval or parsing problem, check
  [docs/failure-modes.md](https://ragsage.readthedocs.io/en/latest/failure-modes.html)
  first — it lists the known weak spots symptom-first, and tells you how to confirm each
  from the parser's own output.
- **Security vulnerabilities:** do **not** open an issue. See
  [SECURITY.md](SECURITY.md).

## Licence

ragsage is [MIT](LICENSE). By contributing you agree your contribution is licensed under
the same terms.
