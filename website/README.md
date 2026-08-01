# The ragsage documentation site

A [Fumadocs](https://fumadocs.dev) site on Next.js, exported statically and served
by nginx. Not part of the Python package: `website/` is excluded from the sdist,
and `tests/test_documentation_boundary.py` keeps it that way.

## What is written here, and what is not

| Path | Who writes it |
| --- | --- |
| `content/*.mdx` | You. Prose, in MDX, with React components available. |
| `content/meta.json` | You. Navigation — the order and grouping in the sidebar. |
| `content/api/` | **The generator.** Wiped on every run; never hand-edit. |

The file path is the URL, minus the extension: `content/quickstart.mdx` is
`/quickstart`. There is no `/docs` prefix — the host is already `docs.`-prefixed.

The API reference is generated from the docstrings in `src/` by
`tools/generate_api_docs.py`, which runs Sphinx as an invisible build step. Sphinx
is the only thing that resolves the ~200 `:class:`/`:meth:` cross-references those
docstrings are written in; it emits Markdown, and nothing Sphinx-shaped is served.

## Running it

```bash
uv run --group docs python tools/generate_api_docs.py   # from the repository root
npm install
npm run dev
```

The generated pages are gitignored, so that first command is what makes `/api`
exist in a fresh checkout. Re-run it after editing a docstring.

## Building it the way it deploys

```bash
docker build -f website/Dockerfile -t ragsage-docs .    # from the repository root
docker run --rm -p 8080:80 ragsage-docs
```

Three stages: Python generates the API reference, Node builds the static export
and runs the gate, nginx serves the result. Neither Python nor Node survives into
the image that runs.

The build context is the **repository root**, not `website/` — the generator has to
read `src/`. On Coolify that means Base Directory stays `/` and the Dockerfile is
located by Dockerfile Location; the two settings are not interchangeable.

## The gate

`npm run check` (also run inside the image, on the very files nginx will serve)
asserts that every internal link resolves down to its `#fragment`, that no
signature lost its keyword-only `*`, that every source link the generator injected
is still served, that navigation and content are in bijection, and that every
expected page exists at its expected route.

Each assertion is paired with a canary establishing that the corpus it ran over was
not empty, so an empty or partial build reports failure rather than a vacuous
success. See the header of `scripts/check-built-site.mjs` for why that matters more
than usual here.
