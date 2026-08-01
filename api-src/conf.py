"""Sphinx configuration — a build-time generator, not a documentation site.

The site is Fumadocs (``website/``). Sphinx survives here for exactly one reason:
the docstrings in ``src/`` carry around two hundred ``:class:``/``:meth:``/
``:mod:``/``:attr:`` cross-references written in Sphinx's dialect, plus numpydoc
``Parameters`` sections. Sphinx resolves every one of those into a working link
before a single byte of Markdown is emitted; every JavaScript-side generator
renders them as literal ``:class:`Scope``` text, and no upstream resolver exists.
So Sphinx reads the docstrings, ``sphinx-markdown-builder`` writes Markdown, and
the result is served by a stack that has never heard of reStructuredText.

Nothing here is ever served. The output of this build is an intermediate — six
Markdown pages that land in ``website/content/api/`` and are then the site's API
reference. Prose is authored directly as MDX and does not pass through here.

Run it through the generator, which also applies the two post-passes the builder
needs (see ``tools/generate_api_docs.py``)::

    uv run --group docs python tools/generate_api_docs.py

The raw builder invocation, for debugging that script — note the absence of
``-W``, which is deliberate and explained beside ``_EXPECTED_WARNINGS`` there::

    uv run --group docs sphinx-build -b markdown api-src <outdir>
"""

from __future__ import annotations

from importlib.metadata import version as installed_version

# -- Project ---------------------------------------------------------------- #

project = "ragsage"
author = "Niraj Kumar"
copyright = "2026, Niraj Kumar"

# Read from the installed distribution rather than duplicated here: a docs build
# that advertises a different version than the package it documents is worse than
# no version at all.
release = installed_version("ragsage")
version = ".".join(release.split(".")[:2])

# -- Extensions ------------------------------------------------------------- #

extensions = [
    # The API reference: pull docstrings straight out of the package.
    "sphinx.ext.autodoc",
    # Understands the numpydoc `Parameters ---------` sections in Scope and
    # PostgresConfig. Google style is accepted too; nothing in src/ uses it.
    "sphinx.ext.napoleon",
    # `:class:`AsyncSession`` and friends resolve to the upstream project's docs
    # instead of rendering as dead literals.
    "sphinx.ext.intersphinx",
    # Markdown authoring for these stub pages, so the `automodule` directives sit
    # inside `eval-rst` blocks in files that are otherwise plain Markdown.
    "myst_parser",
    # The whole point of this tree: emit Markdown rather than HTML.
    "sphinx_markdown_builder",
]

# `sphinx.ext.viewcode` is deliberately absent. It is what used to put a
# "[source]" link next to every documented object, and it still works — in HTML.
# The Markdown builder never visits the nodes that carry those links, so all 222
# of them vanish silently. `sphinx.ext.linkcode` was tested as the replacement and
# is dropped the same way. Source links are injected by the generator's own
# post-pass instead, off the dotted name in each generated heading.

# -- Source files ----------------------------------------------------------- #

# Nothing to exclude but editor and OS debris: this directory holds `index` and
# the five layer pages and nothing else. It has to stay that way — the generator
# owns `website/content/api/` wholesale, so a stray file here becomes a stray page
# there, and the generator refuses to run rather than emit one it cannot name.
exclude_patterns = ["Thumbs.db", ".DS_Store"]

# What these six pages actually use. `smartquotes` matters — the prose here is
# written with real apostrophes and it renders them as such. `linkify` is
# deliberately absent (it rewrites bare URLs and would surprise the authors), and
# `substitution` went with the prose that used it.
myst_enable_extensions = ["colon_fence", "deflist", "smartquotes"]

myst_heading_anchors = 3

# These stub pages occasionally want to link into the hand-written prose, which
# lives in the Fumadocs app and is invisible to Sphinx. A plain `/failure-modes`
# is parsed as a document reference, fails to resolve, and is emitted as an empty
# href — a dead link that raises a warning nobody has to act on. Declaring a
# scheme makes the intent explicit and the expansion mechanical:
# `[Failure modes](site:failure-modes)` becomes the root-served route.
myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "site": "/{{path}}",
}

# Every unmarked literal block in a ragsage docstring is Python — the `::` blocks
# are all short usage examples. Sphinx's own default, the pseudo-language
# `"default"`, means "guess, and fall back to nothing", and the Markdown builder
# writes that guess out as the fence's language: ```default, which no syntax
# highlighter has heard of. Naming the language says what is true and emits a
# fence the site can highlight.
highlight_language = "python"

# -- Markdown output -------------------------------------------------------- #

# The two flags this whole migration turns on. Measured across all six generated
# pages: without them 3 of 367 anchored internal links resolve; with them, 367 of
# 367. The builder emits the links either way — it is the anchor *targets* that
# are missing — so every `#ragsage.models.RawSource` lands nowhere while still
# looking, in the source and in the browser, like a working link. There is no
# failure mode here that reads as a failure; do not turn these off.
markdown_anchor_signatures = True
markdown_anchor_sections = True

# Routes are extensionless (`/api/ports`, not `/api/ports.md`), so the links the
# builder writes between these six pages need their suffix removed. They are all
# siblings in one directory, so relative links resolve as-is once the suffix goes.
markdown_uri_doc_suffix = ""

# `.md`, not `.mdx`: this output is generated and contains type annotations,
# braces and angle brackets straight from the source. MDX would parse those as
# JSX and expressions; Markdown does not. Prose, which is hand-written and wants
# React components, is `.mdx` — the extension is the seam.
markdown_file_suffix = ".md"

# -- autodoc ---------------------------------------------------------------- #

# Source order, not alphabetical: these modules are written to be read top to
# bottom, and the ports in particular are ordered by where they sit in the pipeline.
autodoc_member_order = "bysource"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    # The dataclass field docs and the Protocol method stubs are the reference;
    # inherited object.__init__ noise is not.
    "undoc-members": False,
}

# The library is strictly typed, so the annotations *are* documentation — but
# repeating a long union in both the signature and the parameter list reads badly.
# Signature only.
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

# `from __future__ import annotations` is on in every module, so every annotation
# reaches autodoc as a string; this keeps them rendering as written.
autodoc_preserve_defaults = True

# -- intersphinx ------------------------------------------------------------ #

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
}

# Not nitpicky (`-n`): the module-level type aliases — ``CallConfig``,
# ``AnswerEvent``, ``Vector`` — and the Protocol parameter annotations have no
# inventory target, and turning every one of those into a build failure would buy
# noise rather than a working link.
#
# A *broken* reference still fails the build — a bad ``:class:`` target, a page
# missing from the toctree. Not through ``-W``, though: the Markdown builder emits
# one known warning of its own that ``-W`` would fail on and ``suppress_warnings``
# cannot name, so the generator collects the warnings and fails on any that is not
# that one. See ``_EXPECTED_WARNINGS`` in ``tools/generate_api_docs.py``.
