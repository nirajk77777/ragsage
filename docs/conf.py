"""Sphinx configuration for the ragsage documentation site.

Sphinx rather than MkDocs, for one concrete reason: the docstrings in ``src/`` are
already written in Sphinx's dialect — around two hundred ``:class:``/``:meth:``/
``:mod:``/``:attr:`` cross-references plus numpydoc ``Parameters`` sections. Sphinx
turns those into working links with no edit to a single docstring; mkdocstrings
would render them as literal ``:class:`Scope``` text. MyST-Parser keeps the prose
side (the ADRs, the failure-mode catalogue, these notes) in Markdown, so nothing
here has to be written in reStructuredText.

The source directory is ``docs/`` itself rather than a nested ``docs/source``, so
the Markdown files already in the repo are the site's pages — they stay readable
on GitHub and get served here without being copied or moved.

Built on Read the Docs by ``.readthedocs.yaml``; locally by::

    uv run --group docs sphinx-build -b html docs docs/_build/html
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
    # "[source]" links next to every documented object. Worth more than usual
    # here: the docstrings explain decisions, and the code is the evidence.
    "sphinx.ext.viewcode",
    # Markdown authoring, so the ADRs and guides need no conversion.
    "myst_parser",
    # A copy button on every code block.
    "sphinx_copybutton",
]

# -- Source files ----------------------------------------------------------- #

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Instructions for coding agents working *on* this repo, not documentation of
    # the library. They point at repo-local scratch space and would be noise to a
    # reader — the sdist excludes CLAUDE.md for the same reason.
    "agents/**",
    # Release notes are the body of a GitHub release: release.yml passes the file
    # to `gh release create --notes-file`, so it deliberately has no H1 (the tag is
    # the title). Serving it here would mean bending the file to suit two masters;
    # the canonical home is the releases page, which index.md links to.
    "release-notes/**",
]

# MyST extensions, kept to the ones the existing Markdown actually needs:
# `colon_fence` for ::: blocks, `deflist` for definition lists, `linkify` is
# deliberately absent (it rewrites bare URLs and would surprise the ADR authors).
myst_enable_extensions = ["colon_fence", "deflist", "smartquotes", "substitution"]

# The ADRs and the failure-mode catalogue use `##`-level headings for structure and
# are linked to by section elsewhere; generate anchors so those links resolve.
myst_heading_anchors = 3

# -- HTML output ------------------------------------------------------------ #

html_theme = "furo"
html_title = "ragsage"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/nirajk77777/ragsage/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/nirajk77777/ragsage",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                "8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01"
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95"
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.63-.18"
                "1.31-.27 1.98-.27.67 0 1.35.09 1.98.27 1.53-1.04 2.2-.82 2.2-.82.44 "
                "1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
                "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 "
                '.21.15.46.55.38A7.995 7.995 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

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
# noise rather than a working link. The build does run with ``-W``, so a *broken*
# reference — a bad ``:class:`` target, a missing toctree entry — still fails.
