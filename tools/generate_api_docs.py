"""Generate the site's API reference from the docstrings in ``src/``.

Sphinx is not the documentation site any more — Fumadocs is (``website/``). But
the docstrings in ``src/`` carry 202 cross-references written in Sphinx's dialect,
and Sphinx is the only thing that resolves them. So it stays, as an invisible
build-time data source: it reads the docstrings, ``sphinx-markdown-builder``
writes Markdown, and this script corrects the two things that builder gets wrong
before dropping the result into ``website/content/api/``.

    uv run --group docs python tools/generate_api_docs.py

That directory belongs to this script and is wiped on every run. Nothing
hand-written may live there; prose lives one level up, in ``website/content/``,
and never passes through Sphinx at all.

The two corrections are pure text transforms, tested in
``tests/test_api_docs_generator.py``, and each is documented at its definition.
Read those docstrings before changing either — both defects produce output that
*looks* correct, which is why they are worth this much ceremony.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPHINX_SRC = _REPO_ROOT / "api-src"
_OUTPUT_DIR = _REPO_ROOT / "website" / "content" / "api"

# What this run produced, written for the gate that walks the built site.
#
# It is what makes the gate a *conservation* check rather than a check against a
# number somebody typed: the site must serve as many source links as the
# generator injected, whatever that number happens to be this week. A hard-coded
# 222 would have to be edited every time a class is documented, and a threshold
# loose enough not to need editing is loose enough to miss a page.
#
# Outside `content/`, deliberately: a stray `.json` inside it would be read as a
# navigation file.
_MANIFEST = _REPO_ROOT / "website" / ".api-manifest.json"

# Where a `[source]` link points. `main` rather than the released tag, matching
# what the retired furo theme linked to: the reference documents the code as it
# stands on the default branch, which is also the code these docstrings came from.
_BLOB_BASE = "https://github.com/nirajk77777/ragsage/blob/main"

#: What a source link looks like once injected. Escaped brackets so the rendered
#: link text is the literal ``[source]`` the old site used, rather than a stray
#: bracket either side of a link labelled "source".
_SOURCE_LINK = "[\\[source\\]]({url})"

_SOURCE_LINK_RE = re.compile(r"^\[\\\[source\\\]\]\(", re.MULTILINE)

# Every generated heading is preceded by an anchor carrying the object's full
# dotted name — the method headings themselves are bare (`#### query(...)`), so
# the anchor is the only place the name survives. Module headings are prefixed.
_ANCHOR_RE = re.compile(r'^<a id="(?:module-)?(ragsage\.[A-Za-z_][\w.]*)"></a>$')

_HEADING_RE = re.compile(r"^#{2,6} \S")

# Warnings this build is expected to emit. Anything else fails the generation.
#
# This list is why the build is not simply run under `-W`. The eaten keyword-only
# marker (below) surfaces as an untyped Sphinx warning, so `suppress_warnings`
# cannot name it and `-W` would fail on a defect we are already correcting. An
# explicit allow-list is the stricter arrangement anyway: it states which known
# defect is tolerated, and a *new* warning — a broken `:class:` target, a
# malformed docstring — still fails the build instead of joining the noise.
_EXPECTED_WARNINGS = (
    # The `*` separator, wrapped by Sphinx's Python domain in an `abbreviation`
    # node the Markdown builder has no handler for. Corrected by
    # `restore_keyword_only_markers`.
    "unknown node type: <abbreviation: <#text: '*'>>",
)


# ---------------------------------------------------------------------------- #
# Post-pass one: restore the eaten keyword-only marker
# ---------------------------------------------------------------------------- #

# `f(a, *, b)` arrives as `f(a, , b)`; `f(*, a)` as `f(, a)`. Both are corrected
# by putting the marker back where the comma pair or the bracket-comma sits.
_MARKER_SUBSTITUTIONS = (
    (re.compile(r", , (?=\S)"), ", *, "),
    (re.compile(r"\(, (?=\S)"), "(*, "),
)


def restore_keyword_only_markers(text: str) -> str:
    """Put back the PEP 3102 ``*`` that the Markdown builder ate.

    Sphinx's Python domain wraps the keyword-only separator in a docutils
    ``abbreviation`` node so HTML can show a tooltip explaining it. The Markdown
    builder has no handler for that node type: it warns once, skips the node, and
    emits the surrounding punctuation with a hole in the middle. A reader is then
    shown ``chunk(document, pages, , size, overlap)`` for a method whose source
    reads ``chunk(self, document, pages, *, size, overlap)`` — which does not
    merely look untidy, it removes the only signal that ``size`` and ``overlap``
    are keyword-only.

    Restoration is unambiguous because neither artefact can occur in a valid
    Python signature: an argument list has no empty slots. The rewrite is confined
    to heading lines all the same. Signatures are headings here, and a docstring
    is free to contain ``, ,`` in prose or in a code sample — correcting a
    signature is not worth corrupting prose that happens to match.
    """

    def restored(line: str) -> str:
        for pattern, replacement in _MARKER_SUBSTITUTIONS:
            line = pattern.sub(replacement, line)
        return line

    return "\n".join(
        restored(line) if _HEADING_RE.match(line) else line for line in text.split("\n")
    )


def count_eaten_markers(text: str) -> int:
    """How many markers :func:`restore_keyword_only_markers` would put back.

    Shares its patterns and its heading filter with the pass itself, so the two
    cannot disagree about what an artefact is — the number this reports is what
    the gate later checks survived into the built site.
    """
    return sum(
        len(pattern.findall(line))
        for line in text.split("\n")
        if _HEADING_RE.match(line)
        for pattern, _ in _MARKER_SUBSTITUTIONS
    )


# ---------------------------------------------------------------------------- #
# Post-pass two: inject the dropped source links
# ---------------------------------------------------------------------------- #


def inject_source_links(text: str, resolve: Callable[[str], str | None]) -> str:
    """Put a ``[source]`` link under every heading whose object can be located.

    ``sphinx.ext.viewcode`` produced 222 of these in the HTML build, and the
    Markdown builder drops all 222 — the nodes carrying them are never visited.
    ``sphinx.ext.linkcode`` was tested as the remedy: it works in HTML, and is
    dropped identically. So the links are rebuilt from the outside, off the dotted
    name in the anchor the builder writes above each heading.

    ``resolve`` maps a dotted name to a URL, or to ``None`` for the objects that
    have no source location — dataclass fields, enum members, module constants,
    properties. Those are skipped rather than guessed at: they are exactly the
    anchors ``viewcode`` also passed over, so skipping them reproduces the old
    site's link set instead of inventing 187 links it never had.

    The link goes on its own line *below* the heading rather than at the end of
    it, which is where the HTML build put it. In Markdown the heading text is also
    the table-of-contents entry, and a ``[source]`` welded onto every signature
    would repeat itself down the whole sidebar.
    """
    lines = text.split("\n")
    out: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        out.append(line)
        index += 1

        anchor = _ANCHOR_RE.match(line)
        if anchor is None:
            continue

        heading = _next_content_line(lines, index)
        if heading is None or not _HEADING_RE.match(lines[heading]):
            continue

        # Replay the blank run and the heading itself, so the link can be appended
        # after them without reordering anything.
        out.extend(lines[index : heading + 1])
        index = heading + 1

        injected = _next_content_line(lines, index)
        if injected is not None and _SOURCE_LINK_RE.match(lines[injected]):
            continue

        url = resolve(anchor.group(1))
        if url is not None:
            out.append(_SOURCE_LINK.format(url=url))

    return "\n".join(out)


def _next_content_line(lines: list[str], start: int) -> int | None:
    """Index of the next non-blank line at or after ``start``, if any."""
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


# ---------------------------------------------------------------------------- #
# Emitting for the target: frontmatter
# ---------------------------------------------------------------------------- #

# The heading, its newline, and the blank line under it, so lifting it out leaves
# no gap where it stood.
_H1_RE = re.compile(r"^# (?P<title>.+)\n(?:\n|$)", re.MULTILINE)


def add_frontmatter(text: str, description: str) -> str:
    """Lift the page's ``# Heading`` into the frontmatter the site reads.

    Fumadocs renders the title above the body from frontmatter, so a surviving H1
    would print the page's name twice. The anchor above it stays: cross-page links
    target the section, and the heading is not the only thing that pointed there.

    A page with no H1 gets frontmatter with no ``title``, which the site's schema
    rejects by name. That is deliberate — it means a generator bug surfaces as
    "this page has no title" rather than as a page called *Untitled* sitting in
    the navigation looking like someone meant it.
    """
    heading = _H1_RE.search(text)
    fields = {} if heading is None else {"title": heading.group("title")}
    fields["description"] = description

    body = text if heading is None else _H1_RE.sub("", text, count=1)
    # JSON string literals are valid YAML scalars, which saves hand-rolling the
    # quoting rules for a title like `Models: the domain data`.
    front = "".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}\n" for key, value in fields.items()
    )
    return f"---\n{front}---\n\n{body}"


# ---------------------------------------------------------------------------- #
# Resolving a dotted name to a line of source
# ---------------------------------------------------------------------------- #


def resolve_source_url(dotted: str) -> str | None:
    """Map ``ragsage.query.QueryEngine`` to a GitHub blob URL, or ``None``.

    ``None`` is the honest answer for anything ``inspect`` cannot place: a
    dataclass field is an annotation, not an object with a line number, and a
    module constant is a value that has forgotten where it came from.
    """
    obj = _import_dotted(dotted)
    if obj is None:
        return None

    # Unwrap first, and unwrap *once* for both questions. `getsourcelines`
    # unwraps internally and `getsourcefile` does not, so asking them separately
    # about an `@asynccontextmanager` — `open_scoped_session` is one — pairs a
    # line number in `storage/session.py` with a filename of `contextlib.py`.
    obj = inspect.unwrap(obj)

    try:
        source_file = inspect.getsourcefile(obj)
        _, line_number = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None

    if source_file is None:
        return None

    relative = _package_relative_path(Path(source_file))
    if relative is None:
        return None

    return f"{_BLOB_BASE}/{relative}#L{line_number}"


def _import_dotted(dotted: str) -> Any | None:
    """Resolve a dotted name by importing the longest prefix that is a module."""
    parts = dotted.split(".")

    for split in range(len(parts), 0, -1):
        module = _import_module(".".join(parts[:split]))
        if module is None:
            continue

        obj: Any = module
        for attribute in parts[split:]:
            obj = getattr(obj, attribute, None)
            if obj is None:
                return None
        return obj

    return None


def _import_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _package_relative_path(source_file: Path) -> str | None:
    """``.../site-packages/ragsage/query.py`` to ``src/ragsage/query.py``.

    Read off the installed package's own location rather than assumed, so the
    generator produces the same URLs whether ``ragsage`` was installed editable
    from this checkout or unpacked into a container's site-packages.

    ``None`` for anything outside the package. A re-exported third-party object
    would otherwise be given a blob URL into this repository at a path that does
    not exist here — a 404 dressed as a source link.
    """
    import ragsage

    package_parent = Path(ragsage.__file__).resolve().parent.parent
    try:
        return f"src/{source_file.resolve().relative_to(package_parent)}"
    except ValueError:
        return None


# ---------------------------------------------------------------------------- #
# Navigation
# ---------------------------------------------------------------------------- #

# The order the engine is assembled in, which is also the order that teaches the
# architecture when read top to bottom: what you call, what it calls through,
# what flows across it, what configures it, what implements it. Alphabetical would
# open on the adapters, which is the one layer a reader can ignore.
#
# The descriptions live here rather than in `api-src/`, because they are the
# navigation's copy, not the page's: the site shows them beside the link.
_PAGES: tuple[tuple[str, str], ...] = (
    ("index", "The five layers, and how the engine is put together from them."),
    ("facades", "The four entry points: ingest, query, evaluate, and the assembler."),
    ("ports", "The thirteen Protocols every adapter conforms to."),
    ("models", "The domain data — documents, chunks, answers, citations, scope."),
    ("configuration", "The frozen dataclasses you construct and pass in."),
    ("adapters", "What ships in the box: the parser, Voyage/OpenAI, Postgres, the fakes."),
)

_DESCRIPTIONS = dict(_PAGES)

# `index` is the folder's own page, so it is served at the folder's route and not
# at `/api/index`. Stated once, here, rather than re-derived wherever a stem has
# to become a URL.
_ROUTES = tuple("/api" if stem == "index" else f"/api/{stem}" for stem, _ in _PAGES)

_META = {
    "title": "API reference",
    "description": "Every public class, port and configuration object in ragsage.",
    "pages": [stem for stem, _ in _PAGES],
}


# ---------------------------------------------------------------------------- #
# Driving the build
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenerationReport:
    """What one run produced, for the operator and for the fail-fast checks."""

    pages: int
    markers_restored: int
    source_links: int


def generate() -> GenerationReport:
    """Run Sphinx, correct its output, and write the API reference.

    The output location is not a parameter. Both destinations — the pages and the
    manifest the gate reads — are fixed, and they have to agree: a run that wrote
    pages to one place and a manifest describing them to another would leave the
    gate checking one build against another build's numbers.
    """
    with tempfile.TemporaryDirectory() as raw_dir:
        raw = Path(raw_dir)
        _run_sphinx(raw)
        report = _write_pages(raw, _OUTPUT_DIR)

    _write_manifest(report)
    return report


def _run_sphinx(destination: Path) -> None:
    warnings_file = destination / "warnings.txt"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "markdown",
            "-q",
            "-w",
            str(warnings_file),
            str(_SPHINX_SRC),
            str(destination),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"sphinx-build failed with exit code {result.returncode}")

    _check_warnings(warnings_file)


def _check_warnings(warnings_file: Path) -> None:
    """Fail on any warning that is not the one defect we correct downstream."""
    if not warnings_file.exists():
        return

    unexpected = [
        line
        for line in warnings_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not any(expected in line for expected in _EXPECTED_WARNINGS)
    ]
    if unexpected:
        raise SystemExit(
            "sphinx-build emitted warnings that are not known builder defects:\n  "
            + "\n  ".join(unexpected)
        )


def _write_pages(raw: Path, output_dir: Path) -> GenerationReport:
    # Wiped, not merged: the generator owns this directory outright, so a page
    # renamed upstream cannot leave its old self behind to be served forever.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    pages = markers = links = 0

    emitted = {source.stem for source in raw.glob("*.md")}
    if emitted != set(_DESCRIPTIONS):
        raise SystemExit(
            "Sphinx emitted pages this script does not know how to describe or order.\n"
            f"  emitted:  {sorted(emitted)}\n"
            f"  expected: {sorted(_DESCRIPTIONS)}\n"
            "Add or remove the page in `_PAGES` alongside the `api-src/` change."
        )

    for source in sorted(raw.glob("*.md")):
        original = source.read_text(encoding="utf-8")
        corrected = add_frontmatter(
            inject_source_links(restore_keyword_only_markers(original), resolve_source_url),
            description=_DESCRIPTIONS[source.stem],
        )
        (output_dir / source.name).write_text(corrected, encoding="utf-8")

        pages += 1
        markers += count_eaten_markers(original)
        links += len(_SOURCE_LINK_RE.findall(corrected))

    (output_dir / "meta.json").write_text(json.dumps(_META, indent=2) + "\n", encoding="utf-8")

    return GenerationReport(pages=pages, markers_restored=markers, source_links=links)


def _write_manifest(report: GenerationReport) -> None:
    _MANIFEST.write_text(
        json.dumps(
            {
                "routes": list(_ROUTES),
                "sourceLinks": report.source_links,
                "markersRestored": report.markers_restored,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    report = generate()

    # A fail-fast canary, not the real gate. The built site is walked link by link
    # before it can deploy, and that is what proves the reference is sound — but it
    # costs a full image build. An empty or link-less generation is a broken
    # pipeline, and saying so here saves three stages of confusion.
    if report.pages == 0 or report.source_links == 0:
        raise SystemExit(
            f"generated {report.pages} pages with {report.source_links} source links — "
            "nothing downstream can be sound"
        )

    print(
        f"wrote {report.pages} pages to {_OUTPUT_DIR}: "
        f"{report.markers_restored} keyword-only markers restored, "
        f"{report.source_links} source links injected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
