"""The documentation site stays out of the Python package, and stays one site.

Four invariants, all structural, all easy to break by accident and impossible to
notice once broken.

**The site is not part of the distribution.** ``website/`` is a Next.js
application: TypeScript, a lockfile, and — the moment anyone runs the dev server
— a ``node_modules`` tree of several hundred megabytes. ``ragsage`` is a Python
library, and its sdist is what someone downloads to build and test it from
source. Hatchling's sdist target is *deny-list* based: it ships everything not
explicitly named, so a new top-level directory is included by default and the
exclusion is the deliberate act. This is the guard on that act.

Both directions are asserted, because a deny-list fails both ways. Excluding too
little publishes a JavaScript application to the package index; excluding too
much hands someone a download they cannot build or test, and that failure is the
quieter of the two — nobody who has the repository ever sees it.

**The repository is a Python project.** GitHub decides what a repository *is* by
counting lines, and the site's TypeScript tree outweighs ``src/``. The count is
not cosmetic: it drives the language shown on the repository, the search filters
that surface it, and what a first-time reader assumes they are looking at. A
``.gitattributes`` entry marking the site as documentation keeps the count
honest, and the entry is one deleted line away from not existing.

**There is only one documentation site.** The Sphinx site on Read the Docs was
retired, and no redirects are possible from our side. A surviving link to the old
host therefore does not merely point somewhere stale — it points at a project that
no longer exists, from a repository that is the canonical source of truth about
where the documentation lives. That is worse than no link.

**The release notes have one home, and it is not here.** Their canonical home is
the GitHub releases page, which is generated from the tag that publishes the
package. A copy on the documentation site would be a second home that nothing
updates: the release that adds a note to GitHub does not rebuild this site, so the
copy is stale from the release after the one that created it — and it is the copy
a reader lands on, because it is the one with a documentation URL.

Each assertion has a canary, for the reason the whole documentation gate does:
"no forbidden file in the sdist" is trivially true of an sdist that failed to
build, and "no forbidden string in the repository" is trivially true of a search
that matched no files.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The application itself. This is the one thing `[tool.hatch.build.targets.sdist]`
# has to name, and the only one of the three groups here whose absence is our own
# doing.
_FORBIDDEN_SDIST_PREFIXES = ("website/",)

# Installed dependencies and build output, wherever they sit. Today they sit only
# under `website/` — `website/.gitignore` anchors both patterns with a leading
# slash — so the exclusion above already covers them and this assertion cannot be
# the one that fails first. It is deliberately the outer boundary rather than the
# guard on today's mechanism: it is what answers "the artifact ships no dependency
# directory" for a *second* JavaScript toolchain, arriving somewhere the exclusion
# list has no entry for, which is the shape this repository has grown once already.
_FORBIDDEN_SDIST_COMPONENTS = ("node_modules", ".next")

# What someone who downloads the sdist to build and test from source needs to
# find in it: the library, its tests, and the four runnable examples the README
# sends a reader to.
_REQUIRED_SDIST_PREFIXES = ("src/ragsage/", "tests/", "examples/")

# Concatenated at runtime so this guard does not match itself — the formatter
# joins adjacent string literals, so that spelling would not survive `ruff format`.
#
# The needle is the *host*, not the product name: prose that says "Read the Docs
# was retired" is a true statement about history and should not fail the build,
# while a URL anyone could click should.
_RETIRED_HOST = "readthedocs" + ".io"
_RETIRED_CONFIG = ".readthedocs" + ".yaml"


def _build_sdist(destination: Path) -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "sdist", "-d", str(destination)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:  # pragma: no cover - a build failure is the message
        raise AssertionError(f"could not build an sdist to inspect:\n{result.stderr}")

    built = sorted(destination.glob("*.tar.gz"))
    assert len(built) == 1, f"expected exactly one sdist, found {[p.name for p in built]}"
    return built[0]


def test_the_sdist_contains_no_documentation_application(tmp_path: Path) -> None:
    """A Python artifact ships Python, not a JavaScript app.

    Asserted over a freshly built sdist rather than over the exclusion list,
    because the exclusion list is not the invariant — it is one of the mechanisms
    that happen to produce it today, and the other two (hatchling's VCS-ignore
    handling, and `.gitignore` itself) are not ours. Reading the list back would
    only restate what `pyproject.toml` already says.

    The required half runs first for the same reason: an sdist that excluded
    `examples/` along with the site satisfies every "no JavaScript here"
    assertion below, so a green from them means nothing until something has
    established the artifact is the one we meant to build.
    """
    with tarfile.open(_build_sdist(tmp_path)) as archive:
        # Strip the `ragsage-0.1.0/` prefix every sdist member carries.
        members = [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]

    # Canaries: an sdist that shipped nothing would satisfy every "not present"
    # assertion below, and a failed build is the likeliest way to get one.
    missing = [
        prefix
        for prefix in _REQUIRED_SDIST_PREFIXES
        if not any(name.startswith(prefix) for name in members)
    ]
    assert not missing, (
        "the sdist is missing what someone builds and tests the library from, so it "
        f"proves nothing about what it excludes: {missing}"
    )

    smuggled = [
        name
        for name in members
        if any(name.startswith(prefix) for prefix in _FORBIDDEN_SDIST_PREFIXES)
    ]
    assert not smuggled, f"the sdist ships the documentation application: {smuggled[:10]}"

    vendored = [
        name
        for name in members
        if any(part in _FORBIDDEN_SDIST_COMPONENTS for part in name.split("/"))
    ]
    assert not vendored, f"the sdist ships installed dependencies or build output: {vendored[:10]}"


def _tracked_files() -> list[str]:
    """Every file git tracks — which is exactly what "in the repository" means.

    Asking git rather than walking the tree is not a shortcut. It excludes build
    output, virtual environments, ``node_modules`` and agent worktrees for free,
    and it cannot drift out of step with ``.gitignore`` the way a second,
    hand-maintained exclusion list would.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [name for name in result.stdout.split("\0") if name]


def test_no_reference_to_read_the_docs_survives() -> None:
    """The old site is deleted, so a link to it is a link to a 404.

    The lockfile is exempt, and only the lockfile: it records the resolved source
    URL of every wheel, and some of those are hosted on the retired host's domain
    by projects that have nothing to do with us. Those are package provenance,
    not documentation links, and they are not ours to rewrite.
    """
    offenders = []
    searched = 0

    for name in _tracked_files():
        assert _RETIRED_CONFIG not in name, f"the retired build configuration is still here: {name}"
        if name == "uv.lock":
            continue
        try:
            text = (_REPO_ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError, FileNotFoundError):
            continue  # a binary file cannot carry a link a reader will follow

        searched += 1
        if _RETIRED_HOST in text.lower():
            offenders.append(name)

    # Canary: a search that walked nothing finds nothing.
    assert searched > 50, f"only {searched} files were searched; the guard is not looking"

    assert not offenders, (
        "these files still point at the retired documentation site, which no longer exists "
        f"and cannot redirect: {offenders}"
    )


# ---------------------------------------------------------------------------- #
# The repository still reads as a Python project
# ---------------------------------------------------------------------------- #

# What `git check-attr` reports for a path an attribute applies to. The other
# three answers — `unset`, `unspecified`, and a custom value — all mean linguist
# counts the file.
_ATTRIBUTE_APPLIES = "set"


def _documentation_attribute_by_path(paths: list[str]) -> dict[str, str]:
    """Ask git what `linguist-documentation` resolves to for each path.

    Asking git, rather than reading `.gitattributes` and matching the patterns
    here, is what makes this a guard rather than a second implementation of
    attribute resolution. Pattern precedence, a later line overriding an earlier
    one, an ancestor `.gitattributes`, an entry that silently matches nothing —
    all of it is answered the way the thing doing the counting will answer it.
    """
    result = subprocess.run(
        ["git", "check-attr", "--stdin", "-z", "linguist-documentation"],
        cwd=_REPO_ROOT,
        input="".join(f"{path}\0" for path in paths),
        capture_output=True,
        text=True,
        check=True,
    )
    # NUL-separated triples: path, attribute, value, with a trailing separator.
    fields = result.stdout.split("\0")[:-1]
    return {fields[index]: fields[index + 2] for index in range(0, len(fields), 3)}


def test_the_repository_still_reads_as_a_python_project() -> None:
    """Asserted over resolved attributes, and in both directions.

    Over resolved attributes because the fix is one line of `.gitattributes` and
    the failure is what git makes of it: a pattern that matches nothing looks
    exactly like a pattern that matches everything it should, right up until
    something counts the files.

    Both directions because marking the library as documentation along with the
    site is the same misclassification arrived at from the other side — the
    repository is then detected from everything except its own source — and an
    over-broad pattern (`**` rather than `website/**`) is the plausible way to
    write it.
    """
    tracked = _tracked_files()
    site = [name for name in tracked if name.startswith("website/")]
    library = [name for name in tracked if name.startswith("src/ragsage/")]

    # Canaries: "every site file is marked" is trivially true of no site files,
    # and so is "no library file is marked" of no library files.
    assert site, "no tracked files under website/, so nothing was classified"
    assert library, "no tracked files under src/ragsage/, so nothing was classified"

    attributes = _documentation_attribute_by_path(site + library)

    unmarked = [name for name in site if attributes.get(name) != _ATTRIBUTE_APPLIES]
    assert not unmarked, (
        "these documentation-site files count towards the repository's detected "
        f"language, which would reclassify a Python library as JavaScript: {unmarked[:10]}"
    )

    miscounted = [name for name in library if attributes.get(name) == _ATTRIBUTE_APPLIES]
    assert not miscounted, (
        "these library files are marked as documentation, so the language the repository "
        f"is detected as is counted from everything but its own source: {miscounted[:10]}"
    )


# ---------------------------------------------------------------------------- #
# The release notes are linked, not copied
# ---------------------------------------------------------------------------- #

_RELEASE_NOTES = _REPO_ROOT / "docs/release-notes"
_SITE_CONTENT = _REPO_ROOT / "website/content"
_RELEASES_PAGE = "github.com/nirajk77777/ragsage/releases"


def test_the_site_links_the_release_notes_rather_than_serving_them() -> None:
    """One home for the release notes, and the site points at it.

    Both halves matter, and only together. Excluding the notes without naming
    where they went leaves a reader on a documentation site with no way to find
    out what changed in a version — which is the pressure that would put a copy
    here in the first place.

    Structural, over what the repository holds: prose is authored by hand, so the
    way this breaks is somebody copying a note in, not the build emitting one.
    """
    notes = sorted(_RELEASE_NOTES.glob("v*.md"))
    pages = sorted(_SITE_CONTENT.rglob("*.mdx"))

    # Canaries. "The site serves no release note" is trivially true when none has
    # been written, and "some page links to the releases page" cannot be believed
    # of a search that found no pages to read.
    assert notes, f"no release notes were found under {_RELEASE_NOTES}, so nothing is excluded"
    assert pages, f"no prose pages were found under {_SITE_CONTENT}, so nothing was searched"

    # A copied note arrives one of two ways: the directory comes with it, or the
    # file keeps the version-stamped name it had. Both are read *inside the content
    # tree* — an absolute path would answer for the checkout's own ancestry, and a
    # clone under a directory called `release-notes` would fail this for no reason.
    #
    # The name pattern mirrors the notes themselves rather than anything starting
    # with a `v`: `releasing.mdx` is the guide to *cutting* a release and belongs
    # here, and so would a `v2-migration.mdx` that is prose about a version rather
    # than the notes for one.
    served = [
        page
        for page in pages
        if "release-notes" in page.relative_to(_SITE_CONTENT).parts
        or re.fullmatch(r"v\d+(?:\.\d+)*", page.stem)
    ]
    assert not served, (
        "these pages would serve the release notes from the documentation site, whose "
        f"canonical home is the releases page: {[str(page) for page in served]}"
    )

    linked = [page for page in pages if _RELEASES_PAGE in page.read_text(encoding="utf-8")]
    assert linked, (
        f"no page links to {_RELEASES_PAGE}, so the release notes were excluded from the "
        "site without the site saying where they went"
    )


# ---------------------------------------------------------------------------- #
# The docs gate cannot be skipped for a change that reaches it
# ---------------------------------------------------------------------------- #

_DOCS_WORKFLOW = _REPO_ROOT / ".github/workflows/docs.yml"
_DOCKERFILE = _REPO_ROOT / "website/Dockerfile"


def _docs_ignore_lists() -> list[list[str]]:
    """The `paths-ignore` list under every trigger in the docs workflow."""
    workflow = yaml.safe_load(_DOCS_WORKFLOW.read_text(encoding="utf-8"))
    # `on:` is YAML 1.1 truthy, so PyYAML reads the key as the boolean `True`.
    triggers = workflow.get("on", workflow.get(True))
    # A trigger with no body at all (`pull_request:`) parses as `None`.
    return [(trigger or {}).get("paths-ignore", []) for trigger in triggers.values()]


def _image_inputs() -> set[str]:
    """Top-level paths the documentation image copies out of the repository.

    Read from the Dockerfile rather than restated, so a new `COPY` is covered the
    day it lands instead of the day someone remembers this test. Stage-to-stage
    copies are skipped: they carry what an earlier stage built, not repository
    files, and their sources do not exist here.
    """
    inputs = set()
    for line in _DOCKERFILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY ") or "--from=" in stripped:
            continue
        # `COPY <src>... <dest>` — every argument but the last is a source.
        for source in stripped.split()[1:-1]:
            inputs.add(source.rstrip("/").split("/")[0])
    return inputs


def test_the_docs_gate_is_not_skipped_for_its_own_inputs() -> None:
    """A deny-list that names an input silently stops checking it.

    The docs workflow skips the image build for changes that cannot reach it. The
    filter is a deny-list precisely so that forgetting a path costs a redundant
    rebuild rather than an unrun gate — but that only holds while nothing the
    image actually consumes appears in it. `src/` is the one to watch: the API
    reference is generated from its docstrings, so it looks like library code and
    is in fact the gate's principal input.

    A mismatch between the triggers is the same failure wearing a disguise: a path
    ignored on push but built on pull request merely moves the rebuild, while the
    reverse lets a broken docs build merge unchecked.
    """
    ignore_lists = _docs_ignore_lists()
    inputs = _image_inputs()

    # Canaries: an empty filter forbids nothing, and an empty input set is
    # satisfied by any filter at all.
    assert len(ignore_lists) == 2, (
        f"expected a filter on push and on pull_request, found {len(ignore_lists)}"
    )
    assert all(ignore_lists), "the docs workflow declares an empty paths-ignore list"
    assert {"src", "website", "api-src", "tools"} <= inputs, (
        f"the Dockerfile no longer copies the API reference inputs; found {sorted(inputs)}"
    )

    first, *rest = ignore_lists
    assert all(other == first for other in rest), (
        f"the docs workflow's triggers filter different paths: {ignore_lists}"
    )

    smuggled = sorted(pattern for pattern in first if pattern.rstrip("/*").rstrip("/") in inputs)
    assert not smuggled, (
        "these paths are copied into the documentation image but excluded from the "
        f"workflow that builds it, so changing them would skip the gate: {smuggled}"
    )
