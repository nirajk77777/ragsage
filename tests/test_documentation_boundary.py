"""The documentation site stays out of the Python package, and stays one site.

Five invariants, all structural, all easy to break by accident and impossible to
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

**There is only one documentation site.** The Sphinx site on Read the Docs is
being retired, and no redirects are possible from our side. A surviving link to
the old host therefore does not merely point somewhere stale — it points, from the
canonical source of truth about where the documentation lives, at a project on its
way to deletion. That is worse than no link. Deleting the project is the
maintainer's step and is not visible from here; what is asserted is the half that
is, which is that nothing in this repository sends a reader there.

**And the links into it name pages it has.** The repointed links are deep — they
name ``/failure-modes`` and ``/releasing``, not the homepage, because a reader
sent to a homepage to hunt for the page they were promised has been sent nowhere
useful. Nothing else in the project is watching them: the site's own gate walks
the links *inside* the built site, and these are links from outside it. So a page
renamed in ``website/content/`` breaks the README silently, which is the failure
this whole retirement was about, arriving by a different road.

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
from collections.abc import Iterator
from pathlib import Path

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

# Where the retired links lived: the three files a reader arrives through. Named
# rather than counted, because a count is satisfied by any fifty files in the
# tree, and these are the ones whose links cost something when they rot.
_PUBLISHED_ENTRY_POINTS = ("README.md", "CONTRIBUTING.md", "SECURITY.md")

# The hand-authored pages, which are also the routes the site serves: the path
# is the URL. Read by two of the claims below, so it sits with the constants
# rather than under either one's heading.
_SITE_CONTENT = _REPO_ROOT / "website/content"


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


def _readable_tracked_files() -> Iterator[tuple[str, str]]:
    """Every tracked file whose contents could carry a link, with those contents.

    A file that will not decode is skipped rather than reported: it cannot hold a
    link a reader follows, and the alternative is a guard that fails on the day
    someone commits a PNG.
    """
    for name in _tracked_files():
        try:
            yield name, (_REPO_ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError, FileNotFoundError):
            continue


def test_no_reference_to_read_the_docs_survives() -> None:
    """This repository points at one documentation site, and it is the new one.

    The old project's deletion is the maintainer's to perform and is not
    observable from here, so this asserts the half that is: whatever survives on
    the retired host, nothing here sends a reader to it.

    The lockfile is exempt, and only the lockfile: it records the resolved source
    URL of every wheel, and some of those are hosted on the retired host's domain
    by projects that have nothing to do with us. Those are package provenance,
    not documentation links, and they are not ours to rewrite.
    """
    offenders = []
    scanned = set()

    for name in _tracked_files():
        assert _RETIRED_CONFIG not in name, f"the retired build configuration is still here: {name}"

    for name, text in _readable_tracked_files():
        if name == "uv.lock" or not text.strip():
            continue

        scanned.add(name)
        if _RETIRED_HOST in text.lower():
            offenders.append(name)

    # Canaries. A search that walked nothing finds nothing — and neither does one
    # that walked the right files while they held nothing: "no retired link in
    # README.md" is trivially true of a README.md that is zero bytes long, which
    # is why an empty file is not counted as scanned. The named three are checked
    # by name because the count above is satisfied by any fifty files, including
    # fifty that are not these.
    assert len(scanned) > 50, f"only {len(scanned)} files were searched; the guard is not looking"

    unread = [name for name in _PUBLISHED_ENTRY_POINTS if name not in scanned]
    assert not unread, (
        "the files the retired links lived in were not scanned, or hold nothing to scan, so a "
        f"link surviving in one of them would not be found: {unread}"
    )

    assert not offenders, (
        "these files still point at the retired documentation site, which is being deleted and "
        f"cannot redirect: {offenders}"
    )


# ---------------------------------------------------------------------------- #
# The links into the site name pages it has
# ---------------------------------------------------------------------------- #

_DOCS_SITE = "https://ragsage-docs.nirajk.dev"

# A link into the site, capturing everything after the host. It ends at
# whitespace or at the punctuation that closes it in Markdown prose and in a
# badge's parentheses; a trailing sentence mark is trimmed after the fact rather
# than excluded here, since a `.` is legal inside a path and only suspicious at
# the end of one.
_DOCS_LINK = re.compile(re.escape(_DOCS_SITE) + r"([^\s)\"'`,<>\]]*)")
_SENTENCE_MARKS = ".,;:!?"

# The generated reference. None of it is committed — it is written into
# `website/content/api/` by a build — so this guard, which reads the repository,
# can answer for prose and cannot answer for those. Resolving them only when a
# local build happens to have left them on disk would be worse than not trying:
# the guard would be strict on the machine that just built the site and lax
# everywhere else, including CI.
#
# Matched as a route rather than as a string prefix, the way the site's own gate
# matches it: `/api-tour` is a prose page that merely starts with the same
# letters, and exempting it would be this guard quietly declining to check a page
# it can see.
_GENERATED_ROOT = "/api"


def _is_generated(route: str) -> bool:
    return route == _GENERATED_ROOT or route.startswith(f"{_GENERATED_ROOT}/")


def _published_links() -> dict[str, list[str]]:
    """Every link into the documentation site, by the route it names.

    The fragment is dropped — anchors are the site gate's business, and it checks
    them against the built pages. The trailing slash is emphatically *not*, and
    this is the whole reason the route is read literally: `trailingSlash` is off
    in the Next.js configuration, so the export writes `releasing.html`, and
    nginx answers `/releasing/` by trying `/releasing/.html` and giving up. A
    reader following a link with a slash on the end lands on a not-found, so this
    must report it rather than tidy it away.

    This file is scanned along with the rest, and `_DOCS_SITE` above is found in
    it. That yields `/`, which is a real page — so the guard's own source neither
    invents a failure nor hides one.
    """
    links: dict[str, list[str]] = {}
    for name, text in _readable_tracked_files():
        for match in _DOCS_LINK.finditer(text):
            path = match.group(1).split("#")[0].rstrip(_SENTENCE_MARKS)
            links.setdefault(path or "/", []).append(name)
    return links


def _prose_routes() -> set[str]:
    """The routes the committed prose provides, under the path-is-the-URL rule."""
    routes = set()
    for path in _SITE_CONTENT.rglob("*"):
        if path.suffix not in (".md", ".mdx"):
            continue
        relative = path.relative_to(_SITE_CONTENT)
        if relative.parts[0] == _GENERATED_ROOT.lstrip("/"):
            continue

        stem = relative.with_suffix("").as_posix().removesuffix("/index")
        routes.add("/" if stem == "index" else f"/{stem}")
    return routes


def test_every_published_link_names_a_page_the_site_has() -> None:
    """A link out of the repository is only as good as the page it lands on.

    Checked against the content tree rather than the live site, because a test
    that needs the network fails when someone else's machine is down, and a gate
    that fails for reasons unrelated to the change is a gate that gets disabled.
    The content tree is what determines the routes anyway: the path is the URL.

    What this cannot see is the reference, which is generated rather than
    committed. That boundary is stated rather than papered over — see
    `_GENERATED_ROOT`.
    """
    links = _published_links()
    routes = _prose_routes()

    # Canaries. Every claim below is over the routes the prose provides and the
    # links that name them, so neither set may be empty — and the claim that
    # matters is about the *deep* links, which a corpus of homepage links would
    # satisfy while establishing nothing.
    assert routes, f"no prose pages were found under {_SITE_CONTENT}, so nothing was resolved"
    assert links, f"no links to {_DOCS_SITE} were found, so nothing was checked"

    checkable = {route: sources for route, sources in links.items() if not _is_generated(route)}
    assert any(route != "/" for route in checkable), (
        "every published link points at the homepage, so nothing here establishes that a deep "
        "link lands on the page it names rather than on the front page"
    )

    dangling = {
        route: sorted(set(sources)) for route, sources in checkable.items() if route not in routes
    }
    assert not dangling, (
        "these files link to pages the site does not have, so a reader following them lands on "
        f"a not-found: {dangling}. The routes it does have are {sorted(routes)}"
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
