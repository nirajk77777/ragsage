"""The documentation site stays out of the Python package, and stays one site.

Two invariants, both structural, both easy to break by accident and impossible to
notice once broken.

**The site is not part of the distribution.** ``website/`` is a Next.js
application: TypeScript, a lockfile, and — the moment anyone runs the dev server
— a ``node_modules`` tree of several hundred megabytes. ``ragsage`` is a Python
library, and its sdist is what someone downloads to build and test it from
source. Hatchling's sdist target is *deny-list* based: it ships everything not
explicitly named, so a new top-level directory is included by default and the
exclusion is the deliberate act. This is the guard on that act.

**There is only one documentation site.** The Sphinx site on Read the Docs was
retired, and no redirects are possible from our side. A surviving link to the old
host therefore does not merely point somewhere stale — it points at a project that
no longer exists, from a repository that is the canonical source of truth about
where the documentation lives. That is worse than no link.

Each assertion has a canary, for the reason the whole documentation gate does:
"no forbidden file in the sdist" is trivially true of an sdist that failed to
build, and "no forbidden string in the repository" is trivially true of a search
that matched no files.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The application itself, and the two directories it grows at rest. `node_modules`
# and `.next` are gitignored and hatchling honours VCS ignore, so they need no
# exclusion of their own — but this test does not care *how* they are kept out,
# only that they are. That is the point of asserting over the built artifact.
_FORBIDDEN_SDIST_PREFIXES = ("website/",)

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
    """A Python artifact ships Python, not a JavaScript app."""
    with tarfile.open(_build_sdist(tmp_path)) as archive:
        # Strip the `ragsage-0.1.0/` prefix every sdist member carries.
        members = [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]

    # Canary: an sdist that shipped nothing would satisfy every assertion below.
    assert any(name.startswith("src/ragsage/") for name in members), (
        "the sdist contains no library source, so it proves nothing about what it excludes"
    )
    assert any(name.startswith("tests/") for name in members), (
        "the sdist contains no tests, so the exclusion list is excluding too much"
    )

    smuggled = [
        name
        for name in members
        if any(name.startswith(prefix) for prefix in _FORBIDDEN_SDIST_PREFIXES)
    ]
    assert not smuggled, f"the sdist ships the documentation application: {smuggled[:10]}"


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
