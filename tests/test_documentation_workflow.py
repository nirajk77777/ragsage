"""Continuous integration builds the documentation image, and builds it apart.

A contributor learns a documentation change is broken before merge, not after
deploy. That holds only while three separate things stay true of the workflow
files, and each of them is one plausible edit away from not being.

**The gate is the deploy.** The workflow builds the same image definition the
deployment host builds, from the same repository-root context — so a green pull
request cannot produce a broken deploy. The build carries no ``--target``: the
gate lives in the second stage, the host builds the third, and a target naming
either one is a build of something the host does not deploy. ``--target
api-reference`` would skip the gate outright and leave every build green.

**The gate is not the library's problem.** Generating the API reference pulls
Sphinx and its dependency tree, and building the site pulls npm's. Neither
belongs in the workflow that has to pass on every push, and a library-only pull
request must therefore get its result from the library gates exactly as it did
before the site existed — which also means that workflow keeps no path filter,
since a filter is how a gate stops running for the changes it was written for.

**The gate is not skipped for its own inputs.** The docs workflow does filter,
because it is the expensive one, and the filter is a deny-list so that forgetting
a path costs a redundant rebuild rather than an unrun gate. That holds only while
nothing the image consumes appears in it.

None of this is observable from a passing build: a workflow that stopped
gating passes faster, and passing faster is not a symptom anyone investigates.
The failure that these prevent is the one where CI is green and the deployed site
is broken, which is the situation the whole gate exists to replace.

Each assertion is paired with a canary, for the reason the documentation gate is:
"no step builds the documentation" is trivially true of a workflow that failed to
parse, and so is "no ignored path is an image input" of an empty input set.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

_DOCS_WORKFLOW = _REPO_ROOT / ".github/workflows/docs.yml"
_LIBRARY_WORKFLOW = _REPO_ROOT / ".github/workflows/ci.yml"

# The Dockerfile the deployment host is pointed at, and the context it builds
# from. Coolify derives the build context from its Base Directory, which the spec
# fixes at `/` because the generator has to read `src/` — so the context here is
# the repository root, and `tests/test_documentation_image.py` reads the same file
# to check what the image ships.
_DEPLOYED_DOCKERFILE = "website/Dockerfile"
_DEPLOYED_CONTEXT = "."

# What the documentation build drags in, named as it would appear in a step. The
# library gate type-checks `tools/` — the generator is custom code and is held to
# the library's bar — so `tools` is deliberately not among these: reading the
# generator is not running it.
_DOCUMENTATION_TOOLCHAIN = ("docker", "npm", "npx", "node ", "website/", "generate_api_docs")


def _workflow(path: Path) -> dict[str, Any]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"{path.name} did not parse as a workflow"
    return parsed


def _triggers(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The workflow's triggers, each mapped to its own configuration.

    ``on:`` is YAML 1.1 truthy, so PyYAML reads the key as the boolean ``True``,
    and a trigger with no body of its own (``pull_request:``) parses as ``None``.
    """
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict), f"expected named triggers, found {triggers!r}"
    return {name: body or {} for name, body in triggers.items()}


def _run_commands(workflow: dict[str, Any]) -> list[str]:
    """Every shell command the workflow runs, across all of its jobs.

    Steps that use an action rather than a shell carry no ``run`` and are dropped:
    what they do is the action's business, not this file's.
    """
    return [
        step["run"]
        for job in workflow.get("jobs", {}).values()
        for step in job.get("steps", [])
        if "run" in step
    ]


def test_the_workflow_builds_the_image_the_deployment_host_deploys() -> None:
    """One build, of the deployed definition, from the deployed context.

    Asserted over the command rather than over its effect, because its effect
    takes minutes and a Docker daemon. What it defends against is a diff that
    reads as a speed-up: a ``--target`` added to stop at the stage someone was
    debugging, a ``-f`` repointed at a Dockerfile written for CI alone. Both leave
    a green workflow named "Docs" building something the host will never deploy.
    """
    workflow = _workflow(_DOCS_WORKFLOW)

    builds = [command for command in _run_commands(workflow) if "docker build" in command]

    # Canary: every claim below is about a build command, so there must be one —
    # and exactly one, or a second build could be the one satisfying them.
    assert len(builds) == 1, (
        f"expected {_DOCS_WORKFLOW.name} to run exactly one image build, found {builds or 'none'}"
    )
    arguments = shlex.split(builds[0])[2:]

    assert "--target" not in arguments and not any(
        argument.startswith("--target=") for argument in arguments
    ), (
        f"the documentation build stops at a named stage: {builds[0]} — the deployment host "
        "builds the whole definition, and the stage holding the gate is not the stage it deploys"
    )

    named = [
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument in ("-f", "--file")
    ]
    assert named == [_DEPLOYED_DOCKERFILE], (
        f"the documentation build reads {named or 'no Dockerfile of its own'}, but the deployment "
        f"host is pointed at {_DEPLOYED_DOCKERFILE} — a green pull request would say nothing "
        "about the image that deploys"
    )
    assert (_REPO_ROOT / _DEPLOYED_DOCKERFILE).is_file(), (
        f"{_DEPLOYED_DOCKERFILE} does not exist, so the build is checked against nothing"
    )

    assert arguments[-1] == _DEPLOYED_CONTEXT, (
        f"the documentation build takes `{arguments[-1]}` as its context, not the repository root "
        "— the generator reads `src/`, and the deployment host builds from the root for that "
        "reason, so any narrower context is a build of a different thing"
    )

    triggers = _triggers(workflow)
    assert "pull_request" in triggers, (
        f"{_DOCS_WORKFLOW.name} does not run on pull requests, so a broken link is reported "
        f"after merge rather than before: {sorted(triggers)}"
    )


def test_the_library_gate_carries_no_documentation_toolchain() -> None:
    """The library workflow is what it was, and a library change gets its answer there.

    Two halves, and the second is the one that rots quietly. Keeping the
    documentation build out is what stops Sphinx and npm from slowing the gate
    that runs on every push; keeping the library workflow *unfiltered* is what
    stops a later path filter — added in the same spirit as the docs one, which is
    correct for the expensive workflow and wrong for this one — from skipping the
    library gates for the changes they exist to check.
    """
    workflow = _workflow(_LIBRARY_WORKFLOW)
    commands = _run_commands(workflow)

    # Canary: a workflow parsed down to no commands satisfies every claim about
    # what its commands must not contain.
    assert any("pytest" in command for command in commands), (
        f"{_LIBRARY_WORKFLOW.name} runs no test suite, so it did not parse as the library gate"
    )

    documentation = sorted(
        command for command in commands if any(tool in command for tool in _DOCUMENTATION_TOOLCHAIN)
    )
    assert not documentation, (
        "the library gate builds the documentation, so every push waits on Sphinx and npm: "
        f"{documentation}"
    )

    filtered = {
        name: body
        for name, body in _triggers(workflow).items()
        if "paths" in body or "paths-ignore" in body
    }
    assert not filtered, (
        "the library gate filters by path, so a change to a path nobody thought of merges "
        f"without ever being type-checked or tested: {filtered}"
    )


def _image_inputs() -> set[str]:
    """Top-level paths the documentation image copies out of the repository.

    Read from the Dockerfile rather than restated, so a new `COPY` is covered the
    day it lands instead of the day someone remembers this test. Stage-to-stage
    copies are skipped: they carry what an earlier stage built, not repository
    files, and their sources do not exist here.
    """
    inputs = set()
    dockerfile = (_REPO_ROOT / _DEPLOYED_DOCKERFILE).read_text(encoding="utf-8")
    for line in dockerfile.splitlines():
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
    triggers = _triggers(_workflow(_DOCS_WORKFLOW))
    ignore_lists = [body.get("paths-ignore", []) for body in triggers.values()]
    inputs = _image_inputs()

    # Canaries: an empty filter forbids nothing, and an empty input set is
    # satisfied by any filter at all.
    assert sorted(triggers) == ["pull_request", "push"], (
        f"expected the docs gate on push and on pull_request, found {sorted(triggers)}"
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
