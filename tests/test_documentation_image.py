"""What the documentation image ships, and what it does with a request.

The site is deployed as a single image: three stages build it, one stage runs.
Three of the claims below are about that last stage — the surface a reader
actually reaches — and the other two are about the stages that are thrown away:
the order they work in, and whether the gate among them runs at all. All five are
otherwise only observable by building the image, which takes minutes and a working
Docker daemon. These read the two definitions instead, so a change that would
break the deploy fails in the second it takes to run the suite.

They are not a substitute for the build. The build is the check, and CI runs it.
These catch the class of mistake that *looks* right in a diff: a base image
swapped for convenience, a document root that drifts from the directory the build
writes, a fallback that turns every dead link into the homepage.

**The published image contains no Python and no Node.** The first two stages exist
to run Sphinx and Next.js and are then discarded; what deploys is static files and
a web server. That is the whole reason the site is built this way, and it is one
``FROM`` away from silently ceasing to be true.

**A missing page is missing.** The deployment host offers a single-page-application
configuration that serves the homepage for anything it cannot find. It would pass
a smoke test and hide every broken link the site's own gate exists to catch — and
hide it only in production, where nobody is looking.

**The gate is reached.** A dead link fails the build only because a stage runs
the check and the published image is built from that stage. Either half alone is
worthless in a way no reader would see: a check placed after the copy that ships
the site still fails the build, but a stage nothing depends on is *skipped
entirely* by the builder — the gate would simply never run, and every build would
be green.

**Resolution is layered above source.** The last of the claims about the discarded
stages, and the one with no observable symptom at all: getting it wrong costs
minutes per deploy and breaks nothing, which is why nothing else would ever report
it.

Each assertion is paired with a canary, for the reason the documentation gate is:
"no stage installs Python" is trivially true of a Dockerfile that failed to parse,
and "every fallback ends in not-found" is trivially true of a configuration with
no fallbacks in it.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_DOCKERFILE = _REPO_ROOT / "website/Dockerfile"
_NGINX_CONF = _REPO_ROOT / "website/nginx.conf"
_NEXT_CONFIG = _REPO_ROOT / "website/next.config.mjs"

# The document root `nginx:alpine` ships, already holding its own `index.html`
# and `50x.html`. Copying the export over it merges rather than replaces, so the
# `50x.html` that has no counterpart survives — a stock nginx page served at a
# route no content file corresponds to, in a site whose gate asserts exactly the
# opposite about every other route.
_STOCK_DOCUMENT_ROOT = "/usr/share/nginx/html"

# Enough to catch the swap that would actually be made — a Node stage kept alive
# to run a JavaScript static server, or a Python stage kept to run Sphinx at boot.
_RUNTIME_IMAGES = ("python", "node", "nodejs", "bun", "deno")


def _stages() -> dict[str, list[str]]:
    """Every named build stage, mapped to the instructions under it, in order.

    Comments and blank lines are dropped; nothing else is interpreted. The
    Dockerfile uses no line continuations, and if it grows one this returns its
    halves separately — which would weaken an assertion below rather than break
    it, so it is left unhandled rather than half-handled.
    """
    stages: dict[str, list[str]] = {}
    current: list[str] = []

    for raw in _DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        named = re.match(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", line, re.IGNORECASE)
        if named:
            current = []
            stages[named.group(2) or named.group(1)] = current
        current.append(line)

    return stages


def _base_image(instructions: list[str]) -> str:
    """The image or stage a stage is built on: the argument of its own ``FROM``."""
    return instructions[0].split()[1]


def _numbered_copies(instructions: list[str]) -> list[tuple[int, list[str]]]:
    """Every ``COPY``, as an argument list with flags stripped, and its position."""
    return [
        (index, [word for word in line.split()[1:] if not word.startswith("--")])
        for index, line in enumerate(instructions)
        if line.upper().startswith("COPY ")
    ]


def _copies(instructions: list[str]) -> list[list[str]]:
    """``COPY`` instructions, as argument lists, with any flags stripped."""
    return [args for _, args in _numbered_copies(instructions)]


def _copy_of(instructions: list[str], source: str) -> int:
    """Where a stage copies ``source`` in, or ``-1`` if it never does.

    Matching the source exactly, rather than looking for a substring, is what
    keeps this from quietly answering a different question: ``website/`` appears
    in the path of a stage-to-stage copy too, and that one carries what an earlier
    stage generated rather than anything a contributor edits.
    """
    wanted = source.rstrip("/")
    for index, args in _numbered_copies(instructions):
        if any(arg.rstrip("/") == wanted for arg in args[:-1]):
            return index
    return -1


def _stages_copied_from(instructions: list[str]) -> set[str]:
    """The earlier stages a stage copies out of, read off its ``COPY --from=`` flags.

    This is the only thing that makes an earlier stage part of the build: the
    builder resolves what the target stage needs and skips everything else.
    """
    return {
        flag.removeprefix("--from=")
        for line in instructions
        if line.upper().startswith("COPY ")
        for flag in line.split()[1:]
        if flag.startswith("--from=")
    }


def _run_of(instructions: list[str], command: str) -> int:
    """Where a stage runs ``command``, or ``-1`` if it never does."""
    for index, line in enumerate(instructions):
        if line.upper().startswith("RUN ") and command in line:
            return index
    return -1


def test_the_published_image_carries_no_interpreter() -> None:
    """The stages that need a language runtime are the stages that are thrown away.

    Two things have to hold for that. The last stage must not be *built on* an
    earlier one — ``FROM site`` would inherit Node and every file it installed —
    and its own base must not be a language image. Either alone is satisfiable
    while shipping an interpreter.

    What crosses the seam is checked too, and it is the same claim from the other
    side: a stage copying ``/usr/local/bin/python`` out of an earlier stage carries
    no ``FROM`` a reader would notice.
    """
    stages = _stages()

    # Canaries: a single-stage build discards nothing, and a parse that found no
    # Python and no Node anywhere found nothing at all.
    assert len(stages) >= 3, f"expected the three-stage build, parsed {sorted(stages)}"
    toolchain = " ".join(_base_image(body) for body in list(stages.values())[:-1]).lower()
    assert "python" in toolchain and "node" in toolchain, (
        f"neither build stage runs Python or Node, so the Dockerfile did not parse: {toolchain}"
    )

    name, final = list(stages.items())[-1]
    base = _base_image(final)

    assert base not in stages, (
        f"the serving stage `{name}` is built on `{base}`, an earlier stage, so it inherits "
        "the toolchain that stage exists to contain"
    )
    assert not any(base.lower().startswith(image) for image in _RUNTIME_IMAGES), (
        f"the serving stage `{name}` is built on `{base}`, which ships a language runtime"
    )

    smuggled = [
        args for args in _copies(final) if any(re.search(r"bin/(python|node)", arg) for arg in args)
    ]
    assert not smuggled, f"the serving stage copies an interpreter into itself: {smuggled}"


def test_nginx_serves_exactly_the_directory_the_build_wrote() -> None:
    """The document root and the copy destination are one decision in two files.

    Drift between them is not a subtle failure — it is every page at once, and it
    cannot happen locally, because the two are only ever brought together inside
    the image.

    The root is also deliberately *not* the one ``nginx:alpine`` ships. Copying
    over that directory leaves whatever it already held sitting alongside the
    site: stock nginx pages served at routes the site's own gate would reject, and
    which the gate cannot see, because it runs over the export rather than over
    the image.
    """
    conf = _NGINX_CONF.read_text(encoding="utf-8")
    roots = re.findall(r"^\s*root\s+(\S+?);", conf, re.MULTILINE)

    # Canary: a configuration declaring no root proves nothing about where it serves from.
    assert len(roots) == 1, f"expected one document root, found {roots}"
    (root,) = roots

    final = list(_stages().values())[-1]
    destinations = [args[-1].rstrip("/") for args in _copies(final) if args[0].endswith("/")]

    # Canary: the claim is about a directory that gets copied in, so one must be.
    assert destinations, "the serving stage copies no directory in, so it serves nothing"

    assert root.rstrip("/") in destinations, (
        f"nginx serves from {root}, but the image writes the site to {destinations} — "
        "the site would be built and then not served"
    )
    assert root.rstrip("/") != _STOCK_DOCUMENT_ROOT, (
        f"the site is served from {_STOCK_DOCUMENT_ROOT}, the document root the base image "
        "already populates, so its stock pages are served as part of the site"
    )


def test_a_page_that_does_not_exist_is_not_the_homepage() -> None:
    """A miss must stay a miss, all the way out to the reader.

    Two ways to lose that, and the configuration has to refuse both. A
    ``try_files`` ending in ``/index.html`` rather than ``=404`` answers every
    dead link with the front page — the deployment host offers precisely that as a
    preset, and it would leave the site looking healthy while its links rotted. An
    ``error_page`` that rewrites the status does the same thing one layer up: a
    branded not-found page is welcome, a branded not-found page returning ``200``
    is the same lie in better clothes.

    The ``$uri.html`` fallback is the other half of ``trailingSlash`` being off in
    the Next.js configuration, which is why that file is read here. The export
    writes ``quickstart.html``; change either half alone and every page 404s.
    """
    conf = _NGINX_CONF.read_text(encoding="utf-8")
    fallbacks = re.findall(r"^\s*try_files\s+(.+?);", conf, re.MULTILINE)

    # Canary: a configuration with no fallbacks satisfies every claim about them.
    assert fallbacks, "nginx declares no try_files fallback, so nothing here was checked"

    for fallback in fallbacks:
        assert fallback.split()[-1] == "=404", (
            f"the fallback `try_files {fallback}` does not end in not-found, so a request for a "
            "page that does not exist is answered with one that does"
        )

    assert any("$uri.html" in fallback for fallback in fallbacks), (
        "nginx never tries `$uri.html`, but the static export writes `quickstart.html` rather "
        "than `quickstart/index.html`, so every page would 404"
    )
    assert not re.search(r"trailingSlash\s*:\s*true", _NEXT_CONFIG.read_text(encoding="utf-8")), (
        "the static export was switched to trailing slashes, which writes "
        "`quickstart/index.html` — nginx's `$uri.html` fallback no longer matches it"
    )

    not_found = re.findall(r"^\s*error_page\s+404\s+(=\S+\s+)?(\S+?);", conf, re.MULTILINE)
    assert not_found, (
        "nginx serves its own stock not-found page, so a reader who mistypes a URL sees a bare "
        "server error rather than the site's — while the export's 404.html ships unused"
    )

    for rewrite, page in not_found:
        assert not rewrite, (
            f"error_page rewrites the status of {page} to `{rewrite.strip()}`, so a missing page "
            "is reported as a page that exists"
        )
        assert page == "/404.html", (
            f"error_page serves {page}, but the static export writes its not-found page to "
            "/404.html — nginx would fail to resolve this against the document root"
        )


def test_a_prose_edit_does_not_re_resolve_the_toolchain() -> None:
    """Dependency resolution is layered above source, so editing a page is cheap.

    Sphinx and its dependency tree are minutes of a deploy, and npm's are more.
    Both are pinned by a lockfile, and neither is changed by editing the thing the
    stage exists to read — so both belong in a layer only that lockfile
    invalidates.

    The input to watch is not prose. It is ``src/``: the API reference is
    generated from its docstrings, which makes library source the Python stage's
    most-edited input by far, and putting it in ahead of the resolve would re-run
    that resolve on every docstring edit while still leaving the narrower claim
    about prose perfectly true.

    So each stage is checked against the source it actually reads, and the Python
    stage is checked twice: prose must not enter it at all, and what does enter it
    must arrive after resolution rather than before.
    """
    api_reference, site, *_ = _stages().values()

    resolve = _run_of(api_reference, "uv sync")
    source = _copy_of(api_reference, "src/")
    install = _run_of(site, "npm ci")
    application = _copy_of(site, "website/")

    # Canaries: an ordering claim proves nothing about steps that are not there.
    assert resolve >= 0, "the first stage no longer resolves the Python toolchain"
    assert source >= 0, "the first stage no longer copies `src/` in, so it generates nothing"
    assert install >= 0, "the second stage no longer installs from the lockfile"
    assert application >= 0, "the second stage no longer copies the site in, so it builds nothing"

    assert resolve < source, (
        "`src/` is copied in before the toolchain is resolved, so every docstring edit "
        "re-resolves Sphinx and its dependency tree — and the API reference is generated "
        "from those docstrings, so that is this stage's most-edited input"
    )
    assert install < application, (
        "the site is copied in before `npm ci`, so every prose edit reinstalls node_modules"
    )

    prose = [
        args for args in _copies(api_reference) if any(arg.startswith("website/") for arg in args)
    ]
    assert not prose, (
        "the stage that resolves the Python toolchain reads the documentation site, so editing "
        f"a page re-resolves Sphinx and its dependency tree: {prose}"
    )


def test_a_dead_link_cannot_reach_the_published_image() -> None:
    """The gate runs over the export, and the image is built from the gated stage.

    Two independent ways for this to stop being true, and neither shows up
    anywhere a reader or a reviewer would look.

    The first is ordering: the gate reads ``out/``, so it has to run after the
    build that writes it. A check that runs first passes over the previous
    export, or over nothing.

    The second is subtler and is the reason this test exists at all. Docker builds
    only the stages the target needs. Move the gate into a stage nothing copies
    from — a fourth stage added for tidiness, say — and it is not that the gate
    runs late, it is that it never runs. The build stays green over a site with
    every link dead, and the only symptom is that it got faster.

    The gate's own tests are held to the same standard, and for the same reason
    the gate holds the site to it: an assertion nobody has watched fail is an
    assertion nobody should believe. They run here rather than in this suite
    because they measure the gate against the real export, which only exists
    inside this stage.
    """
    stages = _stages()
    final = list(stages.values())[-1]

    gated = [
        name
        for name, body in stages.items()
        if _run_of(body, "npm run check") >= 0 and _run_of(body, "npm test") >= 0
    ]

    # Canary: an ordering-and-reachability claim about a check that is not in the
    # Dockerfile at all is satisfied by its absence.
    assert len(gated) == 1, (
        "expected exactly one stage to run the site gate and its tests, found "
        f"{gated or 'none'} — a dead link would reach the published image unreported"
    )
    (gate_stage,) = gated

    body = stages[gate_stage]
    export = _run_of(body, "npm run build")
    assert export >= 0, f"`{gate_stage}` gates a static export it never builds"
    assert export < _run_of(body, "npm run check"), (
        f"`{gate_stage}` runs the gate before the build that writes the export it reads, so it "
        "checks the previous build or nothing at all"
    )
    assert export < _run_of(body, "npm test"), (
        f"`{gate_stage}` runs the gate's own tests before the export they mutate exists"
    )

    # Canary: a serving stage that copies from nothing is not a multi-stage build,
    # and the reachability claim below would hold over an empty set.
    reached = _stages_copied_from(final)
    assert reached, "the serving stage copies out of no earlier stage, so it serves nothing"

    assert gate_stage in reached, (
        f"the serving stage is built from {sorted(reached)}, not from `{gate_stage}` — Docker "
        f"builds only the stages the target needs, so `{gate_stage}` and the gate in it are "
        "skipped entirely and every build is green"
    )
