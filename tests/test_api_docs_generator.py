"""The two post-passes that correct ``sphinx-markdown-builder``'s known defects.

These are the riskiest custom code in the documentation pipeline, and the only
part of it that is ours. The generator's *output* is checked far more thoroughly
than this — the built site is walked link by link before it can deploy — but that
check costs a full multi-stage image build, which is a multi-minute cycle on
exactly the code most likely to need another attempt. Both passes are pure
functions over the page text, so they get a fast seam of their own — as does the
one thing they call out to, which turns a dotted name into a line of source.

**Defect one: the keyword-only marker is eaten.** Sphinx's Python domain wraps
PEP 3102's ``*`` separator in a docutils ``abbreviation`` node (for the tooltip
that explains it). The Markdown builder has no handler for that node type, warns
once, and drops it — so ``chunk(self, document, pages, *, size, overlap)`` is
emitted as ``chunk(document, pages, , size, overlap)``, and a reader cannot tell
which arguments are keyword-only. There are 35 of these across the six generated
pages, in two shapes: 27 mid-signature (``, ,``) and 8 where the marker is the
first parameter (``(,``). Neither string can occur in a valid Python signature,
which is what makes the restoration unambiguous rather than a guess.

The restoration tests mostly assert an artefact is *gone*, which is trivially true
of a fixture that never carried one — so they share a canary establishing that the
fixtures are still corrupt to begin with. The count the pass reports gets the same
treatment, because it is what the built-site gate later checks survived: a counter
stuck at zero would leave that gate asserting nothing over nothing.

**Defect two: every source link is dropped.** ``sphinx.ext.viewcode`` puts a
``[source]`` link on all 222 documented objects in the HTML build; the Markdown
builder never visits the nodes that carry them, so all 222 vanish.
``sphinx.ext.linkcode`` was tried as the remedy and is dropped the same way. The
links are therefore injected here instead, off the dotted name in the anchor that
the builder emits above every generated heading.

The injection has two halves and they are tested apart. Placing the link is pure
text, so it is exercised against a stub resolver. *Resolving* a dotted name to a
file and a line is the half that can be quietly wrong — a plausible URL that
lands on the wrong line reads as a working link — so those tests take the URL the
generator produced, open the checked-in file at the line it names, and assert the
object is declared there.

**The warning gate.** Sphinx is still the only thing that resolves the 202
cross-reference roles in the docstrings, and a role that resolves nowhere is a
warning rather than an error — the build succeeds and emits the role as plain
text. So the run is gated on its own warnings, and the gate is an allow-list of
one known builder defect rather than ``-W``, because that defect warns on every
real run. An allow-list applied to the wrong file, or applied to the whole text
instead of line by line, is green forever over a reference full of dead
references. That is the same vacuity the built-site gate is arranged against, one
stage earlier, so it is checked the same way.

**What the run writes, and in what order.** The last two sections leave the
transforms and take the page-writing step whole, against a stand-in for the raw
Sphinx build. Two of this pipeline's decisions live there and nowhere else: that
the generator owns one directory and cannot reach a hand-written page outside it,
and that the five layers are navigated in the order the engine is assembled in.
Both are invisible to the built-site gate, which checks that navigation and
content agree — not who wrote them, and not what order they are in.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest
from tools import generate_api_docs
from tools.generate_api_docs import (
    _DESCRIPTIONS,
    _EXPECTED_WARNINGS,
    _OUTPUT_DIR,
    _check_warnings,
    _run_sphinx,
    _write_pages,
    add_frontmatter,
    count_eaten_markers,
    inject_source_links,
    report_unresolved,
    resolve_source_url,
    restore_keyword_only_markers,
)

# ---------------------------------------------------------------------------- #
# Defect one: the eaten keyword-only marker
# ---------------------------------------------------------------------------- #

# Trimmed from the real generated `adapters.md`. The links are left in because
# they are what a signature line actually looks like — the pass has to survive
# commas inside link targets and inside subscripted generics without being
# fooled by them.
_EATEN_MID = (
    "#### chunk(document: [Document](models#ragsage.models.Document), "
    "pages: [Sequence](https://docs.python.org/3/x#Sequence)[[Page](models#ragsage.models.Page)], , "
    "size: [int](https://docs.python.org/3/y#int), overlap: [int](https://docs.python.org/3/y#int)) "
    "→ [Sequence](https://docs.python.org/3/x#Sequence)[[Chunk](models#ragsage.models.Chunk)]"
)

_EATEN_LEADING = (
    "### *class* ragsage.query.QueryEngine(, embedder: [Embedder](ports#ragsage.ports.Embedder), "
    "tracer: [Tracer](ports#ragsage.ports.Tracer) | [None](https://docs.python.org/3/z#None) = None)"
)


def test_the_fixtures_carry_the_artefact() -> None:
    # The canary. Every test below asserts an artefact is *absent* after the pass,
    # which is trivially true of a fixture that never carried one — and these are
    # trimmed from real generated pages, so a re-trim from a page the generator had
    # already corrected would silence the lot while the pass did nothing at all.
    assert ", , " in _EATEN_MID, (
        "the mid-signature fixture is no longer corrupt, so the tests that assert the "
        "artefact is gone prove nothing about the pass"
    )
    assert "(, " in _EATEN_LEADING, (
        "the leading-marker fixture is no longer corrupt, so the tests that assert the "
        "artefact is gone prove nothing about the pass"
    )


def test_restores_a_marker_eaten_mid_signature() -> None:
    restored = restore_keyword_only_markers(_EATEN_MID)

    assert ", *, size:" in restored
    assert ", , " not in restored


def test_restores_a_marker_eaten_as_the_first_parameter() -> None:
    """``def f(*, a, b)`` loses its marker against the opening bracket, not a comma."""
    restored = restore_keyword_only_markers(_EATEN_LEADING)

    assert "QueryEngine(*, embedder:" in restored
    assert "(, " not in restored


def test_restores_every_marker_in_a_signature_with_two() -> None:
    line = "#### f(a, , b, , c)"

    assert restore_keyword_only_markers(line) == "#### f(a, *, b, *, c)"


def test_leaves_body_text_alone() -> None:
    """Only signature headings are rewritten.

    A docstring is free to contain ``, ,`` — in a prose aside, in a code block, in
    a quoted example of malformed input. The artefact is unambiguous *within a
    signature*; a blanket search-and-replace over the whole page would corrupt
    prose to fix headings, which is a bad trade in a file this size.
    """
    body = "The parser emits `, ,` for an empty cell, and (, is not a valid prefix."

    assert restore_keyword_only_markers(body) == body


def test_leaves_a_correct_signature_alone_and_is_idempotent() -> None:
    correct = "#### chunk(document: Document, *, size: int, overlap: int) → Sequence[Chunk]"

    assert restore_keyword_only_markers(correct) == correct
    assert restore_keyword_only_markers(restore_keyword_only_markers(_EATEN_MID)) == (
        restore_keyword_only_markers(_EATEN_MID)
    )


def test_leaves_var_positional_arguments_alone() -> None:
    """``*args`` survives the builder intact; only the bare separator is eaten."""
    line = "### ragsage.query.reciprocal_rank_fusion(\\*ranked_lists: list[ScoredChunk])"

    assert restore_keyword_only_markers(line) == line


def test_restores_across_a_whole_page() -> None:
    page = "\n".join([_EATEN_MID, "", "Some prose.", "", _EATEN_LEADING, ""])

    restored = restore_keyword_only_markers(page)

    # Both shapes, positively: an absence alone would not distinguish a marker put
    # back from a comma pair quietly deleted.
    assert restored.count(", *, ") == 1
    assert restored.count("(*, ") == 1
    assert ", , " not in restored
    assert "(, " not in restored
    assert "Some prose." in restored


def test_counts_the_markers_it_would_restore() -> None:
    """The count is the gate's canary, so it cannot be left to trust.

    ``count_eaten_markers`` is what the manifest reports and what
    ``check-built-site.mjs`` asserts survived into the built pages. That check is
    ``restored >= manifest.markersRestored`` — so a counter stuck at zero passes
    it vacuously, and takes the gate's "no signature still carries the artefact"
    assertion down with it, since both are then ranging over nothing.
    """
    page = "\n".join([_EATEN_MID, "", "Prose with `, ,` in it.", "", _EATEN_LEADING, ""])

    assert count_eaten_markers(page) == 2
    assert count_eaten_markers(restore_keyword_only_markers(page)) == 0


# ---------------------------------------------------------------------------- #
# Defect two: the dropped source links
# ---------------------------------------------------------------------------- #

_URL = "https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/query.py#L42"


def _resolve_only_query_engine(dotted: str) -> str | None:
    return _URL if dotted == "ragsage.query.QueryEngine" else None


def test_injects_a_source_link_below_a_documented_heading() -> None:
    page = '<a id="ragsage.query.QueryEngine"></a>\n\n### *class* ragsage.query.QueryEngine\n\nBases: `object`\n'

    injected = inject_source_links(page, _resolve_only_query_engine)

    lines = injected.text.splitlines()
    heading = lines.index("### *class* ragsage.query.QueryEngine")
    # Immediately below the heading, not appended to it: a `[source]` inside the
    # heading text lands in the page's table of contents, once per entry.
    assert lines[heading + 1] == f"[\\[source\\]]({_URL})"
    assert injected.unresolved == ()


def test_skips_an_object_with_no_locatable_source() -> None:
    """Dataclass fields, enum members and module constants never had one.

    ``inspect`` cannot locate them and neither could ``viewcode``: the 222 links
    the HTML build produced are exactly the objects that resolve. Emitting a link
    for the other 187 anchors would invent 187 links the old site never had.
    """
    page = '<a id="ragsage.config.QueryOptions.top_k"></a>\n\n#### top_k *: int = 5*\n'

    assert inject_source_links(page, _resolve_only_query_engine).text == page


def test_reports_the_objects_it_could_not_link() -> None:
    """Skipped is not the same as unnoticed.

    Most of the 187 are attributes that never had a source link and never will.
    But the set is also where a *regression* shows up: a class that stops
    resolving loses its link and changes nothing else about the page, so the only
    way anyone finds out is if the pass says which objects it passed over.
    """
    page = (
        '<a id="ragsage.query.QueryEngine"></a>\n\n### *class* ragsage.query.QueryEngine\n\n'
        '<a id="ragsage.config.QueryOptions.top_k"></a>\n\n#### top_k *: int = 5*\n\n'
        "<a id=\"ragsage.parsing.DocumentFormat.PDF\"></a>\n\n#### PDF *= 'pdf'*\n"
    )

    injected = inject_source_links(page, _resolve_only_query_engine)

    assert injected.unresolved == (
        "ragsage.config.QueryOptions.top_k",
        "ragsage.parsing.DocumentFormat.PDF",
    )


def test_reports_an_object_once_however_often_it_appears() -> None:
    """A name repeated down the list reads as several defects rather than one."""
    anchor = '<a id="ragsage.config.QueryOptions.top_k"></a>\n\n#### top_k *: int = 5*\n'

    injected = inject_source_links(anchor + "\n" + anchor, _resolve_only_query_engine)

    assert injected.unresolved == ("ragsage.config.QueryOptions.top_k",)


def test_ignores_prose_section_anchors() -> None:
    """``markdown_anchor_sections`` anchors every heading, not only the API ones."""
    page = '<a id="the-shape-of-it"></a>\n\n## The shape of it\n'

    injected = inject_source_links(page, lambda dotted: _URL)

    assert injected.text == page
    assert injected.unresolved == ()


def test_ignores_an_anchor_that_is_not_followed_by_a_heading() -> None:
    """A cross-reference target mid-paragraph is an anchor too, and is not an object."""
    page = '<a id="ragsage.query.QueryEngine"></a>\n\nSome prose that is not a heading.\n'

    injected = inject_source_links(page, _resolve_only_query_engine)

    assert injected.text == page
    # Not an object, so not something the pass failed to link: reporting it would
    # put a name in the operator's list that no source link was ever owed.
    assert injected.unresolved == ()


def test_handles_a_module_anchor() -> None:
    """Module headings are anchored ``module-ragsage.smalltalk``, not bare."""
    url = "https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/smalltalk.py#L1"
    page = '<a id="module-ragsage.smalltalk"></a>\n\n### ragsage.smalltalk\n'

    injected = inject_source_links(
        page, lambda dotted: url if dotted == "ragsage.smalltalk" else None
    )

    assert injected.text.splitlines()[-1] == f"[\\[source\\]]({url})"


def test_injects_once_per_object_across_a_page() -> None:
    page = (
        '<a id="ragsage.query.QueryEngine"></a>\n\n### *class* ragsage.query.QueryEngine\n\n'
        "Prose.\n\n"
        '<a id="ragsage.query.QueryEngine"></a>\n\n#### *async* query(question: str)\n'
    )

    injected = inject_source_links(page, _resolve_only_query_engine)

    assert injected.text.count("[\\[source\\]]") == 2


def test_is_idempotent() -> None:
    """A second pass over already-injected output must not double the links.

    The generator writes into a directory it wipes first, so this should never
    happen in production — but the pass is the kind of thing someone runs twice
    while debugging, and a silent doubling would be read as a builder bug.
    """
    page = '<a id="ragsage.query.QueryEngine"></a>\n\n### *class* ragsage.query.QueryEngine\n'

    once = inject_source_links(page, _resolve_only_query_engine)

    assert inject_source_links(once.text, _resolve_only_query_engine).text == once.text


def test_lists_every_unlinkable_object_for_the_operator() -> None:
    """The report is the whole list, not a count.

    A count says the number changed; the list says which object changed, which is
    the difference between "something regressed somewhere in six pages" and a
    one-line diff against the previous run's output.
    """
    stream = io.StringIO()

    report_unresolved(
        ("ragsage.storage.PostgresConfig.dsn", "ragsage.caching.PARSE_CACHE_VERSION"), stream
    )

    written = stream.getvalue()
    assert "2" in written
    assert "ragsage.storage.PostgresConfig.dsn" in written
    assert "ragsage.caching.PARSE_CACHE_VERSION" in written


def test_says_nothing_when_every_object_was_linked() -> None:
    stream = io.StringIO()

    report_unresolved((), stream)

    assert stream.getvalue() == ""


# ---------------------------------------------------------------------------- #
# Resolving a dotted name to a line of source
# ---------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Spelt out rather than imported from the generator, which is the whole assertion:
# imported, a link rewritten to point at a fork or a branch that does not exist
# would agree with itself and say nothing. The repository and the revision are
# half of what makes a source link correct.
_BLOB_PREFIX = "https://github.com/nirajk77777/ragsage/blob/main/"


def _source_at(url: str, count: int = 1) -> str:
    """The lines of checked-in source a generated URL points at.

    Deliberately not ``inspect``. The generator asked ``inspect`` where an object
    lives, so asking it again here would prove only that it is consistent with
    itself — and the failure this guards against is a URL that is well-formed,
    plausible and off by a file or a definition. Opening the repository file at
    the line the link names is what a reader clicking it actually gets.
    """
    assert url.startswith(_BLOB_PREFIX), f"not a link into this repository: {url}"

    path, _, fragment = url.removeprefix(_BLOB_PREFIX).partition("#")
    first = int(fragment.removeprefix("L"))
    lines = (_REPO_ROOT / path).read_text(encoding="utf-8").splitlines()

    return "\n".join(lines[first - 1 : first - 1 + count])


def test_resolves_a_class_to_the_line_it_is_declared_on() -> None:
    url = resolve_source_url("ragsage.query.QueryEngine")

    assert url is not None
    assert url.startswith(f"{_BLOB_PREFIX}src/ragsage/query.py#L")
    assert _source_at(url) == "class QueryEngine:"


def test_resolves_a_method_to_the_line_it_is_declared_on() -> None:
    """A method resolves through its class, and lands inside the class body."""
    url = resolve_source_url("ragsage.query.QueryEngine.query")

    assert url is not None
    assert url.startswith(f"{_BLOB_PREFIX}src/ragsage/query.py#L")
    assert _source_at(url) == "    async def query("


def test_resolves_a_module_level_function_to_the_line_it_is_declared_on() -> None:
    url = resolve_source_url("ragsage.query.reciprocal_rank_fusion")

    assert url is not None
    assert url.startswith(f"{_BLOB_PREFIX}src/ragsage/query.py#L")
    assert _source_at(url).startswith("def reciprocal_rank_fusion(")


def test_resolves_a_decorated_function_to_its_own_file() -> None:
    """The trap that made the unwrap in ``resolve_source_url`` one call, not two.

    ``getsourcelines`` unwraps a decorated object internally and ``getsourcefile``
    does not, so asking them separately about an ``@asynccontextmanager`` pairs a
    line number in ``storage/session.py`` with a filename of ``contextlib.py`` —
    a link that resolves, to a line of somebody else's standard library.
    """
    url = resolve_source_url("ragsage.storage.open_scoped_session")

    assert url is not None
    assert url.startswith(f"{_BLOB_PREFIX}src/ragsage/storage/session.py#L")
    assert _source_at(url, count=2) == "@asynccontextmanager\nasync def open_scoped_session("


def test_returns_none_for_a_dataclass_field() -> None:
    """A field is an annotation, not an object: there is nothing to locate."""
    assert resolve_source_url("ragsage.storage.PostgresConfig.dsn") is None


def test_returns_none_for_an_enum_member() -> None:
    """A member is a value, and a value has forgotten where it came from."""
    assert resolve_source_url("ragsage.parsing.DocumentFormat.PDF") is None


def test_returns_none_for_a_name_that_does_not_exist() -> None:
    """Guards the import walk: a typo must not raise out of a documentation build."""
    assert resolve_source_url("ragsage.query.NoSuchThing") is None


# ---------------------------------------------------------------------------- #
# Emitting for the target: frontmatter
# ---------------------------------------------------------------------------- #


def test_lifts_the_heading_into_frontmatter() -> None:
    """The site renders the title itself, so a surviving H1 would print it twice."""
    page = '<a id="facades"></a>\n\n# Façades\n\nFour entry points.\n'

    front = add_frontmatter(page, description="The four entry points.")

    assert front.startswith('---\ntitle: "Façades"\n')
    assert 'description: "The four entry points."' in front
    assert "\n# Façades\n" not in front
    assert "Four entry points." in front


def test_keeps_the_anchor_above_the_lifted_heading() -> None:
    """Cross-page links target it — `models#facades` must still land somewhere."""
    page = '<a id="facades"></a>\n\n# Façades\n\nBody.\n'

    assert '<a id="facades"></a>' in add_frontmatter(page, description="d")


def test_quotes_a_title_that_would_otherwise_break_yaml() -> None:
    page = "# Models: the domain data\n\nBody.\n"

    front = add_frontmatter(page, description='He said "no": really')

    assert 'title: "Models: the domain data"' in front
    assert 'description: "He said \\"no\\": really"' in front


def test_falls_back_to_no_title_when_a_page_has_no_heading() -> None:
    """A page the generator cannot title is a generator bug, not a silent default.

    Emitting frontmatter without a `title` makes the site's own schema reject the
    page by name, which is a far better failure than a page called "Untitled"
    quietly appearing in the navigation.
    """
    front = add_frontmatter("Just a paragraph.\n", description="d")

    assert "title:" not in front
    assert "Just a paragraph." in front


# ---------------------------------------------------------------------------- #
# Where the run is allowed to write
# ---------------------------------------------------------------------------- #


def _raw_build(root: Path) -> Path:
    """A stand-in for the directory ``sphinx-markdown-builder`` leaves behind.

    One page per stem the generator knows how to describe, read off the generator
    itself: the writing step refuses a set it cannot order or describe, and a
    hard-coded six here would fail on that rather than on what the test is about
    the day a seventh page is documented.
    """
    raw = root / "raw"
    raw.mkdir()
    for stem in _DESCRIPTIONS:
        (raw / f"{stem}.md").write_text(f"# {stem.title()}\n\nBody.\n", encoding="utf-8")
    return raw


def test_the_directory_it_wipes_holds_no_hand_written_page() -> None:
    """The two tests below take the destination as an argument. This one pins it.

    They establish that the writing step stays inside whatever directory it is
    handed — which is the whole of the confinement *given a correct destination*,
    and none of it if the constant moves. Point ``_OUTPUT_DIR`` one level up at the
    prose tree and both still pass, while the ``rmtree`` at the top of the run
    deletes every hand-written page on its way to writing the reference.

    Read off the checked-in tree rather than the constant restated: prose is
    ``.mdx`` and the generated reference is ``.md``, so "no hand-written page lives
    where the generator wipes" is a question the repository can answer.
    """
    prose = sorted(_OUTPUT_DIR.parent.rglob("*.mdx"))

    # The canary. The claim below is trivially true of a tree with no prose in it
    # — as this one would look if the content directory were ever moved and this
    # test left pointing at where it used to be.
    assert prose, (
        f"no hand-written pages were found under {_OUTPUT_DIR.parent}, so this test "
        "proves nothing about what the generator can reach"
    )

    inside = [page for page in prose if _OUTPUT_DIR in page.parents]
    assert not inside, (
        f"{len(inside)} hand-written page(s) live under {_OUTPUT_DIR}, which the "
        f"generator wipes on every run: {[str(page) for page in inside]}"
    )


def test_writes_nothing_outside_the_directory_it_owns(tmp_path: Path) -> None:
    """Prose sits one level up from the generated reference, and must survive a run.

    The two live in the same tree so that search and navigation range over one
    content source, which is what puts a hand-written page within reach of a
    generator that wipes before it writes. Nothing at the far end would report the
    loss: a clobbered page leaves a site that builds, links and navigates
    perfectly well with a page missing from it.
    """
    content = tmp_path / "content"
    (content / "api").mkdir(parents=True)
    (content / "quickstart.mdx").write_text("hand-written\n", encoding="utf-8")
    (content / "meta.json").write_text('{"pages": ["quickstart"]}\n', encoding="utf-8")

    _write_pages(_raw_build(tmp_path), content / "api")

    # The canary. A step that wrote nothing at all would leave every neighbour
    # below intact, and prove only that it did nothing.
    assert (content / "api" / "models.md").exists()

    assert sorted(path.name for path in content.iterdir()) == [
        "api",
        "meta.json",
        "quickstart.mdx",
    ]
    assert (content / "quickstart.mdx").read_text(encoding="utf-8") == "hand-written\n"
    assert (content / "meta.json").read_text(encoding="utf-8") == '{"pages": ["quickstart"]}\n'


def test_wipes_its_own_directory_so_a_retired_page_cannot_linger(tmp_path: Path) -> None:
    """The other half of owning it: a page renamed upstream must not be served forever.

    Merging would leave the old file behind, navigated by nothing, linked by
    nothing, and served to anyone still holding its URL — a page that says what
    the docstrings used to say, with no way to tell from the site that it is
    stale.
    """
    output = tmp_path / "api"
    output.mkdir()
    (output / "retired.md").write_text("what the docstrings used to say\n", encoding="utf-8")

    _write_pages(_raw_build(tmp_path), output)

    # The canary again: a step that wrote nothing would leave a directory holding
    # only the retired page, and the assertion below would be the one that failed.
    assert (output / "models.md").exists()

    assert not (output / "retired.md").exists()


# ---------------------------------------------------------------------------- #
# The order the reference is read in
# ---------------------------------------------------------------------------- #


def test_navigates_the_five_layers_in_the_order_the_engine_is_assembled(tmp_path: Path) -> None:
    """Façades, ports, models, configuration, adapters — read top to bottom.

    What you call, what it calls through, what flows across it, what configures
    it, what implements it. The order is the argument: it teaches the architecture
    to someone reading the sidebar down. Alphabetical would open on the adapters,
    which is the one layer a reader is free to ignore.

    Spelled out rather than compared against the generator's own tuple, which
    would restate the ordering from the thing being ordered and hold for any
    order at all.
    """
    output = tmp_path / "api"

    _write_pages(_raw_build(tmp_path), output)

    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    assert meta["pages"] == [
        "index",
        "facades",
        "ports",
        "models",
        "configuration",
        "adapters",
    ]


# ---------------------------------------------------------------------------- #
# The warning gate over the generated pages
# ---------------------------------------------------------------------------- #

# A dead `:class:` target, in the shape Sphinx reports one. This is the warning
# the gate exists for: the build still succeeds, the page still renders, and
# `:class:`Scope`` comes out as the literal text a reader cannot click.
_NEW_WARNING = (
    "/repo/src/ragsage/query.py:docstring of ragsage.query.QueryEngine:1: WARNING: "
    "py:class reference target not found: Scope"
)

# The one defect the run corrects downstream, and therefore tolerates here.
_KNOWN_DEFECT = f"WARNING: {_EXPECTED_WARNINGS[0]}"


def test_the_fixtures_say_what_the_gate_is_asked_about() -> None:
    """The canary for the two tests below, which is the whole allow-list.

    "Tolerates the known defect" is trivially true of a gate that tolerates
    everything, and "fails on a new warning" is trivially true of one that
    tolerates nothing. Both claims are only worth making if the two fixtures land
    on opposite sides of the same list — so that is what is asserted, off the list
    itself rather than off a copy of it.
    """
    assert _EXPECTED_WARNINGS, "the allow-list is empty, so nothing below distinguishes anything"
    assert any(expected in _KNOWN_DEFECT for expected in _EXPECTED_WARNINGS), (
        "the tolerated fixture is no longer on the allow-list, so the test that it passes "
        "proves nothing about the allow-list"
    )
    assert not any(expected in _NEW_WARNING for expected in _EXPECTED_WARNINGS), (
        "the fixture standing for a new warning is on the allow-list, so the test that it "
        "fails proves nothing"
    )


def test_a_dead_cross_reference_fails_the_generation(tmp_path: Path) -> None:
    """A role that resolves nowhere stops the build, and says which role.

    Sphinx reports this as a warning and exits zero. Nothing downstream would
    catch it either: the page builds, the site builds, and the built-site gate
    walks links — of which this is not one, because the role never became a link.
    It became the literal text ``:class:`Scope```, which is exactly the failure
    mode retaining Sphinx was supposed to buy the project out of.
    """
    warnings = tmp_path / "warnings.txt"
    warnings.write_text(f"{_NEW_WARNING}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        _check_warnings(warnings)

    assert "reference target not found: Scope" in str(raised.value), (
        f"the generation failed without naming the reference that failed: {raised.value}"
    )


def test_the_known_builder_defect_alone_does_not_fail_the_generation(tmp_path: Path) -> None:
    """The tolerated warning is tolerated — otherwise the gate is off on day one.

    The eaten keyword-only marker warns once per signature on every real run, and
    it is corrected two functions later. A gate red on arrival is a gate that gets
    switched off, and a switched-off gate is worse than the ``-W`` it replaced.
    """
    warnings = tmp_path / "warnings.txt"
    warnings.write_text(f"{_KNOWN_DEFECT}\n" * 35, encoding="utf-8")

    _check_warnings(warnings)


def test_a_new_warning_is_caught_among_the_tolerated_ones(tmp_path: Path) -> None:
    """The allow-list is applied per line, not to the text as a whole.

    This is the way the gate would actually stop working. A real run emits the
    tolerated defect 35 times, so a check asking "does this build's output contain
    a known defect?" is satisfied on every run there has ever been — and would go
    on being satisfied over a reference whose cross-references had all gone dead.
    The new warning is put in the middle, where a check that stops at the first
    tolerated line would never reach it.
    """
    warnings = tmp_path / "warnings.txt"
    warnings.write_text(
        f"{_KNOWN_DEFECT}\n{_NEW_WARNING}\n{_KNOWN_DEFECT}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as raised:
        _check_warnings(warnings)

    assert _NEW_WARNING in str(raised.value)
    assert _KNOWN_DEFECT not in str(raised.value), (
        "the failure lists the defect the run corrects, which buries the one line the "
        "reader has to act on among 35 they must not"
    )


def test_the_gate_reads_the_file_the_build_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sphinx's ``-w`` destination and the gated file are one decision.

    Everything above is about a file already on disk. This is about the wiring
    that puts one there — and it is the half with no symptom: point ``-w`` at one
    path and the gate at another, and the gate reads nothing, finds nothing and
    passes, on every build, forever. That is a green gate over an ungated build,
    which is the failure this whole arrangement exists to make impossible.

    Sphinx is stubbed rather than run. The claim is about which path the two
    halves agree on, and running Sphinx for real would take a minute to answer a
    question about two strings.
    """
    recorded: dict[str, list[str]] = {}

    def stub_sphinx(argv: list[str], check: bool = False) -> subprocess.CompletedProcess[bytes]:
        recorded["argv"] = argv
        # Whatever `-w` names is where a real run would put this.
        Path(argv[argv.index("-w") + 1]).write_text(f"{_NEW_WARNING}\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(generate_api_docs.subprocess, "run", stub_sphinx)

    with pytest.raises(SystemExit) as raised:
        _run_sphinx(tmp_path)

    # Canary: a stub that was never called wrote nothing, and a gate reading
    # nothing is exactly the defect this test is about.
    assert "-w" in recorded.get("argv", []), "sphinx was not asked to record its warnings at all"
    assert "reference target not found" in str(raised.value), (
        "sphinx warned and the run continued, so the gate is reading a different file than "
        f"the build writes: {recorded['argv']}"
    )
