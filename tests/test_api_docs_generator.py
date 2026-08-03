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
"""

from __future__ import annotations

import io
from pathlib import Path

from tools.generate_api_docs import (
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
