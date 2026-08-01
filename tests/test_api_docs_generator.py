"""The two post-passes that correct ``sphinx-markdown-builder``'s known defects.

These are the riskiest custom code in the documentation pipeline, and the only
part of it that is ours. The generator's *output* is checked far more thoroughly
than this — the built site is walked link by link before it can deploy — but that
check costs a full multi-stage image build, which is a multi-minute cycle on
exactly the code most likely to need another attempt. Both passes are pure
text-in/text-out, so they get a fast seam of their own.

**Defect one: the keyword-only marker is eaten.** Sphinx's Python domain wraps
PEP 3102's ``*`` separator in a docutils ``abbreviation`` node (for the tooltip
that explains it). The Markdown builder has no handler for that node type, warns
once, and drops it — so ``chunk(self, document, pages, *, size, overlap)`` is
emitted as ``chunk(document, pages, , size, overlap)``, and a reader cannot tell
which arguments are keyword-only. There are 35 of these across the six generated
pages, in two shapes: 27 mid-signature (``, ,``) and 8 where the marker is the
first parameter (``(,``). Neither string can occur in a valid Python signature,
which is what makes the restoration unambiguous rather than a guess.

**Defect two: every source link is dropped.** ``sphinx.ext.viewcode`` puts a
``[source]`` link on all 222 documented objects in the HTML build; the Markdown
builder never visits the nodes that carry them, so all 222 vanish.
``sphinx.ext.linkcode`` was tried as the remedy and is dropped the same way. The
links are therefore injected here instead, off the dotted name in the anchor that
the builder emits above every generated heading.
"""

from __future__ import annotations

from tools.generate_api_docs import (
    add_frontmatter,
    inject_source_links,
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

    assert ", , " not in restored
    assert "(, " not in restored
    assert "Some prose." in restored


# ---------------------------------------------------------------------------- #
# Defect two: the dropped source links
# ---------------------------------------------------------------------------- #

_URL = "https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/query.py#L42"


def _resolve_only_query_engine(dotted: str) -> str | None:
    return _URL if dotted == "ragsage.query.QueryEngine" else None


def test_injects_a_source_link_below_a_documented_heading() -> None:
    page = '<a id="ragsage.query.QueryEngine"></a>\n\n### *class* ragsage.query.QueryEngine\n\nBases: `object`\n'

    injected = inject_source_links(page, _resolve_only_query_engine)

    lines = injected.splitlines()
    heading = lines.index("### *class* ragsage.query.QueryEngine")
    # Immediately below the heading, not appended to it: a `[source]` inside the
    # heading text lands in the page's table of contents, once per entry.
    assert lines[heading + 1] == f"[\\[source\\]]({_URL})"


def test_skips_an_object_with_no_locatable_source() -> None:
    """Dataclass fields, enum members and module constants never had one.

    ``inspect`` cannot locate them and neither could ``viewcode``: the 222 links
    the HTML build produced are exactly the objects that resolve. Emitting a link
    for the other 187 anchors would invent 187 links the old site never had.
    """
    page = '<a id="ragsage.config.QueryOptions.top_k"></a>\n\n#### top_k *: int = 5*\n'

    assert inject_source_links(page, _resolve_only_query_engine) == page


def test_ignores_prose_section_anchors() -> None:
    """``markdown_anchor_sections`` anchors every heading, not only the API ones."""
    page = '<a id="the-shape-of-it"></a>\n\n## The shape of it\n'

    assert inject_source_links(page, lambda dotted: _URL) == page


def test_ignores_an_anchor_that_is_not_followed_by_a_heading() -> None:
    """A cross-reference target mid-paragraph is an anchor too, and is not an object."""
    page = '<a id="ragsage.query.QueryEngine"></a>\n\nSome prose that is not a heading.\n'

    assert inject_source_links(page, _resolve_only_query_engine) == page


def test_handles_a_module_anchor() -> None:
    """Module headings are anchored ``module-ragsage.smalltalk``, not bare."""
    url = "https://github.com/nirajk77777/ragsage/blob/main/src/ragsage/smalltalk.py#L1"
    page = '<a id="module-ragsage.smalltalk"></a>\n\n### ragsage.smalltalk\n'

    injected = inject_source_links(
        page, lambda dotted: url if dotted == "ragsage.smalltalk" else None
    )

    assert injected.splitlines()[-1] == f"[\\[source\\]]({url})"


def test_injects_once_per_object_across_a_page() -> None:
    page = (
        '<a id="ragsage.query.QueryEngine"></a>\n\n### *class* ragsage.query.QueryEngine\n\n'
        "Prose.\n\n"
        '<a id="ragsage.query.QueryEngine"></a>\n\n#### *async* query(question: str)\n'
    )

    injected = inject_source_links(page, _resolve_only_query_engine)

    assert injected.count("[\\[source\\]]") == 2


def test_is_idempotent() -> None:
    """A second pass over already-injected output must not double the links.

    The generator writes into a directory it wipes first, so this should never
    happen in production — but the pass is the kind of thing someone runs twice
    while debugging, and a silent doubling would be read as a builder bug.
    """
    page = '<a id="ragsage.query.QueryEngine"></a>\n\n### *class* ragsage.query.QueryEngine\n'

    once = inject_source_links(page, _resolve_only_query_engine)

    assert inject_source_links(once, _resolve_only_query_engine) == once


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
