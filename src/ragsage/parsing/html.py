"""The HTML / plain-web parse path — main content to Markdown, model-free.

HTML is a *flow* format: it has no page grid, so unlike the paged PDF path this
module collapses the whole document onto a single
:class:`~ragsage.models.Page` (number 1) whose ``text`` is Markdown and whose
``layout`` is ``None`` — which makes the page classifier fall back to
text-presence and route the page as ``TEXT``. The shared two-pass chunker then
splits that Markdown on its heading hierarchy exactly as it does for every other
format.

Two jobs matter for retrieval quality, and both are pure heuristics — no model,
no network:

* **Strip the boilerplate.** A real web page is mostly chrome — nav bars, site
  headers/footers, sidebars, scripts, styling. Ingesting that pollutes the
  corpus with menu links that match everything and mean nothing, so we drop the
  structural chrome elements and read only the main-content root (``<main>`` /
  ``<article>``, or the densest content block).
* **Preserve the heading hierarchy.** ``<h1>``…``<h6>`` become ATX headings
  ``#``…``######`` so the chunker records each section's heading path in
  ``chunk.metadata["headings"]`` and a citation can say which section a passage
  came from. Paragraphs, lists and tables are serialised too (tables as GitHub
  pipe tables, so the chunker keeps them whole).

``beautifulsoup4`` and ``lxml`` are imported *lazily inside* :func:`parse_html`,
never at module top level: the composition root imports :mod:`ragsage.parsing`,
and that import must stay free of any parser library (and of the CPU-compat
dependency guard). We deliberately do **not** use ``trafilatura`` — its
main-content extractor is tuned for full articles and drops the short documents
this path must still handle correctly; a direct BeautifulSoup walk is smaller,
predictable, and keeps headings/tables intact.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ragsage.models import Page, RawSource
from ragsage.parsing.chunking import normalize_newlines

if TYPE_CHECKING:
    from bs4 import BeautifulSoup, Tag

# HTML is a flow format: the whole document lives on a single page.
_FLOW_FORMAT_PAGE = 1

# Structural chrome — everything that wraps the real content but isn't it. These
# are removed wholesale before serialising so their menu links, legal boilerplate
# and inline scripts never reach the corpus.
_BOILERPLATE_TAGS = ("nav", "header", "footer", "aside", "script", "style", "template")

# Container tags we descend *through* looking for content blocks, rather than
# treating their text as one paragraph.
_CONTAINERS = frozenset(
    {"div", "section", "article", "main", "body", "figure", "details", "html"}
)

# ``<hN>`` -> ATX heading level. h1..h4 are what the chunker splits on; h5/h6 are
# still emitted (as ``#####``/``######``) so nothing is lost.
_HEADINGS = {f"h{level}": level for level in range(1, 7)}

# A charset declared in a ``<meta charset=...>`` / ``content="...; charset=..."``.
_META_CHARSET = re.compile(rb"""charset\s*=\s*["']?\s*([\w-]+)""", re.IGNORECASE)

# Collapses any run of whitespace (including newlines from source indentation)
# to a single space when flattening inline text.
_WHITESPACE = re.compile(r"\s+")


def parse_html(source: RawSource, content: bytes) -> list[Page]:
    """Parse HTML bytes into a single Markdown :class:`~ragsage.models.Page`.

    The declared ``<meta charset>`` is honoured when present, otherwise the bytes
    are read as UTF-8 with undecodable sequences replaced. The main-content root
    is walked to Markdown (headings, paragraphs, lists, pipe tables); structural
    chrome is stripped first. ``source`` is accepted to satisfy the
    :data:`~ragsage.parsing.backend.FormatParser` shape and is otherwise unused.
    """
    from bs4 import BeautifulSoup

    _ = source  # part of the format-parser signature; nothing here needs it
    text = _decode(content)
    soup = BeautifulSoup(text, "lxml")

    for element in soup.find_all(_BOILERPLATE_TAGS):
        element.decompose()

    root = _content_root(soup)
    markdown = normalize_newlines(_render(root)) if root is not None else ""
    return [Page(number=_FLOW_FORMAT_PAGE, text=markdown, layout=None)]


def _decode(content: bytes) -> str:
    """Decode the source bytes, honouring a declared charset, else UTF-8-replace."""
    match = _META_CHARSET.search(content[:4096])
    if match:
        try:
            return content.decode(match.group(1).decode("ascii"))
        except (LookupError, UnicodeDecodeError):
            pass
    return content.decode("utf-8", "replace")


def _content_root(soup: BeautifulSoup) -> Tag | None:
    """Pick the element that holds the real content.

    Prefer the semantic ``<main>``/``<article>`` landmark; failing that, the
    text-densest block-level container (a page that wraps its body in an
    unlabelled ``<div>``); failing that, the ``<body>`` (chrome already stripped).
    """
    for name in ("main", "article"):
        landmark = soup.find(name)
        if isinstance(landmark, _tag_type()):
            return landmark

    body = soup.find("body")
    candidates = body.find_all(["div", "section"]) if isinstance(body, _tag_type()) else []
    densest = max(
        candidates,
        key=lambda tag: len(tag.get_text(strip=True)),
        default=None,
    )
    if densest is not None and densest.get_text(strip=True):
        return densest
    return body if isinstance(body, _tag_type()) else None


def _render(root: Tag) -> str:
    """Serialise a content root to Markdown blocks joined by blank lines."""
    blocks = _blocks(root)
    return "\n\n".join(block for block in blocks if block)


def _blocks(node: Tag) -> list[str]:
    """Walk a container's children into ordered Markdown blocks.

    Recognised leaf blocks (headings, paragraphs, lists, tables, code) render
    directly; nested containers are descended into so structure at any depth is
    flattened in document order. Stray inline text between blocks becomes its own
    paragraph so nothing is silently dropped.
    """
    from bs4.element import NavigableString

    blocks: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = _clean(str(child))
            if text:
                blocks.append(text)
            continue
        if not isinstance(child, _tag_type()):
            continue
        name = (child.name or "").lower()
        if name in _HEADINGS:
            heading = _inline(child)
            if heading:
                blocks.append("#" * _HEADINGS[name] + " " + heading)
        elif name == "p":
            paragraph = _inline(child)
            if paragraph:
                blocks.append(paragraph)
        elif name in ("ul", "ol"):
            blocks.append(_render_list(child, ordered=name == "ol"))
        elif name == "table":
            table = _render_table(child)
            if table:
                blocks.append(table)
        elif name in ("pre",):
            code = child.get_text().strip("\n")
            if code:
                blocks.append("```\n" + code + "\n```")
        elif name == "blockquote":
            quote = _inline(child)
            if quote:
                blocks.append("> " + quote)
        elif name in ("hr", "br"):
            continue
        elif name in _CONTAINERS or child.find(True) is not None:
            blocks.extend(_blocks(child))
        else:
            text = _inline(child)
            if text:
                blocks.append(text)
    return blocks


def _render_list(node: Tag, *, ordered: bool) -> str:
    """Render a ``<ul>``/``<ol>`` as a Markdown list, one line per ``<li>``."""
    lines: list[str] = []
    index = 1
    for item in node.find_all("li", recursive=False):
        marker = f"{index}." if ordered else "-"
        lines.append(f"{marker} {_inline(item)}".rstrip())
        index += 1
    return "\n".join(lines)


def _render_table(node: Tag) -> str:
    """Render a ``<table>`` as a GitHub pipe table (header, delimiter, rows).

    The first row (whether in ``<thead>`` or the body) is the header. Cell text is
    flattened inline and pipes escaped so a value never breaks the column grid;
    rows are padded/truncated to the header's width so every line is well-formed.
    """
    rows = node.find_all("tr")
    if not rows:
        return ""
    header = _cells(rows[0])
    width = len(header)
    if width == 0:
        return ""
    lines = [_row(header, width), "| " + " | ".join(["---"] * width) + " |"]
    for row in rows[1:]:
        lines.append(_row(_cells(row), width))
    return "\n".join(lines)


def _cells(row: Tag) -> list[str]:
    """The header/data cell texts of one ``<tr>``, pipes escaped."""
    return [_inline(cell).replace("|", r"\|") for cell in row.find_all(["th", "td"], recursive=False)]


def _row(cells: list[str], width: int) -> str:
    """Format one table row padded/truncated to ``width`` columns."""
    padded = (cells + [""] * width)[:width]
    return "| " + " | ".join(padded) + " |"


def _inline(node: Tag) -> str:
    """Flatten an element's text to a single whitespace-collapsed line."""
    return _clean(node.get_text(" "))


def _clean(text: str) -> str:
    """Collapse whitespace runs and trim — source indentation becomes one space."""
    return _WHITESPACE.sub(" ", text).strip()


def _tag_type() -> type[Tag]:
    """The ``bs4.Tag`` class, imported lazily for isinstance checks."""
    from bs4 import Tag

    return Tag
