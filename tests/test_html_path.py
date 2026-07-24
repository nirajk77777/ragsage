"""Unit coverage for the HTML / plain-web parse path.

The HTML path is a flow format: it must collapse a whole page onto a single
:class:`~ragsage.models.Page`, keep the ``<h1>``…``<h6>`` hierarchy as ATX
headings, pull only the main content (dropping nav/header/footer/aside chrome),
and serialise tables as pipe tables. These tests drive one hand-authored HTML
fixture — nested headings, a main region with paragraphs + a list + a table, and
surrounding boilerplate — straight through :func:`ragsage.parsing.html.parse_html`
and then through the real :class:`~ragsage.parsing.HeuristicBackend` chunker, so
the acceptance criteria (single page, boilerplate excluded, heading context in
``chunk.metadata["headings"]``) are proven against production code, not mocks.
Everything is in-memory — no network, no URL fetches.
"""

from __future__ import annotations

from ragsage.models import Page, RawSource
from ragsage.parsing import HeuristicBackend
from ragsage.parsing.html import parse_html

# A realistic page: chrome (nav + footer) wrapping a <main> whose content nests
# h1 > h2 > h3, with paragraphs, a list, and a table.
_HTML = """\
<!doctype html>
<html>
  <head><title>Ignored Title</title><style>.x{color:red}</style></head>
  <body>
    <nav><a href="/home">Home</a> <a href="/pricing">Pricing menu link</a></nav>
    <header><h1>Site Banner Chrome</h1></header>
    <main>
      <h1>Quarterly Report</h1>
      <p>Opening remarks for the quarter.</p>
      <h2>Revenue</h2>
      <p>Revenue rose across every region.</p>
      <h3>North America</h3>
      <p>Strong growth in enterprise accounts.</p>
      <ul>
        <li>Enterprise up</li>
        <li>SMB flat</li>
      </ul>
      <table>
        <thead><tr><th>Item</th><th>Qty</th></tr></thead>
        <tbody>
          <tr><td>Apple</td><td>3</td></tr>
          <tr><td>Pear</td><td>5</td></tr>
        </tbody>
      </table>
    </main>
    <footer>Copyright 2026 boilerplate footer text</footer>
  </body>
</html>
"""


def _source(html: str = _HTML) -> RawSource:
    return RawSource(name="page.html", content=html.encode("utf-8"), media_type="text/html")


# --------------------------------------------------------------------------- #
# Criterion 1 — a single layout-free page
# --------------------------------------------------------------------------- #


def test_parses_to_single_layout_free_page() -> None:
    """HTML yields exactly one Page(number=1, layout=None) so it routes as TEXT."""
    pages = parse_html(_source(), _HTML.encode("utf-8"))
    assert len(pages) == 1
    page = pages[0]
    assert isinstance(page, Page)
    assert page.number == 1
    assert page.layout is None
    assert page.image is None
    assert page.text.strip()


# --------------------------------------------------------------------------- #
# Criterion 2 — heading hierarchy + main content kept, boilerplate dropped
# --------------------------------------------------------------------------- #


def test_heading_hierarchy_serialised_as_atx() -> None:
    """h1/h2/h3 become #/##/### in document order."""
    markdown = parse_html(_source(), _HTML.encode("utf-8"))[0].text
    assert "# Quarterly Report" in markdown
    assert "## Revenue" in markdown
    assert "### North America" in markdown
    assert markdown.index("# Quarterly Report") < markdown.index("## Revenue") < markdown.index(
        "### North America"
    )


def test_main_content_present_boilerplate_absent() -> None:
    """Body prose/list/table survive; nav, header banner and footer do not."""
    markdown = parse_html(_source(), _HTML.encode("utf-8"))[0].text
    # Main content present.
    assert "Opening remarks for the quarter." in markdown
    assert "Strong growth in enterprise accounts." in markdown
    assert "- Enterprise up" in markdown
    # Boilerplate excluded.
    assert "Pricing menu link" not in markdown
    assert "Site Banner Chrome" not in markdown
    assert "boilerplate footer text" not in markdown
    assert "color:red" not in markdown  # <style> dropped


def test_table_rendered_as_pipe_table() -> None:
    """A <table> becomes a GitHub pipe table the chunker can keep whole."""
    markdown = parse_html(_source(), _HTML.encode("utf-8"))[0].text
    assert "| Item | Qty |" in markdown
    assert "| --- | --- |" in markdown
    assert "| Apple | 3 |" in markdown
    assert "| Pear | 5 |" in markdown


def test_declared_charset_is_honoured() -> None:
    """A non-UTF-8 body declared via <meta charset> decodes without mojibake."""
    html = (
        '<html><head><meta charset="latin-1"></head>'
        "<body><main><h1>Caf\xe9</h1><p>na\xefve</p></main></body></html>"
    )
    markdown = parse_html(_source(), html.encode("latin-1"))[0].text
    assert "# Café" in markdown
    assert "naïve" in markdown


# --------------------------------------------------------------------------- #
# Criterion 3 — heading context reaches the chunker's metadata
# --------------------------------------------------------------------------- #


def test_chunker_carries_nested_heading_path() -> None:
    """Through the real backend, the nested heading path lands in chunk metadata."""
    backend = HeuristicBackend()
    source = _source()
    parsed = backend.parse(source)
    chunks = backend.chunk(parsed.document, parsed.pages, size=512, overlap=64)

    assert chunks, "expected at least one chunk"
    heading_paths = [tuple(chunk.metadata.get("headings", ())) for chunk in chunks]
    # The deepest section carries its full ancestry.
    assert ("Quarterly Report", "Revenue", "North America") in heading_paths
    # And the top section is tagged too.
    assert any(path and path[0] == "Quarterly Report" for path in heading_paths)


def test_backend_routes_html_to_single_page() -> None:
    """The backend's format dispatch reaches parse_html and yields one page."""
    parsed = HeuristicBackend().parse(_source())
    assert len(parsed.pages) == 1
    assert parsed.pages[0].layout is None
