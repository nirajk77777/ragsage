"""Unit coverage for the born-digital PDF parse path (:mod:`ragsage.parsing.pdf`).

The heuristic backend is pure Python, so the whole PDF contract is exercised here
against small PDFs generated in-process with ``reportlab`` — no network, no
checked-in binaries. Each acceptance criterion of issue 05 has a test:

* reading order with column gutters respected (a two-column page);
* a heading hierarchy recovered from font size and numbered/short-line cues, seen
  through the *real* chunker as ``metadata["headings"]``;
* ruled and borderless tables serialised to Markdown and kept whole by the chunker;
* a measured :class:`~ragsage.models.PageLayout` per page, with text-thin /
  image-dominated pages carrying a rendered image and routing to VISION;
* correct per-page numbers across a multi-page document;
* a page that fails to parse degrading to the vision route, not failing the doc.

Chunk-level criteria run the genuine two-pass chunker via
:class:`~ragsage.parsing.HeuristicBackend` rather than asserting on Markdown alone,
so the tests prove the contract the pipeline actually depends on.
"""

from __future__ import annotations

import io
import sys

import pytest

from ragsage.models import PageRoute, RawSource
from ragsage.parsing import HeuristicBackend
from ragsage.parsing.classifier import LayoutPageClassifier

reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402

_WIDTH, _HEIGHT = LETTER


# --------------------------------------------------------------------------- #
# PDF fixture builders (generated in-process, hermetic)
# --------------------------------------------------------------------------- #


def _canvas() -> tuple[reportlab_canvas.Canvas, io.BytesIO]:
    buffer = io.BytesIO()
    return reportlab_canvas.Canvas(buffer, pagesize=LETTER), buffer


def _draw_ruled_table(
    c: reportlab_canvas.Canvas,
    x: float,
    y_top: float,
    headers: list[str],
    rows: list[list[str]],
    *,
    col_w: float = 110.0,
    row_h: float = 20.0,
) -> None:
    """Draw a grid-ruled table: cell text plus the horizontal and vertical rules."""
    data = [headers, *rows]
    ncols = len(headers)
    c.setFont("Helvetica", 11)
    for r, row in enumerate(data):
        for col, cell in enumerate(row):
            c.drawString(x + col * col_w + 4, y_top - (r + 1) * row_h + 6, cell)
    c.setLineWidth(0.7)
    for r in range(len(data) + 1):
        c.line(x, y_top - r * row_h, x + ncols * col_w, y_top - r * row_h)
    for col in range(ncols + 1):
        c.line(x + col * col_w, y_top, x + col * col_w, y_top - len(data) * row_h)


def _heading_and_ruled_table_pdf() -> bytes:
    c, buffer = _canvas()
    y = _HEIGHT - 72
    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, y, "Quarterly Report")
    y -= 40
    c.setFont("Helvetica", 11)
    c.drawString(72, y, "Opening remarks for the quarter.")
    y -= 30
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "Revenue")
    y -= 26
    c.setFont("Helvetica", 11)
    c.drawString(72, y, "Revenue rose across every region.")
    y -= 30
    _draw_ruled_table(
        c, 72, y, ["Item", "Qty", "Notes"], [["Apple", "3", "fresh"], ["Pear", "5", "ripe"]]
    )
    c.showPage()
    c.save()
    return buffer.getvalue()


def _three_level_heading_pdf() -> bytes:
    c, buffer = _canvas()
    y = _HEIGHT - 72
    for text, size in [
        ("Field Manual", 22),
        ("Installation", 16),
        ("Prerequisites", 13),
    ]:
        c.setFont("Helvetica-Bold", size)
        c.drawString(72, y, text)
        y -= size + 12
        c.setFont("Helvetica", 11)
        c.drawString(72, y, f"Body text sitting under the {text} heading.")
        y -= 26
    c.showPage()
    c.save()
    return buffer.getvalue()


def _numbered_heading_pdf() -> bytes:
    """A page whose headings are same-size-as-body but numbered + bold + short."""
    c, buffer = _canvas()
    y = _HEIGHT - 72
    c.setFont("Helvetica-Bold", 20)
    c.drawString(72, y, "Specification")
    y -= 34
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, y, "2.4 Retry Semantics")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(72, y, "Requests are retried with exponential backoff on failure.")
    c.showPage()
    c.save()
    return buffer.getvalue()


_LEFT_COLUMN = [
    "The morning began with a long walk",
    "along the river where the mist still",
    "clung to the reeds. She counted seven",
    "herons before breakfast and lost count",
    "of the smaller birds entirely that day.",
    "By noon the sun had burned away all",
    "trace of the fog over the still water.",
]
_RIGHT_COLUMN = [
    "Meanwhile the town square filled up",
    "with vendors unpacking wooden crates.",
    "Oranges, olives, and bright bolts of",
    "cloth appeared on every table as the",
    "clock above the old hall struck ten.",
    "A stray dog wandered between the stalls",
    "hoping for a fallen scrap or maybe two.",
    "The baker waved to everyone he knew.",
]


def _two_column_pdf() -> bytes:
    c, buffer = _canvas()
    c.setFont("Helvetica", 11)
    top = _HEIGHT - 100
    for i, line in enumerate(_LEFT_COLUMN):
        c.drawString(72, top - i * 16, line)
    # Right column baselines are staggered so rows don't align across the gutter —
    # this is prose in two columns, not a two-column table.
    for i, line in enumerate(_RIGHT_COLUMN):
        c.drawString(330, (top - 8) - i * 16, line)
    c.showPage()
    c.save()
    return buffer.getvalue()


def _borderless_table_pdf() -> bytes:
    c, buffer = _canvas()
    c.setFont("Helvetica-Bold", 15)
    c.drawString(72, _HEIGHT - 80, "Price List")
    c.setFont("Helvetica", 11)
    c.drawString(72, _HEIGHT - 105, "Current prices are shown below.")
    rows = [
        ["Product", "Price", "Stock"],
        ["Widget", "12", "yes"],
        ["Gadget", "34", "no"],
        ["Gizmo", "56", "yes"],
    ]
    y = _HEIGHT - 150
    c.setFont("Helvetica", 11)
    for r, row in enumerate(rows):
        for col, cell in enumerate(row):
            c.drawString(72 + col * 130, y - r * 22, cell)
    c.showPage()
    c.save()
    return buffer.getvalue()


def _image_dominated_pdf() -> bytes:
    """A page covered by one big raster image with only a stray caption of text."""
    from PIL import Image

    picture = Image.new("RGB", (400, 500), (180, 180, 180))
    c, buffer = _canvas()
    c.drawImage(ImageReader(picture), 20, 20, width=_WIDTH - 40, height=_HEIGHT - 40)
    c.setFont("Helvetica", 11)
    c.drawString(72, 40, "fig")
    c.showPage()
    c.save()
    return buffer.getvalue()


def _thin_text_pdf() -> bytes:
    """A page with almost no text layer and no image — a scan-like empty page."""
    c, buffer = _canvas()
    c.setFont("Helvetica", 11)
    c.drawString(72, _HEIGHT - 100, "Fig 1")
    c.showPage()
    c.save()
    return buffer.getvalue()


def _multipage_pdf() -> bytes:
    c, buffer = _canvas()
    c.setFont("Helvetica", 11)
    c.drawString(72, _HEIGHT - 100, "Page one is all about apples and their many uses.")
    c.showPage()
    c.setFont("Helvetica", 11)
    c.drawString(72, _HEIGHT - 100, "Page two turns to bananas, grapes, and pears.")
    c.showPage()
    c.save()
    return buffer.getvalue()


def _pdf_source(content: bytes, *, name: str = "doc.pdf") -> RawSource:
    return RawSource(name=name, content=content, media_type="application/pdf")


def _parse(content: bytes, *, name: str = "doc.pdf") -> list:
    """Parse a PDF straight through the backend (which lazily dispatches to pdf.py)."""
    source = _pdf_source(content, name=name)
    return list(HeuristicBackend().parse(source).pages)


def _chunk(content: bytes, *, name: str = "doc.pdf") -> list:
    backend = HeuristicBackend()
    parsed = backend.parse(_pdf_source(content, name=name))
    return list(backend.chunk(parsed.document, parsed.pages, size=512, overlap=64))


# --------------------------------------------------------------------------- #
# Criterion 1 — per-page Markdown in reading order, columns respected
# --------------------------------------------------------------------------- #


def test_born_digital_page_parses_to_markdown() -> None:
    pages = _parse(_heading_and_ruled_table_pdf())
    assert len(pages) == 1
    text = pages[0].text
    assert "# Quarterly Report" in text  # a Markdown ATX heading
    assert "Revenue rose across every region." in text


def test_multi_column_page_is_read_column_by_column() -> None:
    text = _parse(_two_column_pdf())[0].text
    # The whole left column is emitted before the whole right column — not
    # interleaved line-by-line across the gutter.
    last_left = text.index(_LEFT_COLUMN[-1])
    first_right = text.index(_RIGHT_COLUMN[0])
    assert last_left < first_right
    for line in _LEFT_COLUMN:
        assert text.index(line) < first_right


# --------------------------------------------------------------------------- #
# Criterion 2 — heading hierarchy surfaces in chunk metadata
# --------------------------------------------------------------------------- #


def test_heading_hierarchy_from_font_size_surfaces_in_chunk_metadata() -> None:
    chunks = _chunk(_three_level_heading_pdf())
    paths = {tuple(c.metadata.get("headings", ())) for c in chunks}
    assert ("Field Manual", "Installation", "Prerequisites") in paths


def test_two_level_heading_path_from_ruled_table_doc() -> None:
    chunks = _chunk(_heading_and_ruled_table_pdf())
    paths = {tuple(c.metadata.get("headings", ())) for c in chunks}
    assert ("Quarterly Report", "Revenue") in paths


def test_numbered_short_bold_line_is_detected_as_a_heading() -> None:
    # Same font size as the body, but the numbered prefix + short + bold cues
    # promote it — and it shows up as a heading path in the chunks.
    chunks = _chunk(_numbered_heading_pdf())
    joined_paths = [" ".join(c.metadata.get("headings", ())) for c in chunks]
    assert any("2.4 Retry Semantics" in path for path in joined_paths)


# --------------------------------------------------------------------------- #
# Criterion 3 — tables to Markdown, kept whole by the chunker
# --------------------------------------------------------------------------- #


def test_ruled_table_serialises_to_markdown_and_survives_chunking_whole() -> None:
    chunks = _chunk(_heading_and_ruled_table_pdf())
    rows = ["| Item | Qty | Notes |", "| Apple | 3 | fresh |", "| Pear | 5 | ripe |"]
    holders = [c for c in chunks if all(row in c.text for row in rows)]
    assert len(holders) == 1  # the entire table lives in exactly one chunk


def test_borderless_table_is_extracted_to_markdown() -> None:
    chunks = _chunk(_borderless_table_pdf())
    rows = [
        "| Product | Price | Stock |",
        "| Widget | 12 | yes |",
        "| Gizmo | 56 | yes |",
    ]
    holders = [c for c in chunks if all(row in c.text for row in rows)]
    assert len(holders) == 1


def test_two_column_prose_is_not_mistaken_for_a_table() -> None:
    # Prose in two columns must not be swept into a pipe table.
    assert "|" not in _parse(_two_column_pdf())[0].text


# --------------------------------------------------------------------------- #
# Criterion 4 — measured layout; thin/image-dominated pages route to VISION
# --------------------------------------------------------------------------- #


def test_born_digital_page_carries_layout_and_routes_to_text() -> None:
    page = _parse(_heading_and_ruled_table_pdf())[0]
    assert page.layout is not None
    assert page.layout.text_chars > 24
    assert LayoutPageClassifier().classify(page) is PageRoute.TEXT


def test_image_dominated_page_gets_an_image_and_routes_to_vision() -> None:
    page = _parse(_image_dominated_pdf())[0]
    assert page.layout is not None
    assert page.layout.image_area_ratio >= 0.5  # image covers the page
    assert page.image is not None
    assert page.image.data  # a real rendered PNG, not just a ref
    assert page.image.ref == "doc.pdf#page=1"
    assert LayoutPageClassifier().classify(page) is PageRoute.VISION


def test_text_thin_page_routes_to_vision_with_a_rendered_image() -> None:
    page = _parse(_thin_text_pdf())[0]
    assert page.layout is not None
    assert page.layout.text_chars < 24  # too little text to trust
    assert page.image is not None and page.image.data
    assert LayoutPageClassifier().classify(page) is PageRoute.VISION


def test_rendered_image_is_a_png() -> None:
    page = _parse(_image_dominated_pdf())[0]
    assert page.image is not None and page.image.data is not None
    assert page.image.data.startswith(b"\x89PNG\r\n")


# --------------------------------------------------------------------------- #
# Criterion 5 — multi-page documents keep correct per-page numbers
# --------------------------------------------------------------------------- #


def test_multi_page_pdf_yields_ordered_page_numbers() -> None:
    pages = _parse(_multipage_pdf())
    assert [p.number for p in pages] == [1, 2]
    assert "apples" in pages[0].text
    assert "bananas" in pages[1].text


def test_page_numbers_attribute_chunks_across_pages() -> None:
    chunks = _chunk(_multipage_pdf())
    pages_seen = {c.page for c in chunks}
    assert pages_seen == {1, 2}


# --------------------------------------------------------------------------- #
# Criterion 6 — a hard page degrades to vision instead of failing the document
# --------------------------------------------------------------------------- #


def test_unparseable_page_degrades_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragsage.parsing import pdf as pdf_module

    def _boom(_page: object) -> tuple[str, object]:
        raise RuntimeError("simulated bad page")

    monkeypatch.setattr(pdf_module, "_extract_page", _boom)

    pages = _parse(_multipage_pdf())
    # Both pages still come back, in order — the document is not lost…
    assert [p.number for p in pages] == [1, 2]
    for page in pages:
        # …each degraded to the vision route: empty text, a rendered image, and a
        # layout that forces VISION.
        assert page.text == ""
        assert page.image is not None and page.image.data
        assert LayoutPageClassifier().classify(page) is PageRoute.VISION


# --------------------------------------------------------------------------- #
# Criterion 7 — the parser libraries are imported lazily, not at package import
# --------------------------------------------------------------------------- #


def test_importing_the_pdf_module_does_not_drag_in_the_pdf_libraries() -> None:
    # The dependency guard relies on this: importing the module must not pull in
    # pdfplumber / pypdfium2 — they load only inside parse_pdf, when a PDF is
    # actually parsed. Reloading re-executes the module top to prove it stays clean.
    import importlib

    from ragsage.parsing import pdf as pdf_module

    for module in ("pdfplumber", "pypdfium2"):
        sys.modules.pop(module, None)
    importlib.reload(pdf_module)
    assert "pdfplumber" not in sys.modules
    assert "pypdfium2" not in sys.modules
