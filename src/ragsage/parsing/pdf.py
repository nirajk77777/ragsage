"""The PDF parse path — born-digital text, model-free, with a vision fallback.

This is the highest-value format, so it gets the richest heuristics — but still
*no* ML: layout is recovered from the geometry ``pdfplumber`` already hands us
(per-word position, font size, font name) rather than from a trained model.

What the path does, per page:

* **Reading order + columns.** Words are grouped into lines and lines into
  columns by finding the empty vertical *gutters* between word clusters, so a
  two-column page is emitted left-column-first instead of interleaving the two.
* **Heading hierarchy.** The modal body font size is the baseline; lines set in a
  visibly larger (or bold, short, numbered) face are promoted to Markdown ATX
  headings, ranked biggest-first into ``#``…``####``. That is the structural
  carrier the shared chunker splits on and records as ``metadata["headings"]``.
* **Tables → Markdown.** ``pdfplumber``'s ruled *and* text strategies find tables;
  each is serialised to a GitHub pipe table (and its cell words are pulled out of
  the prose flow) so the chunker keeps it whole.
* **Route signals.** Every page carries a measured
  :class:`~ragsage.models.PageLayout`; a text-thin or image-dominated page (a
  scan, a photo) additionally carries a rendered :class:`~ragsage.models.PageImage`
  so the unchanged classifier sends it to the vision route.
* **Degrade, don't die.** A page that throws mid-parse still yields a ``Page`` —
  a rendered image plus a vision-forcing layout — so one bad page never takes the
  whole document down.

``pdfplumber`` and ``pypdfium2`` are imported *lazily inside* the entry point so
that ``import ragsage.parsing`` stays free of any parser library (keeping the
composition root's import graph — and the CPU-compatibility dependency guard —
clean). ``pypdfium2`` is what ``pdfplumber`` already renders with, so it costs no
new wheel.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from ragsage.models import Page, PageImage, PageLayout, PageRoute, RawSource
from ragsage.parsing.classifier import LayoutPageClassifier

# The routing policy is parser-neutral, so we reuse the very classifier the
# pipeline will run — a page we would send to vision is exactly one that gets a
# rendered image attached. Defaults only; the backend may reconfigure its own.
_CLASSIFIER = LayoutPageClassifier()

# A vision-forced layout for the degrade path: no trustworthy text, page-filling
# raster — so the classifier routes to VISION on every one of its three signals.
_VISION_FALLBACK_LAYOUT = PageLayout(
    text_chars=0, image_area_ratio=1.0, bitmap_area=1_000_000.0
)

# Rendered-image scale (2x ~= 144 DPI) — enough for a vision model to read a
# scan, small enough to keep the PNG cheap.
_RENDER_SCALE = 2.0

# A heading is a *short* line — a running paragraph set large is not a title.
_MAX_HEADING_WORDS = 14

# A leading section number like ``1``, ``2.3`` or ``4.1.2`` (then real text).
_NUMBER_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)\s+\S")

# Ruled tables come from pdfplumber's line-based strategy — drawn rules are strong
# evidence, so this is trusted outright.
_RULED_SETTINGS: dict[str, Any] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
}

# Borderless tables are found ourselves rather than by pdfplumber's whitespace
# strategy (which shreds ordinary prose into "columns"). A gap this wide between
# two words on a line is a column separator, not word spacing…
_COLUMN_GAP = 24.0
# …and a run of at least this many consecutive lines sharing the same column count
# is a table — one stray wide-gapped line is not.
_MIN_TABLE_ROWS = 2


def parse_pdf(source: RawSource, content: bytes) -> list[Page]:
    """Parse a PDF into one :class:`~ragsage.models.Page` per page, in order.

    Heavy parser libraries are imported here (never at module top) so importing
    :mod:`ragsage.parsing` stays dependency-light. Each page is parsed under its
    own guard: a failure degrades that page to the vision route rather than
    raising and losing the whole document.
    """
    import io

    import pdfplumber
    import pypdfium2 as pdfium

    renderer = pdfium.PdfDocument(content)
    try:
        pages: list[Page] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for index, plumber_page in enumerate(pdf.pages, start=1):
                pages.append(_parse_page(source, plumber_page, renderer, index))
        return pages
    finally:
        renderer.close()


def _parse_page(
    source: RawSource, plumber_page: Any, renderer: Any, number: int
) -> Page:
    """One page → a ``Page``, routing to vision (with a rendered image) as needed."""
    ref = f"{source.name}#page={number}"
    try:
        markdown, layout = _extract_page(plumber_page)
    except Exception:
        # A page pdfplumber can't make sense of is not a document-killer: hand it
        # to the vision route with a rendered image and a layout that forces it.
        return Page(
            number=number,
            text="",
            image=PageImage(ref=ref, data=_render_png(renderer, number - 1)),
            layout=_VISION_FALLBACK_LAYOUT,
        )

    if _routes_to_vision(layout, markdown):
        image: PageImage | None = PageImage(
            ref=ref, data=_render_png(renderer, number - 1)
        )
    else:
        image = None
    return Page(number=number, text=markdown, image=image, layout=layout)


def _routes_to_vision(layout: PageLayout, markdown: str) -> bool:
    """Whether the shared classifier would send this page to the vision route.

    Asking the real policy (rather than re-deriving its thresholds here) keeps the
    parser and the classifier from drifting apart.
    """
    probe = Page(number=0, text=markdown, image=PageImage(ref="probe"), layout=layout)
    return _CLASSIFIER.classify(probe) is PageRoute.VISION


# --------------------------------------------------------------------------- #
# Extraction: layout signals + Markdown body
# --------------------------------------------------------------------------- #


def _extract_page(page: Any) -> tuple[str, PageLayout]:
    words = page.extract_words(extra_attrs=["size", "fontname"])
    layout = _measure_layout(page, words)
    markdown = _serialize(page, words)
    return markdown, layout


def _measure_layout(page: Any, words: list[dict[str, Any]]) -> PageLayout:
    """Measure the geometric signals the classifier routes on.

    ``text_chars`` is the size of the born-digital text layer; ``image_area_ratio``
    the fraction of the page under raster images; ``bitmap_area`` the absolute area
    of the largest one — a page-filling scan reads high on the latter two even when
    a stray OCR character or two is present.
    """
    page_area = float(page.width) * float(page.height)
    text_chars = sum(len(w["text"]) for w in words)

    areas = [_image_area(image) for image in (page.images or [])]
    total_image_area = sum(areas)
    ratio = min(1.0, total_image_area / page_area) if page_area else 0.0
    return PageLayout(
        text_chars=text_chars,
        image_area_ratio=ratio,
        bitmap_area=max(areas) if areas else 0.0,
    )


def _image_area(image: dict[str, Any]) -> float:
    width = image.get("width")
    height = image.get("height")
    if width is None:
        width = float(image["x1"]) - float(image["x0"])
    if height is None:
        height = float(image["bottom"]) - float(image["top"])
    return abs(float(width) * float(height))


def _serialize(page: Any, words: list[dict[str, Any]]) -> str:
    """Serialise a page to Markdown in reading order, columns and tables respected.

    Columns are resolved first; tables are then looked for *within each column*
    (not across the whole page), which is what keeps a genuine two-column prose
    layout from being mistaken for one wide table by the borderless strategy.
    """
    # Ruled tables are found once on the whole page: their geometry is explicit, so
    # (unlike a per-column crop) nothing clips a table's outer rule, and column
    # detection already refuses to split a page on a table's inner gaps.
    ruled = _ruled_tables(page)
    bands = _column_bands(words, float(page.width))
    body_size = _modal_size(words)

    # Per column: its prose lines (once table rows are pulled) and its tables.
    per_column: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    prose_of_all: list[dict[str, Any]] = []
    for start, end in bands:
        band_ruled = [t for t in ruled if start <= _bbox_center_x(t["bbox"]) <= end]
        band_words = [
            w
            for w in words
            if start <= (float(w["x0"]) + float(w["x1"])) / 2 <= end
            and not _inside_any(w, [t["bbox"] for t in band_ruled])
        ]
        lines = _group_lines(band_words)
        borderless, consumed = _borderless_tables(lines)
        prose = [line for index, line in enumerate(lines) if index not in consumed]
        prose_of_all.extend(prose)
        per_column.append((prose, [*band_ruled, *borderless]))

    size_to_level = _heading_levels(prose_of_all, body_size)
    blocks: list[str] = []
    for prose, tables in per_column:
        blocks.extend(_emit_column([*prose, *tables], body_size, size_to_level))
    return "\n\n".join(block for block in blocks if block).strip() + "\n"


def _bbox_center_x(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[0] + bbox[2]) / 2


def _emit_column(
    items: list[dict[str, Any]],
    body_size: float,
    size_to_level: dict[float, int],
) -> list[str]:
    """Emit one column's items top-to-bottom, merging wrapped body lines.

    Consecutive body lines join into one paragraph block (so wrapped prose isn't
    shredded), while headings and tables stand alone as their own blocks — which is
    what lets the chunker split on the headings and keep each table whole.
    """
    ordered = sorted(items, key=lambda it: it["top"])
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append("\n".join(paragraph))
            paragraph.clear()

    for item in ordered:
        if item["kind"] == "table":
            flush()
            blocks.append(item["md"])
        elif _is_heading(item, body_size):
            flush()
            blocks.append(_render_heading(item, size_to_level))
        else:
            paragraph.append(item["text"])
    flush()
    return blocks


# --------------------------------------------------------------------------- #
# Lines, columns, headings
# --------------------------------------------------------------------------- #


def _group_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group words into text lines by vertical position, in reading order."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))
    lines: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    line_top: float | None = None
    for word in ordered:
        top = float(word["top"])
        height = float(word["bottom"]) - top
        tolerance = max(3.0, height * 0.6)
        if line_top is None or abs(top - line_top) <= tolerance:
            current.append(word)
            line_top = top if line_top is None else line_top
        else:
            lines.append(_make_line(current))
            current = [word]
            line_top = top
    if current:
        lines.append(_make_line(current))
    return lines


def _make_line(words: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(words, key=lambda w: float(w["x0"]))
    return {
        "kind": "text",
        "text": " ".join(w["text"] for w in ordered).strip(),
        "size": max(round(float(w["size"])) for w in ordered),
        "bold": any("bold" in str(w.get("fontname", "")).lower() for w in ordered),
        "top": min(float(w["top"]) for w in ordered),
        # (text, x0, x1) per word, in reading order — the borderless-table detector
        # reads the inter-word gaps to find column separators.
        "words": [(w["text"], float(w["x0"]), float(w["x1"])) for w in ordered],
    }


# A real page gutter runs (near) the full height of the page's text; each side of
# it must carry text spanning at least this fraction of that height. A table's
# inter-column gap fails this — the gap only exists across the table's few rows —
# which is what stops a table from being torn into separate "columns".
_MIN_COLUMN_HEIGHT_FRAC = 0.6


def _column_bands(
    words: list[dict[str, Any]], page_width: float
) -> list[tuple[float, float]]:
    """Find column x-ranges by locating the true vertical gutters between words.

    Word x-spans accumulated over *every* line fill the width of a single-column
    page, so no wide gap survives. A candidate gap only becomes a column boundary
    when text on *both* sides spans most of the page's text height — the test that
    tells a genuine two-column layout from the merely-wide gaps a table's columns
    leave.
    """
    if not words:
        return [(0.0, page_width)]

    text_top = min(float(w["top"]) for w in words)
    text_bottom = max(float(w["bottom"]) for w in words)
    text_height = text_bottom - text_top or 1.0

    spans = sorted((float(w["x0"]), float(w["x1"])) for w in words)
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    gutter = max(18.0, page_width * 0.04)
    separators = [
        (left[1] + right[0]) / 2
        for left, right in pairwise(merged)
        if right[0] - left[1] >= gutter
        and _both_sides_span_height(words, (left[1] + right[0]) / 2, text_height)
    ]
    if not separators:
        return [(merged[0][0], merged[-1][1])]

    edges = [merged[0][0], *separators, merged[-1][1]]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _both_sides_span_height(
    words: list[dict[str, Any]], split_x: float, text_height: float
) -> bool:
    """Whether text on each side of ``split_x`` spans most of the page's text height."""
    left = [w for w in words if (float(w["x0"]) + float(w["x1"])) / 2 < split_x]
    right = [w for w in words if (float(w["x0"]) + float(w["x1"])) / 2 >= split_x]
    if not left or not right:
        return False
    threshold = _MIN_COLUMN_HEIGHT_FRAC * text_height
    return _vertical_extent(left) >= threshold and _vertical_extent(right) >= threshold


def _vertical_extent(words: list[dict[str, Any]]) -> float:
    return max(float(w["bottom"]) for w in words) - min(float(w["top"]) for w in words)


def _modal_size(words: list[dict[str, Any]]) -> float:
    """The most common (rounded) font size — the document's body text size."""
    sizes = Counter(round(float(w["size"])) for w in words)
    if not sizes:
        return 0.0
    return float(sizes.most_common(1)[0][0])


def _heading_levels(
    lines: list[dict[str, Any]], body_size: float
) -> dict[float, int]:
    """Rank the distinct heading font sizes biggest-first into levels 1…4."""
    heading_sizes = {
        line["size"] for line in lines if _is_heading(line, body_size)
    }
    ordered = sorted(heading_sizes, reverse=True)
    return {size: min(index + 1, 4) for index, size in enumerate(ordered)}


def _is_heading(line: dict[str, Any], body_size: float) -> bool:
    """Whether a line reads as a heading: larger-than-body, or numbered/bold + short.

    A line set visibly larger than the modal body size and kept short is a heading.
    A same-size line is promoted only when a section number *and* shortness (plus
    bold, or the number itself) reinforce it — so a numbered body sentence isn't
    mistaken for a title.
    """
    text = line["text"]
    if not text:
        return False
    short = len(text.split()) <= _MAX_HEADING_WORDS
    larger = bool(body_size) and line["size"] >= body_size + 1
    numbered = bool(_NUMBER_PREFIX.match(text))
    if larger and short:
        return True
    if numbered and short and (larger or line["bold"]):
        return True
    return False


def _render_heading(line: dict[str, Any], size_to_level: dict[float, int]) -> str:
    """Render a heading line as a Markdown ATX heading at its ranked level.

    Font-size rank sets the base level; a numbered prefix refines it deeper (so
    ``1.1 Background`` nests under ``1 Introduction`` even when both are set in the
    same face).
    """
    level = size_to_level.get(line["size"], 3)
    match = _NUMBER_PREFIX.match(line["text"])
    if match:
        depth = match.group(1).count(".") + 1
        level = max(level, depth)
    level = min(max(level, 1), 4)
    return "#" * level + " " + str(line["text"])


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def _ruled_tables(page: Any) -> list[dict[str, Any]]:
    """Find tables drawn with rules via pdfplumber, each serialised to Markdown."""
    try:
        found = page.find_tables(table_settings=_RULED_SETTINGS)
    except Exception:
        return []
    tables: list[dict[str, Any]] = []
    for table in found:
        markdown = _table_to_markdown(table.extract())
        if markdown:
            bbox = tuple(float(v) for v in table.bbox)
            tables.append(
                {"kind": "table", "bbox": bbox, "top": bbox[1], "md": markdown}
            )
    return tables


def _borderless_tables(
    lines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int]]:
    """Find rule-less tables among prose lines, returning them and the lines they claim.

    A line splits into *cells* wherever an inter-word gap is wide enough to be a
    column separator rather than word spacing. A maximal run of consecutive lines
    that all split into the same number of cells (at least two) is a table; the
    first line is its header. Ordinary prose splits into a single cell, so it is
    never swept up — and a lone wide-gapped line is too short a run to qualify.
    """
    cells_per_line = [_split_cells(line) for line in lines]
    tables: list[dict[str, Any]] = []
    consumed: set[int] = set()
    start = 0
    total = len(lines)
    while start < total:
        width = len(cells_per_line[start])
        if width >= 2:
            end = start + 1
            while end < total and len(cells_per_line[end]) == width:
                end += 1
            if end - start >= _MIN_TABLE_ROWS:
                rows = [cells_per_line[index] for index in range(start, end)]
                markdown = _table_to_markdown(rows)
                if markdown:
                    tables.append(
                        {
                            "kind": "table",
                            "top": min(lines[i]["top"] for i in range(start, end)),
                            "md": markdown,
                        }
                    )
                    consumed.update(range(start, end))
                start = end
                continue
        start += 1
    return tables, consumed


def _split_cells(line: dict[str, Any]) -> list[str]:
    """Split a line into cell texts at inter-word gaps wider than a column gap."""
    words: list[tuple[str, float, float]] = line["words"]
    if not words:
        return []
    cells: list[list[str]] = [[words[0][0]]]
    for (_pt, _px0, prev_x1), (text, x0, _x1) in pairwise(words):
        if x0 - prev_x1 >= _COLUMN_GAP:
            cells.append([text])
        else:
            cells[-1].append(text)
    return [" ".join(cell) for cell in cells]


def _table_to_markdown(rows: Sequence[Sequence[str | None]]) -> str:
    """Serialise extracted rows to a GitHub pipe table (header + delimiter + data)."""
    cleaned = [
        [_clean_cell(cell) for cell in row]
        for row in (rows or [])
        if any(_clean_cell(cell) for cell in row)
    ]
    if len(cleaned) < 2:
        return ""  # a lone row isn't a table the chunker can keep whole
    ncols = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (ncols - len(row)) for row in cleaned]

    header = "| " + " | ".join(cleaned[0]) + " |"
    delimiter = "| " + " | ".join(["---"] * ncols) + " |"
    body = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]
    return "\n".join([header, delimiter, *body])


_WHITESPACE = re.compile(r"\s+")


def _clean_cell(cell: str | None) -> str:
    """Flatten a cell to one line and escape pipes so it can't break the table."""
    return _WHITESPACE.sub(" ", (cell or "").replace("|", r"\|")).strip()


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _inside_any(
    word: dict[str, Any], bboxes: list[tuple[float, float, float, float]]
) -> bool:
    """Whether a word's centre falls inside any table bbox (so it's the table's, not prose)."""
    cx = (float(word["x0"]) + float(word["x1"])) / 2
    cy = (float(word["top"]) + float(word["bottom"])) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)


def _render_png(renderer: Any, page_index: int) -> bytes | None:
    """Render a page to PNG bytes for the vision route, or ``None`` if it can't."""
    import io

    try:
        page = renderer[page_index]
        bitmap = page.render(scale=_RENDER_SCALE)
        image = bitmap.to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:  # pragma: no cover - a render failure still yields a Page
        return None
