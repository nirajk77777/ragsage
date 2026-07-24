"""Unit coverage for :class:`ragsage.parsing.LayoutPageClassifier`.

The classifier is pure and parser-agnostic — it reads only a :class:`Page` and
its measured :class:`PageLayout` — so its whole routing policy is exercised here
with hand-built pages: the text-presence fallback when nothing was measured, and
the thin-text / image-area / bitmap thresholds when it was.
"""

from __future__ import annotations

from ragsage.models import Page, PageImage, PageLayout, PageRoute
from ragsage.parsing import LayoutPageClassifier

_SCAN = PageImage(ref="scan#page=1", data=b"\x89PNG")


# ---------------------------------------------------------------------------
# Routing with no measured layout — the text-presence fallback
# ---------------------------------------------------------------------------


def test_text_page_routes_as_text() -> None:
    assert LayoutPageClassifier().classify(Page(number=1, text="real content")) is PageRoute.TEXT


def test_image_only_page_routes_to_vision() -> None:
    page = Page(number=2, text="   ", image=PageImage(ref="scan#page=2"))
    assert LayoutPageClassifier().classify(page) is PageRoute.VISION


def test_blank_page_without_image_defaults_to_text() -> None:
    # Nothing to transcribe — routing to vision would only waste a model call.
    assert LayoutPageClassifier().classify(Page(number=3, text="")) is PageRoute.TEXT


# ---------------------------------------------------------------------------
# Routing on measured layout — the thin-text / image-area / bitmap thresholds
# ---------------------------------------------------------------------------


def test_typed_page_with_small_figure_routes_as_text() -> None:
    # A long text layer with a small inline figure is born-digital: keep the text.
    page = Page(
        number=1,
        text="a paragraph of real body text that is comfortably above the floor",
        image=_SCAN,
        layout=PageLayout(text_chars=200, image_area_ratio=0.1, bitmap_area=1_000.0),
    )
    assert LayoutPageClassifier().classify(page) is PageRoute.TEXT


def test_thin_text_page_routes_to_vision() -> None:
    # Almost no extractable text but a rendered image to read — a scan.
    page = Page(
        number=1,
        text="p. 4",
        image=_SCAN,
        layout=PageLayout(text_chars=4, image_area_ratio=0.0, bitmap_area=0.0),
    )
    assert LayoutPageClassifier().classify(page) is PageRoute.VISION


def test_image_dominated_page_routes_to_vision_despite_hidden_text() -> None:
    # Enough characters to pass the text floor, but a raster image covers the page
    # (a scan carrying a garbage hidden-OCR layer) — trust the pixels, not the text.
    page = Page(
        number=1,
        text="stray hidden ocr text long enough to clear the character floor here",
        image=_SCAN,
        layout=PageLayout(text_chars=120, image_area_ratio=0.92, bitmap_area=400_000.0),
    )
    assert LayoutPageClassifier().classify(page) is PageRoute.VISION


def test_large_bitmap_forces_vision_even_at_low_ratio() -> None:
    # A single page-filling bitmap trips the absolute-area threshold on its own.
    page = Page(
        number=1,
        text="caption",
        image=_SCAN,
        layout=PageLayout(text_chars=7, image_area_ratio=0.2, bitmap_area=120_000.0),
    )
    assert LayoutPageClassifier().classify(page) is PageRoute.VISION


def test_thin_text_without_rendered_image_stays_text() -> None:
    # No usable text layer *and* no image to transcribe — vision would send
    # nothing, so the page stays on the text path rather than wasting a call.
    page = Page(
        number=1,
        text="",
        layout=PageLayout(text_chars=0, image_area_ratio=0.9, bitmap_area=500_000.0),
    )
    assert LayoutPageClassifier().classify(page) is PageRoute.TEXT


def test_thresholds_are_configurable() -> None:
    page = Page(
        number=1,
        text="short",
        image=_SCAN,
        layout=PageLayout(text_chars=5, image_area_ratio=0.0, bitmap_area=0.0),
    )
    # A relaxed floor accepts the same page's text layer that the default rejects.
    assert LayoutPageClassifier().classify(page) is PageRoute.VISION
    assert LayoutPageClassifier(min_text_chars=1).classify(page) is PageRoute.TEXT
