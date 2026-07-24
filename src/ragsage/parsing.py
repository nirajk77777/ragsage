"""Parser-neutral parsing helpers — currently the page-route classifier.

A :class:`~ragsage.ports.DocumentParser` measures each page (its text layer, how
much of it a raster image covers) and records that as a
:class:`~ragsage.models.PageLayout`. Deciding what to *do* with that measurement
— trust the text layer or send the page to vision — is a policy that knows
nothing about which parser produced it, so it belongs in the library beside the
port and the fake, not inside any one parser's adapter.

:class:`LayoutPageClassifier` is that policy. It reads only ``ragsage`` models,
imports no parser SDK, and is pure — which is what lets it be unit-tested with
hand-built pages and configured from the backend's composition root.
"""

from __future__ import annotations

from ragsage.models import Page, PageRoute


class LayoutPageClassifier:
    """``PageClassifier`` — routes each page on whether it has a *usable* text layer.

    A born-digital page carries enough extractable text to trust and route as
    ``TEXT``. A scanned, photographed, or handwritten page carries little or no
    text and is dominated by a raster bitmap; it has no usable text layer and,
    when the parser rendered an image for it, routes to the premium vision model.

    The decision reads the :class:`~ragsage.models.PageLayout` the parser
    measured, against three tunable thresholds so the policy is inspectable and
    unit-testable without any parser:

    * ``min_text_chars`` — below this many characters the text layer is treated as
      absent (a scan's stray or hidden-OCR characters don't make it "typed").
    * ``max_image_area_ratio`` — at or above this fraction of the page covered by
      images the page is image-dominated, so any thin text is not the real
      content and the page is read by vision instead.
    * ``min_bitmap_area`` — a single bitmap at least this large (in the page's own
      units) is a page-filling scan; paired with thin text it forces the vision
      route even if the image-area ratio reads low.

    With no layout measured (a flow format with no page grid, or a hand-built
    :class:`~ragsage.models.Page`) it falls back to whether a text layer is
    present at all — the same signal the earlier two-path scaffold used.
    """

    def __init__(
        self,
        *,
        min_text_chars: int = 24,
        max_image_area_ratio: float = 0.5,
        min_bitmap_area: float = 50_000.0,
    ) -> None:
        self._min_text_chars = min_text_chars
        self._max_image_area_ratio = max_image_area_ratio
        self._min_bitmap_area = min_bitmap_area

    def classify(self, page: Page) -> PageRoute:
        if self._has_usable_text_layer(page):
            return PageRoute.TEXT
        # No trustworthy text — send it to vision, but only if there is actually
        # a rendered image to transcribe; otherwise there is nothing to gain.
        if page.image is not None:
            return PageRoute.VISION
        return PageRoute.TEXT

    def _has_usable_text_layer(self, page: Page) -> bool:
        layout = page.layout
        if layout is None:
            # Nothing measured: trust a non-blank text layer, distrust a blank one.
            return bool(page.text.strip())
        if layout.text_chars < self._min_text_chars:
            return False
        if layout.image_area_ratio >= self._max_image_area_ratio:
            return False
        if layout.bitmap_area >= self._min_bitmap_area:
            return False
        return True
