"""The typed content-list item shapes — the input side of the proposed seam.

Mirrors the item shapes RAG-Anything's ``insert_content_list``
(``processor.py:2104``) documents, so the prototype measures the real contract
rather than a convenient subset:

    {"type": "text",     "text": ...,                                    "page_idx": 0}
    {"type": "image",    "img_path": ..., "image_caption": [...],
                                          "image_footnote": [...],       "page_idx": 1}
    {"type": "table",    "table_body": <markdown>, "table_caption": [...],
                                          "table_footnote": [...],       "page_idx": 2}
    {"type": "equation", "latex": ..., "text": ...,                      "page_idx": 3}
    {"type": <custom>,   "content": ...,                                 "page_idx": 4}

The seam takes raw mappings (that is what a foreign parser emits); this module
normalises them into a frozen ``ContentItem`` so both arms of the experiment read
the same thing. Nothing here is production code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

KNOWN_TYPES = ("text", "image", "table", "equation")


@dataclass(frozen=True)
class ContentItem:
    """One already-parsed block from a foreign parser.

    ``type`` is free-form on purpose — RAG-Anything routes anything it does not
    recognise to a generic handler, and the seam has to accept the same.
    ``payload`` keeps every field verbatim so nothing is silently dropped before
    the two arms get a chance to lose it.
    """

    type: str
    page_idx: int
    payload: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ContentItem:
        item_type = str(raw.get("type", "unknown"))
        page_idx = int(raw.get("page_idx", 0))  # type: ignore[arg-type]
        payload = {k: v for k, v in raw.items() if k not in ("type", "page_idx")}
        return cls(type=item_type, page_idx=page_idx, payload=payload)

    # -- the one thing both arms need: a Markdown rendering of the block ------

    def to_markdown(self) -> str:
        """Render this item as the Markdown the page-of-Markdown model speaks.

        This is the *only* lossless-looking step in arm (a): every item type has
        a plausible Markdown rendering. What it costs is measured in
        ``run_experiment.py`` — the rendering is where ``type`` stops existing.
        """
        if self.type == "text":
            return str(self.payload.get("text", "")).strip()
        if self.type == "table":
            return "\n\n".join(
                part
                for part in (
                    _join(self.payload.get("table_caption")),
                    str(self.payload.get("table_body", "")).strip(),
                    _join(self.payload.get("table_footnote")),
                )
                if part
            )
        if self.type == "image":
            caption = _join(self.payload.get("image_caption")) or "figure"
            path = str(self.payload.get("img_path", ""))
            footnote = _join(self.payload.get("image_footnote"))
            rendered = f"![{caption}]({path})"
            return f"{rendered}\n\n{footnote}" if footnote else rendered
        if self.type == "equation":
            latex = str(self.payload.get("latex", "")).strip()
            return f"$$\n{latex}\n$$" if latex else str(self.payload.get("text", "")).strip()
        # custom / unknown
        return str(self.payload.get("content", "")).strip()

    def caption(self) -> str:
        for key in ("table_caption", "image_caption"):
            if key in self.payload:
                return _join(self.payload[key])
        return ""

    def footnote(self) -> str:
        for key in ("table_footnote", "image_footnote"):
            if key in self.payload:
                return _join(self.payload[key])
        return ""


def _join(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def normalise(raw_items: Sequence[Mapping[str, object]]) -> list[ContentItem]:
    return [ContentItem.from_mapping(raw) for raw in raw_items]


# --------------------------------------------------------------------------- #
# The fixture both arms ingest
# --------------------------------------------------------------------------- #

SAMPLE_CONTENT_LIST: list[Mapping[str, object]] = [
    {
        "type": "text",
        "text": "# Orbital Freight Tariffs\n\nThis schedule governs low-earth-orbit "
        "cargo pricing for the 2027 season. Rates are quoted per kilogram and "
        "exclude insurance.",
        "page_idx": 0,
    },
    {
        "type": "text",
        "text": "## Rate schedule\n\nThe committee publishes one rate per destination "
        "band. Rebates are negotiated separately.",
        "page_idx": 0,
    },
    {
        "type": "table",
        "table_caption": ["Table 1: Per-kilogram tariff by destination band."],
        "table_body": (
            "| Band | Destination | Rate (USD/kg) |\n"
            "| --- | --- | --- |\n"
            "| A | Low earth orbit | 2400 |\n"
            "| B | Geostationary transfer | 5900 |\n"
            "| C | Lunar injection | 11200 |\n"
        ),
        "table_footnote": ["Rates exclude the 4% range-safety levy."],
        "page_idx": 0,
    },
    {
        "type": "text",
        "text": "## Scheduling\n\nManifests close ninety days before the launch window.",
        "page_idx": 1,
    },
    {
        "type": "image",
        "img_path": "/artifacts/launch-cadence.png",
        "image_caption": ["Figure 2: Launch cadence by quarter."],
        "image_footnote": ["Source: internal manifest registry."],
        "page_idx": 1,
    },
    {
        "type": "equation",
        "latex": r"C_{total} = m \cdot r_{band} \cdot (1 + \lambda)",
        "text": "Total cost equals mass times band rate times one plus the levy.",
        "page_idx": 1,
    },
    {
        "type": "checklist",  # a custom type — the seam must not reject it
        "content": "Pre-flight: manifest signed, levy paid, range window confirmed.",
        "page_idx": 1,
    },
]
