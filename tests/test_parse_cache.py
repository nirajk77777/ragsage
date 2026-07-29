"""The parse cache: a second ingest of identical bytes must not re-parse.

The scenario that matters is not a repeated upload — the document store dedups
that before parsing is even reached. It is **re-indexing**: fresh stores, same
corpus, because the embedding model or the chunk size changed. That path parses
every document again today, and it is the reason this cache exists.

So these tests share one cache across otherwise-independent ingests and count
what the parser actually did.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragsage.caching import decode_parsed, encode_parsed, parse_cache_key
from ragsage.config import IngestionConfig
from ragsage.fakes import FakeEngineKit
from ragsage.ingestion import IngestionPipeline
from ragsage.models import Chunk, Document, Page, PageImage, PageLayout, ParsedDocument, RawSource
from ragsage.parsing import HeuristicBackend
from ragsage.scope import Scope

_BODY = b"""# Catalog

## Fruit

Apples are crisp and keep well in a cold store.

Pears bruise easily and should be handled gently.
"""


class CountingBackend:
    """A real :class:`HeuristicBackend` that records how often it parsed.

    Wraps rather than subclasses so it also proves the pipeline only uses the
    ports' surface — anything relying on the concrete class would break here.
    """

    def __init__(self) -> None:
        self._backend = HeuristicBackend()
        self.parse_calls = 0

    def parse(self, source: RawSource) -> ParsedDocument:
        self.parse_calls += 1
        return self._backend.parse(source)

    def chunk(
        self, document: Document, pages: Sequence[Page], *, size: int, overlap: int
    ) -> Sequence[Chunk]:
        return self._backend.chunk(document, pages, size=size, overlap=overlap)


def _pipeline(backend: CountingBackend, kit: FakeEngineKit, cache: object) -> IngestionPipeline:
    return IngestionPipeline(
        parser=backend,
        classifier=kit.classifier,
        chunker=backend,
        contextualizer=kit.contextualizer,
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        llm=kit.llm,
        cache=cache,  # type: ignore[arg-type]
        tracer=kit.tracer,
    )


async def test_a_reindex_over_a_shared_cache_parses_once() -> None:
    """The headline requirement: identical bytes, second ingest, no parse."""
    cache = FakeEngineKit().cache
    source = RawSource(name="catalog.md", content=_BODY)

    first_backend = CountingBackend()
    first = _pipeline(first_backend, FakeEngineKit(), cache)
    first_result = await first.ingest(source, Scope(namespace="corpus"))

    # Fresh stores, same cache — the re-index case.
    second_backend = CountingBackend()
    second = _pipeline(second_backend, FakeEngineKit(), cache)
    second_result = await second.ingest(source, Scope(namespace="corpus"))

    assert first_backend.parse_calls == 1
    assert second_backend.parse_calls == 0, "a warm parse cache must skip the parser entirely"

    # And the cached path is not a degraded one: same document, same chunks.
    assert second_result.document == first_result.document
    assert second_result.chunk_count == first_result.chunk_count
    assert second_result.chunk_count > 0


async def test_the_pipeline_is_correct_with_no_cache_at_all() -> None:
    """The port's stated contract: the engine is correct without the cache.

    A cache that returns nothing and stores nothing is the degenerate adapter a
    consumer gets by injecting a stub, so it must simply mean "always parse".
    """

    class NullCache:
        async def get(self, key: str) -> str | None:
            return None

        async def set(self, key: str, value: str) -> None:
            return None

    backend = CountingBackend()
    pipeline = _pipeline(backend, FakeEngineKit(), NullCache())
    result = await pipeline.ingest(
        RawSource(name="catalog.md", content=_BODY), Scope(namespace="c")
    )

    assert backend.parse_calls == 1
    assert result.chunk_count > 0


async def test_a_broken_cache_degrades_to_a_miss_rather_than_failing_the_ingest() -> None:
    """A dead Redis must be a slowdown, not an outage."""

    class BrokenCache:
        async def get(self, key: str) -> str | None:
            raise ConnectionError("redis is down")

        async def set(self, key: str, value: str) -> None:
            raise ConnectionError("redis is down")

    backend = CountingBackend()
    pipeline = _pipeline(backend, FakeEngineKit(), BrokenCache())
    result = await pipeline.ingest(
        RawSource(name="catalog.md", content=_BODY),
        Scope(namespace="c"),
        IngestionConfig(contextualize=True),
    )

    assert backend.parse_calls == 1
    assert result.chunk_count > 0


async def test_different_bytes_do_not_collide() -> None:
    cache = FakeEngineKit().cache

    first_backend = CountingBackend()
    await _pipeline(first_backend, FakeEngineKit(), cache).ingest(
        RawSource(name="a.md", content=_BODY), Scope(namespace="c")
    )

    other_backend = CountingBackend()
    await _pipeline(other_backend, FakeEngineKit(), cache).ingest(
        RawSource(name="b.md", content=b"# Other\n\nCompletely different content.\n"),
        Scope(namespace="c"),
    )

    assert other_backend.parse_calls == 1, "different content must not hit another document's entry"


async def test_the_same_bytes_hit_after_a_rename() -> None:
    """Keyed by content, not filename — a moved or renamed file still hits."""
    cache = FakeEngineKit().cache

    await _pipeline(CountingBackend(), FakeEngineKit(), cache).ingest(
        RawSource(name="catalog.md", content=_BODY), Scope(namespace="c")
    )

    renamed_backend = CountingBackend()
    await _pipeline(renamed_backend, FakeEngineKit(), cache).ingest(
        RawSource(name="catalog-2024-final.md", content=_BODY), Scope(namespace="c")
    )

    assert renamed_backend.parse_calls == 0


# --------------------------------------------------------------------------- #
# Keys and the wire format
# --------------------------------------------------------------------------- #


def test_the_key_carries_the_parser_identity_and_the_format_version() -> None:
    key = parse_cache_key("abc123", HeuristicBackend())

    assert key.startswith("parse:v")
    assert "HeuristicBackend" in key
    assert key.endswith("abc123")


def test_a_different_parser_does_not_read_another_parser_s_entry() -> None:
    class OtherParser:
        pass

    assert parse_cache_key("abc123", HeuristicBackend()) != parse_cache_key("abc123", OtherParser())


def test_pages_survive_the_round_trip_including_image_bytes_and_layout() -> None:
    kit = FakeEngineKit()
    parsed = kit.parser.parse(RawSource(name="catalog.md", content=_BODY))
    original = type(parsed)(
        document=parsed.document,
        pages=[
            Page(number=1, text="page one"),
            Page(
                number=2,
                text="",
                image=PageImage(ref="page-2.png", data=b"\x89PNG\r\n\x1a\nnot-really"),
                layout=PageLayout(text_chars=0, image_area_ratio=0.97, bitmap_area=51_000.0),
            ),
        ],
    )

    restored = decode_parsed(encode_parsed(original))

    assert restored is not None
    assert restored.document == original.document
    assert [p.text for p in restored.pages] == ["page one", ""]
    # The image is the whole point of a cached scanned page — losing it would turn
    # a hit into an empty document.
    assert restored.pages[1].image is not None
    assert restored.pages[1].image.data == b"\x89PNG\r\n\x1a\nnot-really"
    assert restored.pages[1].layout is not None
    assert restored.pages[1].layout.bitmap_area == 51_000.0


def test_a_malformed_entry_is_treated_as_a_miss() -> None:
    for payload in ("", "not json", "{}", '{"document": {}}', '{"pages": []}'):
        assert decode_parsed(payload) is None, f"{payload!r} should decode to a miss"
