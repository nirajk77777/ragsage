"""The three Postgres stores, against a real database.

These are integration tests by necessity rather than by preference. Cosine
distance, ``ts_rank``, a generated ``tsvector``, ``websearch_to_tsquery``'s
tolerance for junk input, and RLS are all *Postgres* behaviour — a fake would only
re-assert what the fake was written to do. The Seam-1 tests over
:class:`~ragsage.fakes.FakeEngineKit` already cover the pipeline logic above these.

The one thing asserted offline is the table description matching the migrated
schema, because two descriptions of one table is the drift risk this design takes on.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from conftest import TEST_EMBEDDING_DIM
from ragsage.models import Chunk, EmbeddedChunk
from ragsage.scope import Scope
from ragsage.storage import (
    TABLE,
    Database,
    PgDocumentStore,
    PgLexicalStore,
    PgVectorStore,
    purge_namespace,
)
from ragsage.storage.tables import WRITABLE_COLUMNS, chunks

pytestmark = pytest.mark.integration

_MODEL = "test-embedder"


def _vector(seed: float) -> tuple[float, ...]:
    """A deterministic unit-ish vector; only its direction matters here."""
    return tuple([seed] + [0.0] * (TEST_EMBEDDING_DIM - 1))


def _chunk(
    ref: str,
    text_body: str,
    *,
    document_id: str = "doc-1",
    ordinal: int = 0,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    return Chunk(
        id=ref,
        document_id=document_id,
        text=text_body,
        page=1,
        ordinal=ordinal,
        embed_text=text_body,
        metadata=metadata or {},
    )


def _embedded(chunk: Chunk, seed: float = 1.0) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=chunk, vector=_vector(seed))


# --------------------------------------------------------------------------- #
# The table description must not drift from the DDL
# --------------------------------------------------------------------------- #


async def test_the_table_description_matches_the_migrated_schema(database: Database) -> None:
    """Both directions: a column in the DDL but not here, or here but not in the DDL."""
    async with database.owner_connection() as connection:
        rows = await connection.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :t ORDER BY column_name"
            ),
            {"t": TABLE},
        )
        actual = {row[0] for row in rows}

    described = {column.name for column in chunks.columns}
    assert described == actual, (
        "ragsage.storage.tables has drifted from ragsage.storage.schema's DDL: "
        f"only in tables.py={described - actual}, only in the database={actual - described}"
    )


async def test_the_writable_columns_are_all_real_columns(database: Database) -> None:
    described = {column.name for column in chunks.columns}
    assert set(WRITABLE_COLUMNS) <= described
    # lexical and created_at are the database's to fill.
    assert "lexical" not in WRITABLE_COLUMNS
    assert "created_at" not in WRITABLE_COLUMNS


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


async def test_a_chunk_round_trips_with_its_metadata(database: Database) -> None:
    """The JSONB column has to survive a write *and* a read, or it is write-only."""
    scope = Scope(namespace="acme")
    payload = {"headings": ["Astronomy", "Greek devices"], "confidence": 0.5, "ok": True}

    async with database.session(scope) as session:
        store = PgVectorStore(session, embedding_model=_MODEL)
        await store.upsert(
            scope, [_embedded(_chunk("c:0", "an ancient analogue computer", metadata=payload))]
        )

    async with database.session(scope) as session:
        results = await PgVectorStore(session).search(scope, _vector(1.0), k=5)

    assert len(results) == 1
    restored = results[0].chunk
    assert restored.id == "c:0"
    assert restored.document_id == "doc-1"
    assert restored.text == "an ancient analogue computer"
    assert dict(restored.metadata) == payload


async def test_a_namespace_string_needs_no_uuid(database: Database) -> None:
    """The re-keying that mattered: the old adapter did uuid.UUID(namespace)."""
    scope = Scope(namespace="local")

    async with database.session(scope) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            scope, [_embedded(_chunk("c:0", "local corpus text"))]
        )

    async with database.session(scope) as session:
        results = await PgVectorStore(session).search(scope, _vector(1.0), k=5)

    assert [r.chunk.id for r in results] == ["c:0"]


async def test_upserting_nothing_needs_no_embedding_model(database: Database) -> None:
    scope = Scope(namespace="acme")
    async with database.session(scope) as session:
        await PgVectorStore(session).upsert(scope, [])  # must not raise


async def test_writing_vectors_without_a_model_is_refused(database: Database) -> None:
    scope = Scope(namespace="acme")
    async with database.session(scope) as session:
        with pytest.raises(ValueError, match="embedding_model"):
            await PgVectorStore(session).upsert(scope, [_embedded(_chunk("c:0", "text"))])


# --------------------------------------------------------------------------- #
# Search semantics
# --------------------------------------------------------------------------- #


async def test_dense_search_returns_cosine_similarity_not_distance(database: Database) -> None:
    """Higher must mean better, or RRF fuses the two halves backwards."""
    scope = Scope(namespace="acme")
    async with database.session(scope) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            scope,
            [
                _embedded(_chunk("near", "near text"), seed=1.0),
                _embedded(_chunk("far", "far text", ordinal=1), seed=-1.0),
            ],
        )

    async with database.session(scope) as session:
        results = await PgVectorStore(session).search(scope, _vector(1.0), k=2)

    assert [r.chunk.id for r in results] == ["near", "far"]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    assert results[1].score == pytest.approx(-1.0, abs=1e-6)


async def test_lexical_search_ranks_by_ts_rank(database: Database) -> None:
    scope = Scope(namespace="acme")
    async with database.session(scope) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            scope,
            [
                _embedded(_chunk("hit", "mitochondria produce cellular energy")),
                _embedded(_chunk("miss", "the seine flows through paris", ordinal=1)),
            ],
        )

    async with database.session(scope) as session:
        results = await PgLexicalStore(session).search(scope, "mitochondria", k=5)

    assert [r.chunk.id for r in results] == ["hit"]
    assert results[0].score > 0.0


@pytest.mark.parametrize("query", ["", "   ", '"', "-", "and or", "!!!", "a -"])
async def test_an_odd_lexical_query_matches_nothing_rather_than_raising(
    database: Database, query: str
) -> None:
    """An arbitrary user question goes straight through; none of this may raise."""
    scope = Scope(namespace="acme")
    async with database.session(scope) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            scope, [_embedded(_chunk("c:0", "some indexable prose"))]
        )

    async with database.session(scope) as session:
        results = await PgLexicalStore(session).search(scope, query, k=5)

    assert results == []


async def test_lexical_search_stems_through_the_generated_column(database: Database) -> None:
    scope = Scope(namespace="acme")
    async with database.session(scope) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            scope, [_embedded(_chunk("c:0", "the mitochondrion produces energy"))]
        )

    async with database.session(scope) as session:
        results = await PgLexicalStore(session).search(scope, "producing", k=5)

    assert [r.chunk.id for r in results] == ["c:0"]


# --------------------------------------------------------------------------- #
# The document filter: absent / present / present-but-empty
# --------------------------------------------------------------------------- #


async def _two_documents(database: Database, scope: Scope) -> None:
    async with database.session(scope) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            scope,
            [
                _embedded(_chunk("a:0", "alpha document text", document_id="doc-a")),
                _embedded(_chunk("b:0", "beta document text", document_id="doc-b", ordinal=1)),
            ],
        )


async def test_an_absent_document_filter_searches_the_whole_corpus(database: Database) -> None:
    scope = Scope(namespace="acme")
    await _two_documents(database, scope)

    async with database.session(scope) as session:
        dense = await PgVectorStore(session).search(scope, _vector(1.0), k=5)
        lexical = await PgLexicalStore(session).search(scope, "document", k=5)

    assert {r.chunk.id for r in dense} == {"a:0", "b:0"}
    assert {r.chunk.id for r in lexical} == {"a:0", "b:0"}


async def test_a_present_document_filter_narrows(database: Database) -> None:
    base = Scope(namespace="acme")
    await _two_documents(database, base)
    scope = base.with_filters(document_ids=["doc-a"])

    async with database.session(scope) as session:
        dense = await PgVectorStore(session).search(scope, _vector(1.0), k=5)
        lexical = await PgLexicalStore(session).search(scope, "document", k=5)

    assert {r.chunk.id for r in dense} == {"a:0"}
    assert {r.chunk.id for r in lexical} == {"a:0"}


async def test_a_present_but_empty_document_filter_matches_nothing(database: Database) -> None:
    """The caller named a set that resolved to no documents — that is not "search all".

    Getting this wrong would silently widen a deliberately narrow query into a
    whole-corpus search, which is the failure a user would never report as a bug.
    """
    base = Scope(namespace="acme")
    await _two_documents(database, base)
    scope = base.with_filters(document_ids=[])

    async with database.session(scope) as session:
        dense = await PgVectorStore(session).search(scope, _vector(1.0), k=5)
        lexical = await PgLexicalStore(session).search(scope, "document", k=5)

    assert dense == []
    assert lexical == []


# --------------------------------------------------------------------------- #
# Idempotence and purge
# --------------------------------------------------------------------------- #


async def test_re_upserting_the_same_chunks_replaces_rather_than_doubles(
    database: Database,
) -> None:
    scope = Scope(namespace="acme")
    records = [_embedded(_chunk("c:0", "first text"))]

    for _ in range(3):
        async with database.session(scope) as session:
            await PgVectorStore(session, embedding_model=_MODEL).upsert(scope, records)

    async with database.session(scope) as session:
        count = await session.scalar(text(f"SELECT count(*) FROM {TABLE}"))

    assert count == 1, "a retried ingest must be idempotent, not additive"


async def test_re_upserting_updated_text_overwrites_it(database: Database) -> None:
    scope = Scope(namespace="acme")

    async with database.session(scope) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            scope, [_embedded(_chunk("c:0", "original text"))]
        )
    async with database.session(scope) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            scope, [_embedded(_chunk("c:0", "revised text"))]
        )

    async with database.session(scope) as session:
        results = await PgVectorStore(session).search(scope, _vector(1.0), k=5)

    assert [r.chunk.text for r in results] == ["revised text"]


async def test_deleting_a_document_leaves_the_others(database: Database) -> None:
    scope = Scope(namespace="acme")
    await _two_documents(database, scope)

    async with database.session(scope) as session:
        await PgVectorStore(session).delete(scope, "doc-a")

    async with database.session(scope) as session:
        remaining = await PgVectorStore(session).search(scope, _vector(1.0), k=5)

    assert [r.chunk.id for r in remaining] == ["b:0"]


async def test_every_store_deletes_the_same_rows(database: Database) -> None:
    """One table, three ports — their deletes must agree and be safe to repeat."""
    scope = Scope(namespace="acme")

    for store_factory in (PgVectorStore, PgLexicalStore, PgDocumentStore):
        await _two_documents(database, scope)
        async with database.session(scope) as session:
            await store_factory(session).delete(scope, "doc-a")  # type: ignore[abstract]
            await store_factory(session).delete(scope, "doc-a")  # idempotent
        async with database.session(scope) as session:
            remaining = await PgVectorStore(session).search(scope, _vector(1.0), k=5)
        assert [r.chunk.id for r in remaining] == ["b:0"]
        async with database.session(scope) as session:
            await purge_namespace(session)


async def test_purging_a_namespace_removes_everything_in_it_and_nothing_else(
    database: Database,
) -> None:
    acme, globex = Scope(namespace="acme"), Scope(namespace="globex")
    await _two_documents(database, acme)
    await _two_documents(database, globex)

    async with database.session(acme) as session:
        await purge_namespace(session)

    async with database.session(acme) as session:
        assert await PgVectorStore(session).search(acme, _vector(1.0), k=5) == []
    async with database.session(globex) as session:
        survivors = await PgVectorStore(session).search(globex, _vector(1.0), k=5)

    assert {r.chunk.id for r in survivors} == {"a:0", "b:0"}


async def test_purging_an_empty_namespace_is_fine(database: Database) -> None:
    async with database.session(Scope(namespace="never-used")) as session:
        await purge_namespace(session)
        await purge_namespace(session)


# --------------------------------------------------------------------------- #
# Isolation, through the stores rather than raw SQL
# --------------------------------------------------------------------------- #


async def test_one_namespace_cannot_search_another(database: Database) -> None:
    acme, globex = Scope(namespace="acme"), Scope(namespace="globex")

    async with database.session(acme) as session:
        await PgVectorStore(session, embedding_model=_MODEL).upsert(
            acme, [_embedded(_chunk("acme:0", "acme confidential text"))]
        )

    async with database.session(globex) as session:
        dense = await PgVectorStore(session).search(globex, _vector(1.0), k=5)
        lexical = await PgLexicalStore(session).search(globex, "confidential", k=5)
        listed = await PgDocumentStore(session).list(globex)

    assert dense == []
    assert lexical == []
    assert listed == []


# --------------------------------------------------------------------------- #
# The document store's deliberate shape
# --------------------------------------------------------------------------- #


async def test_the_document_store_stays_a_no_op_on_identity(database: Database) -> None:
    """Not an oversight: dedup and identity belong to the consumer (ADR-0002)."""
    scope = Scope(namespace="acme")
    async with database.session(scope) as session:
        store = PgDocumentStore(session)

        assert await store.find_by_hash(scope, "any-hash-at-all") is None
        assert await store.save(scope, _reconstructable(), []) is None


async def test_the_document_store_reconstructs_what_was_ingested(database: Database) -> None:
    scope = Scope(namespace="acme")
    await _two_documents(database, scope)

    async with database.session(scope) as session:
        store = PgDocumentStore(session)
        listed = await store.list(scope)
        found = await store.get(scope, "doc-a")
        missing = await store.get(scope, "doc-nope")

    assert {d.id for d in listed} == {"doc-a", "doc-b"}
    assert found is not None and found.id == "doc-a"
    assert missing is None


def _reconstructable() -> object:
    from ragsage.models import Document

    return Document(id="doc-1", source="doc-1", content_hash="h")
