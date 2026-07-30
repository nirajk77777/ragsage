"""Shared Seam-1 fixtures: a fully-wired engine over in-memory fakes.

Every ragsage test drives the public façades — :class:`IngestionPipeline`,
:class:`QueryEngine` — with injected fakes and asserts on what comes back out.
These fixtures assemble that wiring once. The ``pipeline`` and ``engine`` share a
single :class:`FakeEngineKit`, so a document ingested through one is queryable
through the other, exactly as the real backend wires them.

The storage fixtures at the bottom are the exception: Row-Level Security, a
generated ``tsvector`` column and pgvector's operators are all *database*
behaviour, and a fake cannot tell you whether the policy denies. Those tests are
marked ``integration`` and skip unless ``TEST_DATABASE_URL`` names a Postgres with
the ``vector`` extension available.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from ragsage import IngestionPipeline, QueryEngine, Scope
from ragsage.fakes import FakeEngineKit
from ragsage.storage import TABLE, Database, PostgresConfig, SchemaMismatch

#: Deliberately tiny. These tests assert on isolation and SQL, never on retrieval
#: quality, and an 8-wide vector keeps fixtures readable.
TEST_EMBEDDING_DIM = 8


@pytest.fixture
def kit() -> FakeEngineKit:
    return FakeEngineKit()


@pytest.fixture
def scope() -> Scope:
    return Scope(namespace="local")


@pytest.fixture
def pipeline(kit: FakeEngineKit) -> IngestionPipeline:
    return IngestionPipeline(
        parser=kit.parser,
        classifier=kit.classifier,
        chunker=kit.chunker,
        contextualizer=kit.contextualizer,
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        llm=kit.llm,
        cache=kit.cache,
        tracer=kit.tracer,
    )


@pytest.fixture
def engine(kit: FakeEngineKit) -> QueryEngine:
    return QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
        tracer=kit.tracer,
    )


# --------------------------------------------------------------------------- #
# Storage: real Postgres, or skip
# --------------------------------------------------------------------------- #


@pytest.fixture
def postgres_config() -> PostgresConfig:
    """Config pointed at the test database, or skip the test.

    Skipping rather than failing keeps the default suite offline: a contributor
    with no Postgres still gets a green run, and CI opts in by setting the variable.
    """
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set — skipping the real-Postgres storage tests")
    return PostgresConfig(dsn=dsn, embedding_dim=TEST_EMBEDDING_DIM)


@pytest.fixture
async def database(postgres_config: PostgresConfig) -> AsyncIterator[Database]:
    """A migrated database with an empty table.

    ``migrate`` runs per test *on purpose* — it is supposed to be idempotent, so
    calling it in every fixture is both the convenient thing and a standing check
    that it really is. The table is truncated rather than dropped so the indexes
    and the policy survive, which is what a real deployment looks like.
    """
    db = Database(postgres_config)
    try:
        await db.migrate()
    except SchemaMismatch:
        # The test database is throwaway and this fixture owns its schema, so a
        # left-over table at another width is recreated rather than reported. In
        # production that mismatch must fail loudly — a real corpus cannot be
        # dropped to satisfy a config — which is exactly why migrate() raises and
        # why the decision to override it belongs here, visibly, and nowhere else.
        async with db.owner_connection() as connection:
            await connection.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        await db.migrate()
    async with db.owner_connection() as connection:
        await connection.execute(text(f"TRUNCATE {TABLE}"))
    try:
        yield db
    finally:
        await db.dispose()
