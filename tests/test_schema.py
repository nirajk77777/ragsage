"""The schema: what the DDL says, and what Postgres actually enforces.

Two halves, deliberately. The offline half asserts the *shape* of the generated
DDL — that the dimension and text-search configuration came from config rather
than a literal, that the generated column kept the two-argument ``to_tsvector``
that a STORED column requires, that RLS is forced and not merely enabled. Those
are cheap, run everywhere, and catch the mistakes that are easy to make while
editing SQL in a string.

The integration half is the half that matters. No amount of string assertion tells
you whether the policy *denies*, so the isolation claim is checked against a real
Postgres: two namespaces, one table, a query with no owner predicate at all.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy import text

from ragsage.scope import Scope
from ragsage.storage import (
    Database,
    PostgresConfig,
    SchemaMismatch,
    existing_embedding_dim,
    migrate,
)
from ragsage.storage.schema import (
    POLICY,
    TABLE,
    create_table_sql,
    index_statements,
    migration_statements,
    policy_sql,
)

_DSN = "postgresql://user:pw@localhost/db"


def _config(**overrides: object) -> PostgresConfig:
    return PostgresConfig(dsn=_DSN, **overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The DDL, offline
# --------------------------------------------------------------------------- #


def test_the_embedding_width_comes_from_config() -> None:
    assert "VECTOR(1024)" in create_table_sql(_config())
    assert "VECTOR(384)" in create_table_sql(_config(embedding_dim=384))


def test_the_text_search_configuration_comes_from_config() -> None:
    assert "to_tsvector('english', embed_text)" in create_table_sql(_config())
    assert "to_tsvector('simple', embed_text)" in create_table_sql(
        _config(text_search_config="simple")
    )


def test_the_generated_column_keeps_the_two_argument_to_tsvector() -> None:
    """One-argument ``to_tsvector`` is only STABLE; a STORED column needs IMMUTABLE.

    Postgres rejects the one-argument form outright, so getting this wrong fails at
    migrate time — but it fails with a message about immutability that does not
    obviously point here, which is worth a test that names the reason.
    """
    ddl = create_table_sql(_config())

    assert "GENERATED ALWAYS AS" in ddl
    assert "STORED" in ddl
    assert "to_tsvector('english', embed_text)" in ddl


def test_namespace_is_a_column_and_part_of_the_primary_key() -> None:
    ddl = create_table_sql(_config())

    assert "namespace       TEXT NOT NULL" in ddl
    assert "PRIMARY KEY (namespace, chunk_ref)" in ddl


def test_metadata_defaults_to_an_empty_object_rather_than_null() -> None:
    # A NULL default would force every reader to handle two empty cases.
    assert "metadata        JSONB NOT NULL DEFAULT '{}'::jsonb" in create_table_sql(_config())


def test_the_indexes_cover_ann_lexical_and_the_purge_path() -> None:
    statements = " | ".join(index_statements())

    # vector_cosine_ops must match the cosine_distance operator the store searches
    # with, or the index is built and silently never used.
    assert "USING hnsw (embedding vector_cosine_ops)" in statements
    assert "USING gin (lexical)" in statements
    assert f"ON {TABLE} (namespace, rag_document_id)" in statements


def test_the_policy_guards_reads_and_writes_with_missing_ok_semantics() -> None:
    sql = policy_sql(_config())

    predicate = "namespace = current_setting('ragsage.namespace', true)"
    assert f"USING ({predicate})" in sql
    assert f"WITH CHECK ({predicate})" in sql


def test_the_policy_follows_a_configured_isolation_variable() -> None:
    sql = policy_sql(_config(isolation_variable="acme.tenant"))

    assert "current_setting('acme.tenant', true)" in sql
    assert "ragsage.namespace" not in sql


def test_rls_is_forced_not_merely_enabled() -> None:
    """ENABLE alone exempts the owner — and ragsage connects as the owner."""
    statements = migration_statements(_config())

    assert f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY" in statements
    assert f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY" in statements


def test_migration_creates_the_extension_before_the_table() -> None:
    statements = migration_statements(_config())

    extension = next(i for i, s in enumerate(statements) if "CREATE EXTENSION" in s)
    table = next(i for i, s in enumerate(statements) if "CREATE TABLE" in s)
    assert extension < table, "the vector type must exist before a column can use it"


def test_every_statement_is_safe_to_rerun() -> None:
    """Idempotence, asserted structurally rather than by running it twice offline."""
    for statement in migration_statements(_config()):
        collapsed = " ".join(statement.split())
        idempotent = (
            "IF NOT EXISTS" in collapsed
            or "DROP POLICY IF EXISTS" in collapsed
            # ALTER/GRANT/CREATE POLICY are re-runnable: ALTER and GRANT are
            # naturally so, and the policy is dropped immediately before.
            or collapsed.startswith(("ALTER TABLE", "GRANT ", f"CREATE POLICY {POLICY}"))
        )
        assert idempotent, f"not safe to re-run: {collapsed[:80]}"


def test_the_app_role_is_created_without_login() -> None:
    statements = " | ".join(migration_statements(_config(app_role="rag_app")))

    assert "CREATE ROLE rag_app NOLOGIN" in statements
    assert f"GRANT SELECT, INSERT, UPDATE, DELETE ON {TABLE} TO rag_app" in statements


def test_a_hostile_text_search_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        _config(text_search_config="english'); DROP TABLE ragsage_chunks; --")


@pytest.mark.parametrize("dim", [0, -1, 2001])
def test_an_unusable_embedding_dimension_is_rejected(dim: int) -> None:
    with pytest.raises(ValueError, match="embedding_dim"):
        _config(embedding_dim=dim)


# --------------------------------------------------------------------------- #
# What Postgres enforces
# --------------------------------------------------------------------------- #


async def _insert(database: Database, namespace: str, ref: str, text_body: str) -> None:
    async with database.session(Scope(namespace=namespace)) as session:
        await session.execute(
            text(
                f"INSERT INTO {TABLE} "
                "(namespace, chunk_ref, rag_document_id, text, embed_text, page, ordinal, "
                " embedding, embedding_model) "
                "VALUES (:ns, :ref, 'doc-1', :body, :body, 1, 0, :vec, 'test-model')"
            ),
            {"ns": namespace, "ref": ref, "body": text_body, "vec": str([0.1] * 8)},
        )


@pytest.mark.integration
async def test_migrate_is_idempotent_against_a_real_database(database: Database) -> None:
    # The fixture already migrated once; twice more must be a no-op, not an error.
    await database.migrate()
    await database.migrate()

    async with database.owner_connection() as connection:
        count = await connection.scalar(
            text("SELECT count(*) FROM pg_policies WHERE tablename = :t"), {"t": TABLE}
        )
    assert count == 1, "re-migrating must not accumulate duplicate policies"


@pytest.mark.integration
async def test_two_namespaces_cannot_see_each_other(database: Database) -> None:
    """The headline isolation claim, with no owner predicate in the query at all.

    This is the test the whole storage design exists to pass. The SELECT says
    ``WHERE 1=1`` — nothing about namespaces — and still returns only the caller's
    rows, because the policy adds the predicate the query omitted.
    """
    await _insert(database, "acme", "acme:0", "acme private text")
    await _insert(database, "globex", "globex:0", "globex private text")

    async with database.session(Scope(namespace="acme")) as session:
        rows = (await session.execute(text(f"SELECT chunk_ref FROM {TABLE} WHERE 1=1"))).scalars()
        visible = sorted(rows)

    assert visible == ["acme:0"], "a namespace must not see another namespace's rows"


@pytest.mark.integration
async def test_a_write_cannot_be_stamped_with_another_namespace(database: Database) -> None:
    """WITH CHECK, not just USING: forging the column must be refused."""
    with pytest.raises(Exception, match=r"row-level security|violates"):
        async with database.session(Scope(namespace="acme")) as session:
            await session.execute(
                text(
                    f"INSERT INTO {TABLE} "
                    "(namespace, chunk_ref, rag_document_id, text, embed_text, page, ordinal, "
                    " embedding, embedding_model) "
                    "VALUES ('globex', 'forged', 'doc-1', 't', 't', 1, 0, :vec, 'm')"
                ),
                {"vec": str([0.1] * 8)},
            )


@pytest.mark.integration
async def test_an_unset_isolation_variable_denies_rather_than_leaks(database: Database) -> None:
    """missing_ok semantics: no variable means no rows, never every row.

    Run as the app role *without* the preamble, which is what a store that forgot
    to open a scoped session would do.
    """
    await _insert(database, "acme", "acme:0", "acme private text")

    async with database.owner_connection() as connection:
        await connection.execute(text(f'SET LOCAL ROLE "{database.config.app_role}"'))
        visible = (await connection.execute(text(f"SELECT chunk_ref FROM {TABLE}"))).scalars().all()

    assert list(visible) == [], "an unset isolation variable must deny, not expose everything"


@pytest.mark.integration
async def test_the_lexical_column_is_generated_by_the_database(database: Database) -> None:
    await _insert(database, "acme", "acme:0", "mitochondria produce cellular energy")

    async with database.session(Scope(namespace="acme")) as session:
        matched = await session.scalar(
            text(
                f"SELECT count(*) FROM {TABLE} WHERE lexical @@ websearch_to_tsquery('english', :q)"
            ),
            {"q": "mitochondria"},
        )
        # Stemming proves it is a real tsvector, not a copy of the text.
        stemmed = await session.scalar(
            text(
                f"SELECT count(*) FROM {TABLE} WHERE lexical @@ websearch_to_tsquery('english', :q)"
            ),
            {"q": "producing"},
        )

    assert matched == 1
    assert stemmed == 1, "to_tsvector should stem 'produce'/'producing' together"


@pytest.mark.integration
async def test_migrating_at_a_different_embedding_width_fails_loudly(
    database: Database,
) -> None:
    """``CREATE TABLE IF NOT EXISTS`` would accept this silently; migrate() must not.

    The failure it replaces is nasty: the width disagreement surfaces at the first
    insert, after a document has been parsed and embedded, as a driver-level
    "expected N dimensions, not M" with nothing pointing at the config that caused it.
    """
    wrong = replace(database.config, embedding_dim=database.config.embedding_dim + 1)

    with pytest.raises(SchemaMismatch, match=r"embedding_dim"):
        async with database.owner_connection() as connection:
            await migrate(connection, wrong)


@pytest.mark.integration
async def test_the_mismatch_error_names_both_widths_and_the_way_out(
    database: Database,
) -> None:
    """An error a reader can act on without opening the source."""
    actual = database.config.embedding_dim
    wrong = replace(database.config, embedding_dim=384)

    with pytest.raises(SchemaMismatch) as raised:
        async with database.owner_connection() as connection:
            await migrate(connection, wrong)

    message = str(raised.value)
    assert f"vector({actual})" in message
    assert "384" in message
    assert "re-index" in message


@pytest.mark.integration
async def test_a_matching_width_still_migrates(database: Database) -> None:
    """The guard must not make the ordinary re-migration path fail."""
    async with database.owner_connection() as connection:
        await migrate(connection, database.config)
        assert await existing_embedding_dim(connection) == database.config.embedding_dim


@pytest.mark.integration
async def test_the_width_is_unknown_before_the_table_exists(database: Database) -> None:
    """A fresh database has nothing to disagree with, so the guard must stay quiet.

    This is the one test that drops the table, so it restores it at the fixture's own
    width on the way out. Without that, every later test's fixture ``migrate()``
    would hit the very :class:`SchemaMismatch` this file is about — an
    order-dependent failure that looks like a bug in whatever ran next.
    """
    try:
        async with database.owner_connection() as connection:
            await connection.execute(text(f"DROP TABLE {TABLE}"))
            assert await existing_embedding_dim(connection) is None
            # …and on a fresh table, any width is accepted.
            await migrate(connection, replace(database.config, embedding_dim=384))
            assert await existing_embedding_dim(connection) == 384
    finally:
        async with database.owner_connection() as connection:
            await connection.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
            await migrate(connection, database.config)
