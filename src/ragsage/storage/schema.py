"""The one table ragsage owns, and the idempotent ``migrate()`` that creates it.

ADR-0002: ragsage owns ``ragsage_chunks`` — its columns, its indexes, its
non-privileged role, and the Row-Level Security policy that makes it safe to
share one table between namespaces. The consumer supplies a DSN and calls
:func:`migrate`; it writes no DDL of its own.

**``namespace`` is a real, indexed column — not a metadata key** (ADR-0002 §2).
That is the whole isolation design: a policy can compare a column, an index can
serve it, and the planner can use it. ``metadata`` is JSONB payload the consumer
attaches and may filter on, and it is deliberately *not* the isolation mechanism
— nothing on the ANN path reads it.

Everything here is written as ``IF NOT EXISTS``/``DROP … IF EXISTS`` so
:func:`migrate` is safe on a fresh database, on an already-migrated one, on every
process start, and in every test fixture. It is not a migration *framework*: there
is one table and no versioned history, so a hand-written idempotent script is the
honest amount of machinery. A column added later needs an explicit ``ADD COLUMN IF
NOT EXISTS`` here, and a change to ``embedding_dim`` or ``text_search_config``
needs the table rebuilt — neither is silently handled, and both are called out on
:class:`~ragsage.storage.config.PostgresConfig`.

The DDL runs on an **owner** connection, which bypasses RLS. That is unavoidable —
only the owner may create the table and the policy — and it is exactly why
:meth:`~ragsage.storage.engine.Database.owner_connection` is separate from, and
more awkward than, the scoped session every store actually uses.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from ragsage.storage.config import PostgresConfig
from ragsage.storage.identifiers import safe_identifier

#: The table name is fixed, not configurable. It is already namespaced by its
#: ``ragsage_`` prefix, and a configurable name would have to be interpolated into
#: every statement in this module for no benefit a schema cannot provide.
TABLE = "ragsage_chunks"

POLICY = "ragsage_namespace_isolation"


def create_table_sql(config: PostgresConfig) -> str:
    """The ``CREATE TABLE`` for :data:`TABLE`, sized and configured by ``config``.

    Exposed rather than inlined so a test can assert on the generated DDL — that
    the dimension and the text-search configuration came from config, and that the
    generated column kept its two-argument ``to_tsvector`` — without a database.
    """
    return f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            namespace       TEXT NOT NULL,
            chunk_ref       TEXT NOT NULL,
            rag_document_id TEXT NOT NULL,
            text            TEXT NOT NULL,
            embed_text      TEXT NOT NULL,
            page            INTEGER NOT NULL,
            ordinal         INTEGER NOT NULL,
            embedding       VECTOR({config.embedding_dim}) NOT NULL,
            embedding_model TEXT NOT NULL,
            metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            lexical         TSVECTOR GENERATED ALWAYS AS (
                                to_tsvector('{safe_identifier(config.text_search_config)}', embed_text)
                            ) STORED,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (namespace, chunk_ref)
        )
    """


def policy_sql(config: PostgresConfig) -> str:
    """The RLS policy: a row is visible only inside its own namespace.

    ``USING`` filters reads, updates and deletes; ``WITH CHECK`` stops a write
    stamped with someone else's namespace. Both compare the column against the
    isolation variable, so neither direction can be forgotten independently.

    ``current_setting(…, true)`` is the ``missing_ok`` form: when the variable was
    never set it returns NULL, ``namespace = NULL`` is NULL rather than true, and
    the policy therefore **denies**. An unset variable that returned *everything*
    would be the worst possible failure mode, so this is load-bearing — a store
    that forgot the preamble sees an empty table, not another namespace's rows.
    """
    variable = config.isolation_variable
    predicate = f"namespace = current_setting('{variable}', true)"
    return f"CREATE POLICY {POLICY} ON {TABLE} USING ({predicate}) WITH CHECK ({predicate})"


def index_statements() -> tuple[str, ...]:
    """The three indexes the read and purge paths need.

    * HNSW on cosine distance — the ANN path. ``vector_cosine_ops`` has to agree
      with the ``cosine_distance`` operator the dense store searches with, or the
      index is built and never used.
    * GIN on the generated ``lexical`` column — the keyword half of hybrid search.
    * ``(namespace, rag_document_id)`` — namespace-leading, which serves both the
      per-document purge and the ``document_ids`` search filter. Leading with the
      namespace matters: every query carries it, because the policy adds it even
      when the caller does not.
    """
    return (
        f"CREATE INDEX IF NOT EXISTS ix_{TABLE}_embedding_hnsw "
        f"ON {TABLE} USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX IF NOT EXISTS ix_{TABLE}_lexical ON {TABLE} USING gin (lexical)",
        f"CREATE INDEX IF NOT EXISTS ix_{TABLE}_namespace_document "
        f"ON {TABLE} (namespace, rag_document_id)",
    )


def role_statements(config: PostgresConfig) -> tuple[str, ...]:
    """Create the non-privileged role if absent, and grant it row access.

    The role is created with ``NOLOGIN``: it is a privilege context the owner
    ``SET LOCAL ROLE``s into, never a login identity, so giving it a password
    would only widen the attack surface.

    ``CREATE ROLE`` has no ``IF NOT EXISTS``, hence the ``DO`` block — the one
    piece of PL/pgSQL here, and only because the alternative is failing on the
    second run.
    """
    role = safe_identifier(config.app_role)
    return (
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
                CREATE ROLE {role} NOLOGIN;
            END IF;
        END
        $$
        """,
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON {TABLE} TO {role}",
    )


def rls_statements(config: PostgresConfig) -> tuple[str, ...]:
    """Enable **and force** RLS, then install the policy.

    ``FORCE`` is the half that is easy to miss and expensive to omit: ``ENABLE``
    alone exempts the table's owner, and ragsage connects as the owner. Without
    ``FORCE``, the policy would be decorative on exactly the connection that runs
    every query.

    The policy is dropped and recreated so a changed isolation variable takes
    effect on re-migration rather than leaving a stale predicate in place.
    """
    return (
        f"ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {TABLE} FORCE ROW LEVEL SECURITY",
        f"DROP POLICY IF EXISTS {POLICY} ON {TABLE}",
        policy_sql(config),
    )


def migration_statements(config: PostgresConfig) -> tuple[str, ...]:
    """Every statement :func:`migrate` runs, in order.

    Separated from execution so the whole schema is inspectable — and assertable —
    without a live database.
    """
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        create_table_sql(config),
        *index_statements(),
        *role_statements(config),
        *rls_statements(config),
    )


class SchemaMismatch(RuntimeError):
    """An existing table disagrees with the config it is being migrated against.

    Raised instead of letting :func:`migrate` succeed misleadingly. ``CREATE TABLE IF
    NOT EXISTS`` is a no-op on a table that already exists, so it silently accepts a
    config whose ``embedding_dim`` differs from the column that is really there —
    and the disagreement then surfaces at the first insert, thousands of parsed
    tokens later, as ``expected 1024 dimensions, not 384`` from deep inside the
    driver. Failing at ``migrate()`` puts the error where the mistake is.
    """


async def existing_embedding_dim(connection: AsyncConnection) -> int | None:
    """The width of the live ``embedding`` column, or ``None`` if there is no table.

    Read from ``pg_attribute.atttypmod``, which for a pgvector column *is* the
    declared dimension (unlike ``varchar``, it carries no length offset). ``-1``
    means the column was declared unsized as bare ``vector``, which no ragsage
    migration produces — treated as unknown rather than as a mismatch, so a table
    created by something else is not condemned on this evidence alone.
    """
    typmod = await connection.scalar(
        text(
            "SELECT a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relname = :table AND a.attname = 'embedding' "
            "  AND n.nspname = current_schema() "
            "  AND a.attnum > 0 AND NOT a.attisdropped"
        ),
        {"table": TABLE},
    )
    if typmod is None or int(typmod) < 0:
        return None
    return int(typmod)


async def migrate(connection: AsyncConnection, config: PostgresConfig) -> None:
    """Create the table, indexes, role and policy. Idempotent.

    Takes an owner connection rather than a :class:`~ragsage.storage.engine.Database`
    so a consumer already holding one — running this inside a wider startup
    transaction, say — is not forced to open a second.

    Raises :class:`SchemaMismatch` if the table already exists at a different
    ``embedding_dim``. That check runs *first*, before any DDL: a caller who has
    misconfigured the width should not have indexes rebuilt or a policy recreated
    on the way to being told so.
    """
    actual = await existing_embedding_dim(connection)
    if actual is not None and actual != config.embedding_dim:
        raise SchemaMismatch(
            f"{TABLE}.embedding is vector({actual}) but PostgresConfig.embedding_dim "
            f"is {config.embedding_dim}. pgvector fixes a column's width at creation "
            f"and the HNSW index depends on it, so this is not something migrate() "
            f"can widen in place — the embedding model is pinned per corpus. Either "
            f"set embedding_dim={actual} to match the corpus, or re-index: drop "
            f"{TABLE} and re-ingest with the new model."
        )

    for statement in migration_statements(config):
        await connection.execute(text(statement))
