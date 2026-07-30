"""How the caller tells ragsage where its database is — and nothing more.

One frozen dataclass, constructed by the caller and validated on construction.
It reads **no environment variables**, here or anywhere: ADR-0002 rules that out
explicitly, and the research report singled out the class-definition-time
``get_env_value`` default as the specific anti-pattern to avoid. A config object
whose values depend on the ambient process is untestable, un-injectable, and
silently different in production; a config object whose values came from its
caller is none of those. Wiring env to fields is the *consumer's* job, at the
consumer's composition root.

Everything isolation-related is a field rather than a literal, because ragsage
runs inside somebody else's database. The app role, and the setting name Row-Level
Security reads, both belong to whoever is deploying — the defaults here are only
ragsage-owned names that will not collide with the consumer's own.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragsage.storage.identifiers import safe_identifier, safe_setting_name

# ragsage connects with an async driver only — the pool, the session, and every
# store are async. A DSN naming a sync driver is a mistake worth catching at
# construction rather than at the first connect.
_ASYNC_DRIVER = "postgresql+asyncpg"

# The driverless forms every hosting provider prints. Accepted and normalised
# rather than rejected: the caller pasting the DSN from their provider's console
# is the common case, and refusing it teaches nothing.
_BARE_SCHEMES = frozenset({"postgresql", "postgres"})


def _normalise_dsn(dsn: str) -> str:
    """Return ``dsn`` as a ``postgresql+asyncpg://`` URL, or raise ``ValueError``."""
    scheme, separator, remainder = dsn.partition("://")
    if not separator or not remainder:
        raise ValueError(f"PostgresConfig.dsn must be a URL like 'postgresql://...': {dsn!r}")
    if scheme in _BARE_SCHEMES:
        return f"{_ASYNC_DRIVER}://{remainder}"
    if scheme == _ASYNC_DRIVER:
        return dsn
    raise ValueError(
        f"PostgresConfig.dsn must use the {_ASYNC_DRIVER} driver (or a bare "
        f"postgresql:// URL, which is normalised to it); got {scheme!r}"
    )


@dataclass(frozen=True)
class PostgresConfig:
    """Everything ragsage needs to own a connection pool against one database.

    Parameters
    ----------
    dsn:
        The connection URL ragsage connects with. It connects as the *owner* of
        its own table — owners bypass Row-Level Security, which is why every
        scoped operation drops into ``app_role`` first (see
        :func:`~ragsage.storage.session.open_scoped_session`).
    app_role:
        The non-privileged Postgres role scoped work runs under. Not a login
        identity: it is a privilege context the owner switches into with
        ``SET LOCAL ROLE`` so RLS is actually enforced. Reaches interpolated SQL,
        so it is guarded as a plain identifier.
    isolation_variable:
        The run-time setting carrying ``Scope.namespace`` for the duration of a
        transaction, and the value ragsage's RLS policy compares rows against.
        Configurable because two isolation variables coexist in the deployment
        this was extracted from — the consumer's ``app.user_id`` over its own
        tables, and this one over ragsage's — and they must not be confused. The
        default is ragsage-owned for exactly that reason.
    embedding_dim:
        The width of the ``embedding`` column. Fixed at table-creation time and
        **pinned per corpus** — changing it invalidates every stored vector, so it
        belongs in config rather than as a literal in the DDL. Must match whatever
        the configured embedder actually returns.
    text_search_config:
        The Postgres text-search configuration the generated ``lexical`` column is
        built with (``'english'`` was hardcoded in the consuming backend). It has
        to be named explicitly because the two-argument form of ``to_tsvector`` is
        the IMMUTABLE one, and a ``STORED`` generated column accepts nothing else —
        the one-argument form depends on a session setting and is only STABLE.
        Changing it requires rebuilding the column.
    pool_size, max_overflow:
        Pool sizing. Deliberately explicit and deliberately modest: ragsage's
        pool is a *second* pool reaching the same Postgres as the consumer's, and
        the pair have to fit inside ``max_connections`` together.
    pool_pre_ping:
        Check a pooled connection is alive before handing it out. On by default —
        connections idle across a database restart or a proxy timeout otherwise
        fail the first query of an unlucky request.
    """

    dsn: str
    app_role: str = "ragsage_app"
    isolation_variable: str = "ragsage.namespace"
    embedding_dim: int = 1024
    text_search_config: str = "english"
    pool_size: int = 5
    max_overflow: int = 5
    pool_pre_ping: bool = True

    def __post_init__(self) -> None:
        dsn = self.dsn.strip()
        if not dsn:
            raise ValueError("PostgresConfig.dsn must be a non-empty connection URL")
        # Frozen means immutable *to callers*; normalising during construction is
        # the one moment the value is still being decided (Scope does the same to
        # freeze its filters).
        object.__setattr__(self, "dsn", _normalise_dsn(dsn))

        # Fail here, where the mistake was made, rather than deep inside a query.
        safe_identifier(self.app_role)
        safe_setting_name(self.isolation_variable)
        # Both reach the DDL as interpolated text — pgvector's type modifier and
        # to_tsvector's first argument have no bind form — so both are guarded.
        safe_identifier(self.text_search_config)

        if self.embedding_dim < 1:
            raise ValueError("PostgresConfig.embedding_dim must be a positive number of dimensions")
        # pgvector's hard ceiling for an indexable column. Failing here beats a
        # CREATE INDEX that succeeds on the table and then refuses the index.
        if self.embedding_dim > 2000:
            raise ValueError(
                "PostgresConfig.embedding_dim must be at most 2000: pgvector cannot build an "
                f"HNSW index on a wider column, and got {self.embedding_dim}"
            )
        if self.pool_size < 1:
            raise ValueError("PostgresConfig.pool_size must be at least 1")
        if self.max_overflow < 0:
            raise ValueError("PostgresConfig.max_overflow must not be negative")
