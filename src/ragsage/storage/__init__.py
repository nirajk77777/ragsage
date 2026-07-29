"""Postgres storage: the pool, the scoped session, and the config that names them.

ragsage owns its storage end to end (ADR-0002). This package holds the part that
is not about rows at all — how a connection is obtained and what has to be true
of a transaction before a store may run inside it.

**This package is not re-exported from ``ragsage``, on purpose.** Importing the
top-level package, or ``ragsage.parsing``, must not drag in SQLAlchemy or the
Postgres driver: the parser and the pipelines run offline over the in-memory
fakes, and a consumer that only parses should not need a database installed to
import the library. Reach for storage explicitly::

    from ragsage.storage import Database, PostgresConfig

    db = Database(PostgresConfig(dsn="postgresql://user:pw@host/db", app_role="rag_app"))
    async with db.session(scope) as session:
        ...  # runs as the non-privileged role, pinned to scope.namespace
"""

from ragsage.storage.config import PostgresConfig
from ragsage.storage.engine import Database, create_engine, create_sessionmaker
from ragsage.storage.identifiers import safe_identifier, safe_setting_name
from ragsage.storage.schema import TABLE, migrate, migration_statements
from ragsage.storage.session import Statement, isolation_preamble, open_scoped_session

__all__ = [
    "TABLE",
    "Database",
    "PostgresConfig",
    "Statement",
    "create_engine",
    "create_sessionmaker",
    "isolation_preamble",
    "migrate",
    "migration_statements",
    "open_scoped_session",
    "safe_identifier",
    "safe_setting_name",
]
