"""The connection pool ragsage owns, and the handle its stores borrow from.

ADR-0002 §3: given one DSN, ragsage connects as the owner of its own table and
manages its own pool, rather than borrowing a session from the consumer. That is
what lets ``pip install ragsage`` produce a working engine instead of a set of
interfaces the consumer must wire.

Owning the pool is not the same as owning the privileges. The pool connects as
the table owner — necessary for DDL, and unavoidable for a library that creates
its own schema — but an owner bypasses Row-Level Security, so *no scoped work
runs on an owner connection*. :meth:`Database.session` is the only door the
stores use, and it drops into the non-privileged app role first (see
:mod:`ragsage.storage.session`). :meth:`Database.owner_connection` is the
deliberate exception, reserved for DDL, and named so that using it by accident
reads wrong.

Nothing here connects at construction: engines are lazy, so building a
:class:`Database` in a composition root — or in a test — costs nothing and
requires no reachable server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ragsage.scope import Scope
from ragsage.storage.config import PostgresConfig
from ragsage.storage.schema import migrate
from ragsage.storage.session import open_scoped_session


def create_engine(config: PostgresConfig) -> AsyncEngine:
    """Create the pooled async engine described by ``config``."""
    return create_async_engine(
        config.dsn,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_pre_ping=config.pool_pre_ping,
        future=True,
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the session factory bound to ``engine``.

    ``expire_on_commit=False`` so rows read inside a scoped session stay usable
    after it commits — the scoped session commits on the way out of its context
    manager, and its caller is still holding what it returned.
    """
    return async_sessionmaker(engine, expire_on_commit=False)


class Database:
    """ragsage's pool: one engine, one session factory, one way in.

    Construct it from a :class:`PostgresConfig` and hand it to the stores. An
    ``engine`` may be injected instead of created — for tests, or for a consumer
    that genuinely wants both libraries sharing one pool against
    ``max_connections``.
    """

    def __init__(self, config: PostgresConfig, engine: AsyncEngine | None = None) -> None:
        self._config = config
        self._engine = create_engine(config) if engine is None else engine
        self._sessionmaker = create_sessionmaker(self._engine)

    @property
    def config(self) -> PostgresConfig:
        return self._config

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    def session(self, scope: Scope) -> AbstractAsyncContextManager[AsyncSession]:
        """Open a transaction pinned to ``scope``, with RLS actively enforced.

        The one door scoped work goes through. See
        :func:`~ragsage.storage.session.open_scoped_session` for what that costs
        and why both settings are transaction-local.
        """
        return open_scoped_session(self._sessionmaker, self._config, scope)

    @asynccontextmanager
    async def owner_connection(self) -> AsyncIterator[AsyncConnection]:
        """Open an owner-privileged transaction — **DDL only**, RLS not enforced.

        No ``SET LOCAL ROLE``, no isolation variable: this connection sees every
        namespace. It exists so ragsage can create its own table, indexes, role
        and policy (ticket 02's ``migrate``). Reading or writing rows through it
        would bypass the guarantee the rest of this package exists to provide.
        """
        async with self._engine.begin() as connection:
            yield connection

    async def migrate(self) -> None:
        """Create ragsage's table, indexes, role and RLS policy. Idempotent.

        Safe to call on every start. Runs on an owner connection because only the
        owner may issue this DDL — see :mod:`ragsage.storage.schema`.
        """
        async with self.owner_connection() as connection:
            await migrate(connection, self._config)

    async def ping(self) -> bool:
        """Return True if a trivial round-trip to the database succeeds.

        ``SELECT 1`` on a real connection proves the socket, the credentials and
        the database are all live — not merely that a DSN was configured. Any
        failure becomes ``False`` so a caller's health check can report degraded
        rather than raise.
        """
        try:
            async with self._engine.connect() as connection:
                result = await connection.execute(text("SELECT 1"))
                return bool(result.scalar_one() == 1)
        except Exception:
            return False

    async def dispose(self) -> None:
        """Close every pooled connection. Call on shutdown."""
        await self._engine.dispose()
