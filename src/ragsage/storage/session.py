"""The scoped session — where isolation is actually made real.

This is the consuming backend's ``open_user_session`` moved inward, unchanged in
posture (ADR-0002 §3). Two statements run before any caller SQL:

1. ``SET LOCAL ROLE "<app_role>"`` drops the owner privileges ragsage connects
   with. An owner **bypasses** Row-Level Security, so without this step the
   policy is decorative.
2. ``set_config(<isolation_variable>, <namespace>, is_local => true)`` binds the
   scope to *this transaction*, which is what the policy compares rows against.

Both are transaction-local, so both revert on commit **and** on rollback. That
matters more than it looks: connections are pooled, and a role or a namespace
that outlived its transaction would be inherited by the next borrower — one
tenant's session silently reading under another's scope. The context manager
therefore always ends its transaction, on the happy path and on the exception
path alike.

Row-Level Security stays the load-bearing guarantee here. It is deliberately not
downgraded to an application-side ``WHERE namespace = ...``: a store that forgets
the predicate is a leak, whereas a store running under a policy that forgets
nothing is safe by construction.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragsage.scope import Scope
from ragsage.storage.config import PostgresConfig
from ragsage.storage.identifiers import safe_identifier


@dataclass(frozen=True)
class Statement:
    """One SQL statement plus its bound parameters.

    A plain value so the isolation preamble can be *inspected* — asserted on in a
    unit test, logged, or replayed by ticket-02's migration — without a live
    database and without reaching into a session mock.
    """

    sql: str
    parameters: Mapping[str, str] = field(default_factory=dict)


def isolation_preamble(config: PostgresConfig, scope: Scope) -> tuple[Statement, ...]:
    """Return the statements that pin a transaction to ``scope`` under RLS.

    The role is interpolated because Postgres has no bind form for an identifier;
    it goes through :func:`safe_identifier` here, at the interpolation site, even
    though :class:`PostgresConfig` already validated it. The setting *name*, by
    contrast, is an ordinary text argument to ``set_config`` and is bound like
    any other parameter — nothing of the namespace or the variable name is ever
    spliced into SQL text.
    """
    role = safe_identifier(config.app_role)
    return (
        Statement(f'SET LOCAL ROLE "{role}"'),
        Statement(
            "SELECT set_config(:variable, :value, true)",
            {"variable": config.isolation_variable, "value": scope.namespace},
        ),
    )


@asynccontextmanager
async def open_scoped_session(
    sessionmaker: async_sessionmaker[AsyncSession],
    config: PostgresConfig,
    scope: Scope,
) -> AsyncIterator[AsyncSession]:
    """Open a session pinned to one :class:`~ragsage.scope.Scope`, RLS enforced.

    Commits when the block exits cleanly and rolls back when it raises; either
    way the role and the isolation variable are gone before the connection
    returns to the pool. Every ragsage store goes through here, so the isolation
    posture is defined exactly once.
    """
    async with sessionmaker() as session:
        for statement in isolation_preamble(config, scope):
            await session.execute(text(statement.sql), dict(statement.parameters))
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
