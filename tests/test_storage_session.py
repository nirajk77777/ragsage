"""The isolation preamble, and that it cannot outlive its transaction.

These tests need no database. The preamble is exposed as plain
:class:`~ragsage.storage.session.Statement` values precisely so the *posture* can
be asserted directly — which role, which setting, what is bound versus
interpolated — instead of being inferred from a live server's behaviour. The
integration tests that do talk to Postgres prove the policy works; these prove we
are sending what we think we are sending.

The reset guarantee is the one worth being paranoid about. A role or a namespace
that survived its transaction would be inherited by the next borrower of a pooled
connection, which is a cross-namespace read with no failing test to announce it.
Nothing here can prove Postgres honours ``SET LOCAL``; what it proves is that
ragsage always ends the transaction, on both the happy and the exception path,
which is the half we own.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragsage.scope import Scope
from ragsage.storage.config import PostgresConfig
from ragsage.storage.session import isolation_preamble, open_scoped_session


class _RecordingSession:
    """The smallest thing shaped like an ``AsyncSession`` for this purpose.

    Records the SQL it is handed, in order, alongside the transaction verbs — so a
    test can assert not just *what* ran but that the preamble ran *first*.
    """

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.parameters: list[dict[str, Any]] = []

    async def execute(self, clause: Any, parameters: Any = None) -> None:
        self._log.append(str(clause))
        self.parameters.append(dict(parameters or {}))

    async def commit(self) -> None:
        self._log.append("COMMIT")

    async def rollback(self) -> None:
        self._log.append("ROLLBACK")

    async def __aenter__(self) -> _RecordingSession:
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False


def _sessionmaker(log: list[str]) -> Any:
    return lambda: _RecordingSession(log)


def _config(**overrides: Any) -> PostgresConfig:
    return PostgresConfig(dsn="postgresql://user:pw@localhost/ragsage", **overrides)


# --------------------------------------------------------------------------- #
# What the preamble says
# --------------------------------------------------------------------------- #


def test_the_preamble_drops_owner_privileges_first() -> None:
    role_statement, _ = isolation_preamble(_config(app_role="rag_app"), Scope(namespace="acme"))

    # Quoted, because an unquoted role is folded to lower case by Postgres and a
    # mixed-case role would then silently not exist.
    assert role_statement.sql == 'SET LOCAL ROLE "rag_app"'
    assert role_statement.parameters == {}


def test_the_namespace_is_bound_not_interpolated() -> None:
    _, setting_statement = isolation_preamble(
        _config(isolation_variable="ragsage.namespace"), Scope(namespace="acme")
    )

    assert setting_statement.parameters == {
        "variable": "ragsage.namespace",
        "value": "acme",
    }
    # The namespace is caller data. It must appear nowhere in the SQL text.
    assert "acme" not in setting_statement.sql
    assert "ragsage.namespace" not in setting_statement.sql


def test_the_setting_is_transaction_local() -> None:
    _, setting_statement = isolation_preamble(_config(), Scope(namespace="acme"))

    # `is_local => true` is the whole reset guarantee. A missing third argument
    # would make the binding session-wide and leak it through the pool.
    assert "set_config(:variable, :value, true)" in setting_statement.sql


def test_a_role_smuggled_past_the_config_still_cannot_reach_sql() -> None:
    """Defence in depth: the guard sits at the interpolation site, not only in config.

    ``PostgresConfig`` already validates ``app_role``, so this forces a bad value
    past that check the only way a frozen dataclass allows. The point is that the
    interpolation site does not *trust* its config.
    """
    config = _config()
    object.__setattr__(config, "app_role", 'rag_app"; DROP TABLE ragsage_chunks; --')

    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        isolation_preamble(config, Scope(namespace="acme"))


# --------------------------------------------------------------------------- #
# That it always ends its transaction
# --------------------------------------------------------------------------- #


async def test_the_preamble_runs_before_any_caller_sql_and_commits() -> None:
    log: list[str] = []

    async with open_scoped_session(_sessionmaker(log), _config(), Scope(namespace="acme")) as s:
        await s.execute("SELECT 1")

    assert log == [
        'SET LOCAL ROLE "ragsage_app"',
        "SELECT set_config(:variable, :value, true)",
        "SELECT 1",
        "COMMIT",
    ]


async def test_a_failing_block_rolls_back_and_re_raises() -> None:
    log: list[str] = []

    with pytest.raises(RuntimeError, match="boom"):
        async with open_scoped_session(_sessionmaker(log), _config(), Scope(namespace="acme")):
            raise RuntimeError("boom")

    assert log[-1] == "ROLLBACK", "a failed scoped session must not commit"
    assert "COMMIT" not in log


async def test_each_scope_gets_its_own_binding() -> None:
    """Two sequential scopes on one factory must each bind their own namespace.

    This is the pooling hazard in miniature: the second session must not inherit
    the first one's namespace, so the preamble has to re-run per transaction
    rather than once per connection.
    """
    log: list[str] = []
    opened: list[_RecordingSession] = []

    def sessionmaker() -> _RecordingSession:
        session = _RecordingSession(log)
        opened.append(session)
        return session

    for namespace in ("acme", "globex"):
        async with open_scoped_session(sessionmaker, _config(), Scope(namespace=namespace)):
            pass

    bound = [
        parameters["value"]
        for session in opened
        for parameters in session.parameters
        if "value" in parameters
    ]
    assert bound == ["acme", "globex"]
    assert log.count("COMMIT") == 2
