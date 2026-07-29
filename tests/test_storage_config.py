"""Tests for the storage config and the identifier guards.

Both are validation-only and run entirely offline: the point of a frozen config
validated in ``__post_init__`` is that a bad value is impossible to hold, so
these tests are the specification of "bad".
"""

from dataclasses import FrozenInstanceError

import pytest

from ragsage.storage import PostgresConfig, safe_identifier, safe_setting_name


def test_defaults_are_ragsage_owned() -> None:
    config = PostgresConfig(dsn="postgresql+asyncpg://u:p@localhost:5432/db")

    assert config.app_role == "ragsage_app"
    # The consumer's own tables key on `app.user_id`; ragsage must not squat on it.
    assert config.isolation_variable != "app.user_id"
    assert config.isolation_variable.startswith("ragsage.")


def test_config_is_frozen() -> None:
    config = PostgresConfig(dsn="postgresql://u:p@localhost/db")
    with pytest.raises(FrozenInstanceError):
        config.app_role = "other"  # type: ignore[misc]


def test_bare_postgres_url_is_normalised_to_the_async_driver() -> None:
    assert (
        PostgresConfig(dsn="postgresql://u:p@localhost:5432/db").dsn
        == "postgresql+asyncpg://u:p@localhost:5432/db"
    )
    assert (
        PostgresConfig(dsn="  postgres://u:p@localhost:5432/db  ").dsn
        == "postgresql+asyncpg://u:p@localhost:5432/db"
    )


def test_async_driver_url_is_left_alone() -> None:
    dsn = "postgresql+asyncpg://u:p@localhost:5432/db"
    assert PostgresConfig(dsn=dsn).dsn == dsn


def test_sync_driver_is_rejected() -> None:
    with pytest.raises(ValueError, match="asyncpg"):
        PostgresConfig(dsn="postgresql+psycopg2://u:p@localhost/db")


@pytest.mark.parametrize("dsn", ["", "   ", "not-a-url", "postgresql://"])
def test_malformed_dsn_is_rejected(dsn: str) -> None:
    with pytest.raises(ValueError, match="dsn"):
        PostgresConfig(dsn=dsn)


@pytest.mark.parametrize("role", ["rag app", "rag-app", 'app"; DROP TABLE x --', "1app", ""])
def test_unsafe_app_role_is_rejected(role: str) -> None:
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        PostgresConfig(dsn="postgresql://u:p@localhost/db", app_role=role)


@pytest.mark.parametrize("variable", ["namespace", "ragsage.name.space", "ragsage.", "rag sage.ns"])
def test_unsafe_isolation_variable_is_rejected(variable: str) -> None:
    with pytest.raises(ValueError, match="unsafe Postgres setting name"):
        PostgresConfig(dsn="postgresql://u:p@localhost/db", isolation_variable=variable)


def test_pool_sizing_is_validated() -> None:
    with pytest.raises(ValueError, match="pool_size"):
        PostgresConfig(dsn="postgresql://u:p@localhost/db", pool_size=0)
    with pytest.raises(ValueError, match="max_overflow"):
        PostgresConfig(dsn="postgresql://u:p@localhost/db", max_overflow=-1)


def test_config_reads_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config's values come from its caller, never from the ambient process."""
    for name in ("DATABASE_URL", "POSTGRES_DSN", "RAGSAGE_DSN", "RAGSAGE_APP_ROLE"):
        monkeypatch.setenv(name, "postgresql://hijacked:hijacked@evil/db")

    config = PostgresConfig(dsn="postgresql://u:p@localhost/db")

    assert config.dsn == "postgresql+asyncpg://u:p@localhost/db"
    assert config.app_role == "ragsage_app"


@pytest.mark.parametrize("name", ["rag_app", "_role", "Role1"])
def test_safe_identifier_accepts_plain_identifiers(name: str) -> None:
    assert safe_identifier(name) == name


@pytest.mark.parametrize("name", ["ragsage.namespace", "app.user_id", "_x.y1"])
def test_safe_setting_name_accepts_qualified_names(name: str) -> None:
    assert safe_setting_name(name) == name


def test_safe_setting_name_requires_a_prefix() -> None:
    with pytest.raises(ValueError, match="qualified with a prefix"):
        safe_setting_name("namespace")
