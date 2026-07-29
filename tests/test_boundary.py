"""The architectural litmus test: ragsage imports no web/auth/tenant code.

The whole design rests on the library depending on nothing app-specific — the
dependency arrow points inward. This statically scans every source file's
imports so a stray ``import fastapi`` or ``from app...`` fails CI, not review.
A pricing/auth/tenancy change must never require editing ragsage; this test is
what keeps that true.

**Amended by ADR-0002.** The database driver used to be on the forbidden list
outright, because storage was the consumer's business and ragsage only described
it through ports. ADR-0002 reverses that: ragsage owns ``ragsage_chunks``, its
pool and its RLS posture, so SQLAlchemy and asyncpg are now core dependencies.

That reversal is deliberately *scoped* rather than global, and this file is where
the scope is enforced. The driver is legal inside ``ragsage/storage/`` and illegal
everywhere else — so the engine (``ingestion``, ``query``, ``ports``, ``fakes``)
and the parser stay driver-free, and the ports remain the seam a consumer can
inject a non-Postgres store into. A stray ``import sqlalchemy`` in ``query.py``
would quietly turn a swappable store into a hard dependency; that is the
regression this catches.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "ragsage"

# The storage package, relative to _SRC. The one place the driver may appear.
_STORAGE = "storage"

# Web frameworks, auth/JWT, the backend package, task queues, and network
# clients — nothing the tenancy-agnostic engine may know about, anywhere.
#
# ``alembic`` stays here on purpose: ragsage's schema is created by its own
# idempotent ``migrate()`` (ADR-0002), not by a migration framework the consumer
# would then have to run. ``redis`` likewise — the Redis-backed ``Cache`` adapter
# is the consumer's, injected through the port.
_FORBIDDEN_EVERYWHERE = {
    "fastapi",
    "starlette",
    "app",  # the backend's import package
    "alembic",
    "jose",
    "jwt",
    "passlib",
    "taskiq",
    "redis",
    "httpx",
    "requests",
    "aiohttp",
    "urllib",
    "socket",
    "boto3",
}

# Permitted inside ragsage/storage/ (ADR-0002), forbidden outside it.
_DATABASE_PACKAGES = {
    "sqlalchemy",
    "psycopg",
    "asyncpg",
}


def _imported_roots(source: str) -> set[str]:
    """Top-level module names imported by a source file."""
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _is_storage(path: Path) -> bool:
    return path.relative_to(_SRC).parts[0] == _STORAGE


def test_ragsage_imports_nothing_web_auth_or_tenant() -> None:
    offenders: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        bad = _imported_roots(path.read_text()) & _FORBIDDEN_EVERYWHERE
        if bad:
            offenders[str(path.relative_to(_SRC))] = bad

    assert not offenders, f"forbidden imports found: {offenders}"


def test_the_database_driver_stays_inside_the_storage_package() -> None:
    """ADR-0002 lets ragsage own its storage — it does not let the driver spread.

    Everything outside ``ragsage/storage/`` must still reach persistence only
    through the ports, so a consumer can inject a store that is not Postgres.
    """
    offenders: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        if _is_storage(path):
            continue
        bad = _imported_roots(path.read_text()) & _DATABASE_PACKAGES
        if bad:
            offenders[str(path.relative_to(_SRC))] = bad

    assert not offenders, (
        f"database driver imported outside ragsage/storage/: {offenders}. "
        f"ADR-0002 scopes the driver to the storage package; the engine and the "
        f"parser reach persistence through the ports so the stores stay swappable."
    )


def test_source_tree_was_actually_scanned() -> None:
    # Guard the guard: if the glob ever matches nothing the tests above are
    # vacuously green, so assert we really looked at the modules.
    scanned = {p.name for p in _SRC.rglob("*.py")}
    assert {"ingestion.py", "query.py", "ports.py", "fakes.py"} <= scanned


def test_the_storage_carve_out_is_not_vacuous() -> None:
    # Guard the guard, other direction: the carve-out above only means something
    # if the storage package genuinely does import the driver it exempts.
    storage_imports: set[str] = set()
    for path in _SRC.rglob("*.py"):
        if _is_storage(path):
            storage_imports |= _imported_roots(path.read_text())

    assert storage_imports & _DATABASE_PACKAGES, (
        "ragsage/storage/ imports no database driver, so the carve-out in "
        "test_the_database_driver_stays_inside_the_storage_package is exempting "
        "nothing and would not catch a regression."
    )
