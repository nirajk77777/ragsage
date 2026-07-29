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

# Where the database driver may appear, relative to _SRC: the storage package that
# owns the SQL, and the assembler that wires it up. ``sage.py`` is a *composition
# root* — its entire job is to build the Postgres stores from a config — so keeping
# the driver out of it would mean either lying about its parameter types or moving
# the wiring back onto the consumer, which is the problem ticket 05 removed.
_DRIVER_ALLOWED = frozenset({"storage", "sage.py"})

# The modules whose driver-freedom is the actual invariant. Naming them positively
# says what the rule protects rather than what it happens to exclude: these are the
# engine and its contract, and they must reach persistence only through the ports so
# a consumer can inject a store that is not Postgres.
_ENGINE_CORE = (
    "ingestion.py",
    "query.py",
    "ports.py",
    "models.py",
    "fakes.py",
    "config.py",
    "scope.py",
    "evaluation.py",
    "goldens.py",
    "cli.py",
    "contextualizing.py",
    "caching.py",
)

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


def _may_import_the_driver(path: Path) -> bool:
    relative = path.relative_to(_SRC)
    # Either inside an allowed package (storage/) or an allowed module (sage.py).
    return relative.parts[0] in _DRIVER_ALLOWED


def test_ragsage_imports_nothing_web_auth_or_tenant() -> None:
    offenders: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        bad = _imported_roots(path.read_text()) & _FORBIDDEN_EVERYWHERE
        if bad:
            offenders[str(path.relative_to(_SRC))] = bad

    assert not offenders, f"forbidden imports found: {offenders}"


def test_the_engine_core_never_imports_a_database_driver() -> None:
    """The invariant that matters, stated positively over the modules it protects.

    These are the engine and its contract. If any of them reached for the driver
    directly, the ports would stop being a real seam and a consumer could no longer
    inject a store that is not Postgres.
    """
    offenders: dict[str, set[str]] = {}
    for name in _ENGINE_CORE:
        path = _SRC / name
        assert path.exists(), f"{name} is listed in _ENGINE_CORE but does not exist"
        bad = _imported_roots(path.read_text()) & _DATABASE_PACKAGES
        if bad:
            offenders[name] = bad

    assert not offenders, (
        f"the engine core imported a database driver: {offenders}. Persistence must "
        f"reach these modules only through the ports, so the stores stay swappable."
    )


def test_the_parser_never_imports_a_database_driver() -> None:
    offenders: dict[str, set[str]] = {}
    for path in (_SRC / "parsing").rglob("*.py"):
        bad = _imported_roots(path.read_text()) & _DATABASE_PACKAGES
        if bad:
            offenders[str(path.relative_to(_SRC))] = bad

    assert not offenders, f"database driver imported inside ragsage/parsing/: {offenders}"


def test_the_database_driver_stays_where_it_is_allowed() -> None:
    """The catch-all, so a *new* module cannot quietly reach for the driver.

    ADR-0002 lets ragsage own its storage; it does not let the driver spread. Only
    ``storage/`` (which owns the SQL) and ``sage.py`` (the composition root that
    wires it) may import it — anything else is either a mistake or a decision that
    belongs in this list with a reason.
    """
    offenders: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        if _may_import_the_driver(path):
            continue
        bad = _imported_roots(path.read_text()) & _DATABASE_PACKAGES
        if bad:
            offenders[str(path.relative_to(_SRC))] = bad

    assert not offenders, (
        f"database driver imported outside {sorted(_DRIVER_ALLOWED)}: {offenders}. "
        f"Reach persistence through the ports, or add the module here with a reason."
    )


def test_source_tree_was_actually_scanned() -> None:
    # Guard the guard: if the glob ever matches nothing the tests above are
    # vacuously green, so assert we really looked at the modules.
    scanned = {p.name for p in _SRC.rglob("*.py")}
    assert {"ingestion.py", "query.py", "ports.py", "fakes.py"} <= scanned


def test_every_driver_carve_out_is_actually_used() -> None:
    """Guard the guard, other direction: an exemption that exempts nothing is stale.

    Checked per entry rather than in aggregate, so removing the driver from one
    allowed location does not hide behind another still using it.
    """
    unused: list[str] = []
    for allowed in sorted(_DRIVER_ALLOWED):
        target = _SRC / allowed
        paths = target.rglob("*.py") if target.is_dir() else [target]
        imports: set[str] = set()
        for path in paths:
            imports |= _imported_roots(path.read_text())
        if not imports & _DATABASE_PACKAGES:
            unused.append(allowed)

    assert not unused, (
        f"these entries in _DRIVER_ALLOWED no longer import a database driver: "
        f"{unused}. They are exempting nothing — drop them from the list so the "
        f"catch-all keeps its teeth."
    )
