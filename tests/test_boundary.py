"""The architectural litmus test: ragsage imports no web/auth/tenant code.

The whole design rests on the library depending on nothing app-specific — the
dependency arrow points inward. This statically scans every source file's
imports so a stray ``import fastapi`` or ``from app...`` fails CI, not review.
A pricing/auth/tenancy change must never require editing ragsage; this test is
what keeps that true.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "ragsage"

# Web frameworks, auth/JWT, ORM/DB drivers, the backend package, task queues,
# and network clients — nothing the tenancy-agnostic engine may know about.
_FORBIDDEN = {
    "fastapi",
    "starlette",
    "app",  # the backend's import package
    "sqlalchemy",
    "psycopg",
    "asyncpg",
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


def test_ragsage_imports_nothing_web_auth_or_tenant() -> None:
    offenders: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        bad = _imported_roots(path.read_text()) & _FORBIDDEN
        if bad:
            offenders[str(path.relative_to(_SRC))] = bad

    assert not offenders, f"forbidden imports found: {offenders}"


def test_source_tree_was_actually_scanned() -> None:
    # Guard the guard: if the glob ever matches nothing the test above is
    # vacuously green, so assert we really looked at the modules.
    scanned = {p.name for p in _SRC.rglob("*.py")}
    assert {"ingestion.py", "query.py", "ports.py", "fakes.py"} <= scanned
