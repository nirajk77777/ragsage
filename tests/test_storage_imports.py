"""Storage stays opt-in: importing ragsage must not require a database.

ADR-0002 makes SQLAlchemy and asyncpg *core* dependencies — ``pip install
ragsage`` yields a working engine rather than a set of interfaces. Installed is
not the same as imported, though, and the distinction is the point of this guard:

- ``import ragsage`` is what a consumer using only the fakes, the goldens, or the
  evaluation harness pays for. It must cost nothing database-shaped.
- ``import ragsage.parsing`` carries a separate, stronger promise (see
  :mod:`tests.test_dependency_guard`) that it stays importable on a constrained
  box. Dragging a driver in through a stray re-export would erode that.

So the rule is narrow and mechanical: the driver stack is reachable from
``ragsage.storage`` and from nowhere else. That keeps ``PostgresConfig`` a thing
you construct deliberately, and keeps "no database configured" a working state
rather than an import error.

Checked in a *clean subprocess* for the same reason the dependency guard is: a
full pytest run has already imported SQLAlchemy by the time this file executes,
so an in-process ``sys.modules`` check would prove nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys

# The driver stack storage owns. Reachable from ragsage.storage, nowhere else.
_DATABASE_PACKAGES = ("sqlalchemy", "asyncpg")

_PROBE = """
import importlib
import json
import sys

importlib.import_module({module!r})

# Top-level package names only: sqlalchemy.orm et al. all trace back to one root.
print(json.dumps(sorted({{name.split(".")[0] for name in sys.modules}})))
"""


def _roots_imported_by(module: str) -> set[str]:
    """Return the top-level packages a clean ``import <module>`` pulls in."""
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        check=True,
    )
    return {str(name) for name in json.loads(completed.stdout)}


def test_importing_ragsage_needs_no_database_driver() -> None:
    roots = _roots_imported_by("ragsage")

    leaked = sorted(name for name in _DATABASE_PACKAGES if name in roots)
    assert not leaked, (
        f"`import ragsage` pulled in {leaked}. Storage must stay opt-in behind "
        f"`ragsage.storage` so the fakes, the goldens and the CLI keep working "
        f"with no database configured."
    )


def test_importing_the_parser_needs_no_database_driver() -> None:
    roots = _roots_imported_by("ragsage.parsing")

    leaked = sorted(name for name in _DATABASE_PACKAGES if name in roots)
    assert not leaked, (
        f"`import ragsage.parsing` pulled in {leaked}. The parser carries a "
        f"deliberate portability promise (tests/test_dependency_guard.py); a "
        f"database driver has no business in its import graph."
    )


def test_the_storage_package_is_where_the_driver_lives() -> None:
    """The complement of the two guards above — proof they are narrow, not vacuous.

    Without this, deleting the SQLAlchemy import from ``storage`` entirely would
    leave the guards passing while the package no longer worked.
    """
    roots = _roots_imported_by("ragsage.storage")

    assert "sqlalchemy" in roots, (
        "ragsage.storage did not import SQLAlchemy, so the two guards above are "
        "passing for the wrong reason."
    )
