"""CPU-compatibility guard: ``ragsage.parsing`` must stay free of AVX2 libraries.

The deployment box is x86-64-v1. A handful of otherwise-ordinary scientific
libraries either refuse to import there or detonate on the first AVX2 instruction
(the classic *illegal instruction* crash): ``numpy>=2``, ``torch``,
``onnxruntime``, ``transformers``, and ``magika`` (which pulls ``onnxruntime``).
The whole point of the heuristic parser swap was to keep that stack out of the
import graph, and because the parser now lives in the open-source ``ragsage``
library the invariant protects *every* downstream consumer, not just this repo.

This guard encodes that invariant as CI. It passes today, when only the text path
exists, and tightens automatically as the PDF/DOCX/PPTX/HTML format modules land:
each concrete submodule is imported best-effort, so the moment a format path drags
in a forbidden transitive dependency the guard goes red.

The import graph is inspected in a *clean subprocess*. That isolation is
load-bearing: other modules in the full pytest run (e.g. PDF-path tests) may pull
``numpy`` into this process's ``sys.modules``, which would make an in-process
check meaningless. A fresh interpreter shows exactly what ``import
ragsage.parsing`` alone drags in.

``numpy`` is a special case: the 1.x line is AVX-safe and permitted, only ``>=2``
is the trap — so its mere presence is not a failure; the guard parses
``numpy.__version__`` and fails only on a major version of 2 or higher.
"""

from __future__ import annotations

import json
import subprocess
import sys

# Forbidden outright: presence anywhere in the import graph is a failure.
_FORBIDDEN = ("torch", "onnxruntime", "transformers", "magika")

# The probe runs in a fresh interpreter and reports the parsing package's import
# graph as JSON on stdout: the sorted module names plus numpy's version (or null).
# Format submodules that don't exist yet on the skeleton are suppressed, so this
# stays green now and tightens as each format path lands in the merged tree.
_PROBE = """
import contextlib
import importlib
import json
import sys

# The package the real composition root imports (re-exports backend/classifier/router).
import ragsage.parsing

# Concrete submodules that exist today.
for _name in (
    "ragsage.parsing.router",
    "ragsage.parsing.chunking",
    "ragsage.parsing.classifier",
    "ragsage.parsing.backend",
    "ragsage.parsing.identity",
):
    importlib.import_module(_name)

# Format paths land later; import each best-effort so the guard covers them the
# moment they exist without failing on the skeleton where they don't.
for _name in (
    "ragsage.parsing.pdf",
    "ragsage.parsing.docx",
    "ragsage.parsing.pptx",
    "ragsage.parsing.html",
):
    with contextlib.suppress(ImportError, ModuleNotFoundError):
        importlib.import_module(_name)

_numpy_version = sys.modules["numpy"].__version__ if "numpy" in sys.modules else None
print(json.dumps({"modules": sorted(sys.modules), "numpy_version": _numpy_version}))
"""


def _probe_import_graph() -> tuple[set[str], str | None]:
    """Import the parsing package in a clean interpreter; return its module graph.

    Returns the set of imported module names and numpy's version string (or
    ``None`` when numpy was never imported).
    """
    # Same interpreter as the test run (the project venv under ``uv run``), so
    # ragsage and all its real dependencies resolve exactly as they do in CI.
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    graph = json.loads(completed.stdout)
    modules = {str(name) for name in graph["modules"]}
    numpy_version = graph["numpy_version"]
    return modules, None if numpy_version is None else str(numpy_version)


def test_parsing_import_graph_is_cpu_safe() -> None:
    modules, numpy_version = _probe_import_graph()

    leaked = sorted(name for name in _FORBIDDEN if name in modules)
    assert not leaked, (
        f"forbidden AVX2-incompatible libraries leaked into the ragsage.parsing "
        f"import graph: {leaked}. These crash on the x86-64-v1 deployment box; a "
        f"format path must not pull them in (magika drags onnxruntime, "
        f"transformers drags torch)."
    )

    if numpy_version is not None:
        major = int(numpy_version.split(".", 1)[0])
        assert major < 2, (
            f"numpy {numpy_version} leaked into the ragsage.parsing import graph. "
            f"numpy 1.x is AVX-safe and permitted, but numpy>=2 hits the AVX2 wall "
            f"on the x86-64-v1 deployment box."
        )
