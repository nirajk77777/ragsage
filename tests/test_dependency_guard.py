"""Import-graph guard: the parser — and a bare ``import ragsage`` — stay light.

A handful of otherwise-ordinary scientific libraries are heavy, platform-fussy,
or both: ``numpy>=2``, ``torch``, ``onnxruntime``, ``transformers``, and
``magika`` (which pulls ``onnxruntime``). The whole point of the heuristic parser
swap was to keep that stack out of the import graph, and because the parser lives
in the open-source ``ragsage`` library the invariant protects *every* downstream
consumer, not just this repo.

The original driver was hardware — the deployment box advertised only x86-64-v1,
where numpy 2.x refuses to import and AVX2 code detonates with an *illegal
instruction*. That premise is retired (ADR-0001, amended 2026-07-29: the box now
reports x86-64-v4). The invariant survives as a deliberate **portability
promise**: a consumer who only wants to parse documents should not inherit a
multi-hundred-megabyte numerical stack.

That promise now has to coexist with ADR-0002, which brings the Voyage and OpenAI
adapters into core — and ``voyageai`` pulls numpy 2.5.1 and HuggingFace
``tokenizers``. The two live together because the heavy imports are confined to
:mod:`ragsage.providers`, which nothing else in the package imports. So this file
guards three things:

* ``ragsage.parsing`` pulls none of the forbidden libraries (the original
  invariant, unchanged);
* a bare ``import ragsage`` pulls no provider SDK either — the public façades,
  models, and ports stay importable without paying for a provider you may never
  call;
* and, as a canary, ``ragsage.providers`` genuinely *does* pull that stack. If
  that ever stops being true the first two assertions become vacuous, and a
  green guard that proves nothing is worse than a red one.

Every graph is inspected in a *clean subprocess*. That isolation is load-bearing:
other modules in the full pytest run (e.g. the provider tests) pull ``numpy``
into this process's ``sys.modules``, which would make an in-process check
meaningless. A fresh interpreter shows exactly what one import drags in.

``numpy`` is a special case for the parser: the 1.x line is light and permitted,
only ``>=2`` is the trap — so its mere presence is not a failure; the guard parses
``numpy.__version__`` and fails only on a major version of 2 or higher.
"""

from __future__ import annotations

import json
import subprocess
import sys

# Forbidden outright: presence anywhere in the parser's import graph is a failure.
_FORBIDDEN = ("torch", "onnxruntime", "transformers", "magika")

# The provider SDKs and their LangChain integrations. Core dependencies per
# ADR-0002, but confined to ``ragsage.providers``.
_PROVIDER_SDKS = ("voyageai", "openai", "langchain_openai", "langchain_voyageai")

# The probe runs in a fresh interpreter and reports an import graph as JSON on
# stdout: the sorted module names plus numpy's version (or null). Format
# submodules that don't exist yet are suppressed, so the guard tightens as each
# format path lands rather than failing on a tree where one is absent.
_PROBE = """
import contextlib
import importlib
import json
import sys

for _name in {required!r}:
    importlib.import_module(_name)

for _name in {optional!r}:
    with contextlib.suppress(ImportError, ModuleNotFoundError):
        importlib.import_module(_name)

_numpy_version = sys.modules["numpy"].__version__ if "numpy" in sys.modules else None
print(json.dumps({{"modules": sorted(sys.modules), "numpy_version": _numpy_version}}))
"""

# The package the real composition root imports, plus every concrete submodule
# that exists today; the format paths are best-effort.
_PARSING_REQUIRED = (
    "ragsage.parsing",
    "ragsage.parsing.router",
    "ragsage.parsing.chunking",
    "ragsage.parsing.classifier",
    "ragsage.parsing.backend",
    "ragsage.parsing.identity",
)
_PARSING_OPTIONAL = (
    "ragsage.parsing.pdf",
    "ragsage.parsing.docx",
    "ragsage.parsing.pptx",
    "ragsage.parsing.html",
)


def _probe_import_graph(
    required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> tuple[set[str], str | None]:
    """Import modules in a clean interpreter; return the resulting module graph.

    Returns the set of imported module names and numpy's version string (or
    ``None`` when numpy was never imported).
    """
    # Same interpreter as the test run (the project venv under ``uv run``), so
    # ragsage and all its real dependencies resolve exactly as they do in CI.
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE.format(required=required, optional=optional)],
        capture_output=True,
        text=True,
        check=True,
    )
    graph = json.loads(completed.stdout)
    modules = {str(name) for name in graph["modules"]}
    numpy_version = graph["numpy_version"]
    return modules, None if numpy_version is None else str(numpy_version)


def _assert_numpy_is_light(numpy_version: str | None, imported: str) -> None:
    if numpy_version is None:
        return
    major = int(numpy_version.split(".", 1)[0])
    assert major < 2, (
        f"numpy {numpy_version} leaked into the {imported} import graph. numpy 1.x "
        f"is light and permitted, but numpy>=2 drags in the heavy numerical stack "
        f"this package promises to stay clear of."
    )


def test_parsing_import_graph_is_cpu_safe() -> None:
    modules, numpy_version = _probe_import_graph(_PARSING_REQUIRED, _PARSING_OPTIONAL)

    leaked = sorted(name for name in _FORBIDDEN if name in modules)
    assert not leaked, (
        f"forbidden heavy libraries leaked into the ragsage.parsing import graph: "
        f"{leaked}. A format path must not pull them in (magika drags onnxruntime, "
        f"transformers drags torch)."
    )

    leaked_sdks = sorted(name for name in _PROVIDER_SDKS if name in modules)
    assert not leaked_sdks, (
        f"provider SDKs leaked into the ragsage.parsing import graph: {leaked_sdks}. "
        f"They are core dependencies (ADR-0002) but must stay confined to "
        f"ragsage.providers — voyageai alone drags numpy 2.x and HuggingFace "
        f"tokenizers into a graph that only wanted to read a PDF."
    )

    _assert_numpy_is_light(numpy_version, "ragsage.parsing")


def test_bare_ragsage_import_pulls_no_provider_sdk() -> None:
    # The façades, models, ports and fakes must be importable without paying for
    # a provider SDK: `import ragsage` is what every consumer does first, and the
    # in-memory fakes are a complete engine on their own.
    modules, numpy_version = _probe_import_graph(("ragsage", "ragsage.fakes", "ragsage.ports"))

    leaked = sorted(name for name in (*_FORBIDDEN, *_PROVIDER_SDKS) if name in modules)
    assert not leaked, (
        f"importing ragsage pulled {leaked}. The provider adapters are core "
        f"dependencies but must be reached explicitly via ragsage.providers, never "
        f"re-exported from the top-level package."
    )

    _assert_numpy_is_light(numpy_version, "ragsage")


def test_providers_import_graph_is_the_heavy_one() -> None:
    # The canary. The two guards above are only meaningful while the provider SDKs
    # really are the heavy half; if voyageai/openai ever stop pulling this stack,
    # this test fails and the assertions above should be re-derived rather than
    # left to pass vacuously.
    modules, numpy_version = _probe_import_graph(("ragsage.providers",))

    missing = sorted(name for name in _PROVIDER_SDKS if name not in modules)
    assert not missing, (
        f"ragsage.providers did not import {missing}. The adapters are supposed to "
        f"talk to these SDKs directly — if this changed, the isolation the other "
        f"guards assert may no longer be guarding anything."
    )
    assert "numpy" in modules and numpy_version is not None, (
        "ragsage.providers no longer pulls numpy. That is good news, but it means "
        "the parser/provider split is no longer the numpy boundary these tests "
        "describe — revisit them."
    )
