"""The ``ragsage`` command line — shipped *inside* the library.

Its reason to exist is architectural proof: the RAG engine can be driven
end-to-end with no web server, no database, and no tenancy code. ``ragsage
ingest ./docs`` then ``ragsage query "…"`` runs the whole loop against the
in-memory :mod:`ragsage.fakes` adapters, persisting the corpus to a JSON file
between the two commands so they work as separate processes and no network is
ever touched.

Because the CLI wires nothing but ports and fakes, it is the living proof that
the library is self-contained: swapping the fakes for real adapters is the only
difference between this and the backend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import ragsage
from ragsage import IngestionConfig, IngestionPipeline, QueryEngine, RawSource, Scope
from ragsage.fakes import FakeEngineKit

_DEFAULT_STORE = ".ragsage/state.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragsage",
        description="Drive the ragsage engine from the command line.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ragsage {ragsage.__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    info = subparsers.add_parser("info", help="Print engine information")
    info.set_defaults(func=_cmd_info)

    ingest = subparsers.add_parser("ingest", help="Ingest a file or folder into the corpus")
    ingest.add_argument("path", help="A document file, or a folder of documents")
    ingest.add_argument(
        "--store", default=_DEFAULT_STORE, help=f"Corpus state file (default: {_DEFAULT_STORE})"
    )
    ingest.add_argument("--namespace", default="local", help="Scope namespace (default: local)")
    ingest.add_argument("--no-contextualize", action="store_true", help="Skip contextual retrieval")
    ingest.set_defaults(func=_cmd_ingest)

    query = subparsers.add_parser("query", help="Ask a question against the corpus")
    query.add_argument("question", help="The question to answer")
    query.add_argument(
        "--store", default=_DEFAULT_STORE, help=f"Corpus state file (default: {_DEFAULT_STORE})"
    )
    query.add_argument("--namespace", default="local", help="Scope namespace (default: local)")
    query.set_defaults(func=_cmd_query)

    evaluate = subparsers.add_parser(
        "eval",
        help="Score the built-in golden set on the standard RAG metrics",
    )
    evaluate.set_defaults(func=_cmd_eval)

    return parser


def _cmd_info(_: argparse.Namespace) -> int:
    # A deliberately tiny action: it runs the library with nothing web- or
    # tenant-shaped in scope, which is exactly the standalone guarantee.
    scope = ragsage.Scope(namespace="local")
    print(f"ragsage {ragsage.__version__}")
    print(f"default scope namespace: {scope.namespace}")
    return 0


def _load_kit(store: Path) -> FakeEngineKit:
    kit = FakeEngineKit()
    if store.exists():
        kit.restore(json.loads(store.read_text()))
    return kit


def _save_kit(kit: FakeEngineKit, store: Path) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(kit.snapshot()))


def _iter_sources(root: Path) -> list[RawSource]:
    """Turn a file, or every readable file in a folder, into raw sources."""
    if root.is_dir():
        files = sorted(p for p in root.iterdir() if p.is_file() and not p.name.startswith("."))
    else:
        files = [root]
    return [RawSource(name=p.name, path=str(p)) for p in files]


def _cmd_ingest(args: argparse.Namespace) -> int:
    root = Path(args.path)
    if not root.exists():
        print(f"ingest: path not found: {root}")
        return 1

    store = Path(args.store)
    kit = _load_kit(store)
    pipeline = IngestionPipeline(
        parser=kit.parser,
        classifier=kit.classifier,
        chunker=kit.chunker,
        contextualizer=kit.contextualizer,
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        llm=kit.llm,
        cache=kit.cache,
        tracer=kit.tracer,
    )
    scope = Scope(namespace=args.namespace)
    config = IngestionConfig(contextualize=not args.no_contextualize)

    sources = _iter_sources(root)
    if not sources:
        print(f"ingest: no files found under {root}")
        return 1

    async def run() -> None:
        for source in sources:
            result = await pipeline.ingest(source, scope, config)
            if result.deduplicated:
                print(f"  = {source.name}: already ingested (duplicate)")
            else:
                vision = result.route_counts.get(ragsage.PageRoute.VISION, 0)
                suffix = f", {vision} via vision" if vision else ""
                print(f"  + {source.name}: {result.chunk_count} chunks{suffix}")

    asyncio.run(run())
    _save_kit(kit, store)
    print(f"corpus saved to {store}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    store = Path(args.store)
    if not store.exists():
        print(f"query: no corpus at {store} — run `ragsage ingest` first")
        return 1

    kit = _load_kit(store)
    engine = QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
        tracer=kit.tracer,
    )
    scope = Scope(namespace=args.namespace)

    answer = asyncio.run(engine.query(args.question, scope))
    print(answer.text)
    if answer.citations:
        print()
        print("Sources:")
        for citation in answer.citations:
            print(f"  [{citation.marker}] {citation.document_id} (page {citation.page})")
    return 0


def _cmd_eval(_: argparse.Namespace) -> int:
    # The offline eval gate, runnable anywhere: ingest the built-in golden set
    # through the fakes, score it, and exit non-zero if any metric is below its
    # agreed threshold — which is exactly what a CI step needs.
    from ragsage.goldens import default_golden_set, run_golden_eval

    golden = default_golden_set()
    report = asyncio.run(run_golden_eval(golden))
    check = report.check(golden.thresholds)

    print("ragsage offline eval — golden set")
    print(f"  examples:          {len(report.results)}")
    print(f"  answer accuracy:   {report.answer_accuracy:.3f}")
    print(f"  faithfulness:      {report.mean_faithfulness:.3f}")
    print(f"  answer relevancy:  {report.mean_answer_relevancy:.3f}")
    print(f"  context precision: {report.mean_context_precision:.3f}")
    print(f"  context recall:    {report.mean_context_recall:.3f}")
    print(f"  grounded rate:     {report.grounded_rate:.3f}")
    if check.passed:
        print("PASS — all metrics met their thresholds")
        return 0
    print("FAIL — below threshold:")
    for failure in check.failures:
        print(f"  - {failure}")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
