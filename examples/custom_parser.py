"""Bring your own parser — bypass the built-in ``HeuristicBackend`` entirely.

    python examples/custom_parser.py

This is the most likely reason to reach for a port. ragsage's own
:class:`~ragsage.parsing.backend.HeuristicBackend` reads PDF, DOCX, PPTX, HTML and
plain text; hand it the FAQ export below and it would still ingest, but as one
flat page of comma-separated soup — every citation pointing at "page 1". A parser
that understands your format keeps the structure that makes citations useful.

``FaqCsvParser`` satisfies the :class:`~ragsage.ports.DocumentParser` port by
shape alone, so, like any adapter, it subclasses nothing. It does import the
domain models: those are the vocabulary crossing the seam, not an inheritance
hierarchy. Two obligations come with the port:

* **Identity must be content-derived.** The pipeline dedups on ``content_hash``,
  so re-ingesting the same bytes has to produce the same hash and id — sha256 of
  the raw content, first 16 chars as the id, exactly as the built-in paths do.
* **Bring a chunker too, if you replace the paired one.** ``HeuristicBackend``
  implements ``parse`` and ``chunk`` together, the first stashing pages for the
  second. Replacing only its parser would strand that hand-off, so this example
  pairs a standalone parser with a standalone chunker.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io

from ragsage import (
    Document,
    IngestionPipeline,
    Page,
    ParsedDocument,
    QueryEngine,
    RawSource,
    Scope,
)
from ragsage.fakes import FakeEngineKit

FAQ_CSV = b"""question,answer
How do I reset my password?,Open Settings then Security and choose Reset password.
How long do I have to ask for a refund?,Purchases have a 30-day refund window.
Which regions are supported?,The service runs in Frankfurt, Oregon and Singapore.
"""


class FaqCsvParser:
    """Parses a two-column FAQ export into one page per question.

    Page numbers are the engine's finest citation granularity, so mapping one row
    to one page is what makes an answer cite the entry it actually came from.
    """

    def parse(self, source: RawSource) -> ParsedDocument:
        content = source.content if source.content is not None else _read(source.path)
        digest = hashlib.sha256(content).hexdigest()
        document = Document(
            id=digest[:16],
            source=source.name,
            content_hash=digest,
            # Whatever you put here is yours to filter on later; the engine only
            # ever reads it back out.
            metadata={"format": "faq-csv"},
        )
        rows = csv.DictReader(io.StringIO(content.decode("utf-8")))
        pages = [
            Page(number=number, text=f"{row['question']} {row['answer']}")
            for number, row in enumerate(rows, start=1)
        ]
        return ParsedDocument(document=document, pages=pages)


def _read(path: str | None) -> bytes:
    assert path is not None  # RawSource guarantees content or path
    with open(path, "rb") as handle:
        return handle.read()


async def main() -> None:
    kit = FakeEngineKit()
    scope = Scope(namespace="local")
    pipeline = IngestionPipeline(
        parser=FaqCsvParser(),  # in place of HeuristicBackend
        classifier=kit.classifier,
        chunker=kit.chunker,
        contextualizer=kit.contextualizer,
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        llm=kit.llm,
        cache=kit.cache,
    )
    engine = QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
    )

    result = await pipeline.ingest(RawSource(name="support-faq.csv", content=FAQ_CSV), scope)
    print(f"ingested {result.document.source}: {result.chunk_count} chunks (one per FAQ entry)")

    answer = await engine.query("How long is the refund window?", scope)
    print(f"A: {answer.text}")
    for citation in answer.citations:
        print(f"   [{citation.marker}] FAQ entry {citation.page}: {citation.quote}")


if __name__ == "__main__":
    asyncio.run(main())
