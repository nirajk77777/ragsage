"""The whole ingest-and-query loop, offline — no network, no database, no API key.

    python examples/fakes_end_to_end.py

``ragsage.fakes`` ships a genuine in-memory implementation of every port, so the
engine is runnable the moment it is installed. That is not a testing convenience:
if the loop below runs with nothing but fakes, the library really is free of web,
tenant, and provider concepts, and the only difference between this script and a
production deployment is which adapters get passed to the two constructors.

The second question is the interesting one. Nothing in the corpus answers it, so
the engine says so instead of inventing an answer — the honest not-found path is
part of the contract, not a failure mode.
"""

from __future__ import annotations

import asyncio

from ragsage import IngestionPipeline, QueryEngine, RawSource, Scope
from ragsage.fakes import FakeEngineKit

DOCUMENT = b"""Paris is the capital of France. It sits on the Seine river.
The Metro opened in 1900 and now carries four million riders a day.
"""


async def main() -> None:
    # One kit holds one instance of each fake, so the pipeline that writes and
    # the engine that reads share the same in-memory stores.
    kit = FakeEngineKit()

    # A Scope is the engine's only isolation concept: an opaque namespace the
    # caller chooses. A backend passes a tenant id here; a script passes "local".
    scope = Scope(namespace="local")

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
    )
    engine = QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
    )

    result = await pipeline.ingest(RawSource(name="france.txt", content=DOCUMENT), scope)
    print(f"ingested {result.document.source} as {result.document.id}")
    print(f"chunks stored: {result.chunk_count}")

    for question in ("What is the capital of France?", "Who won the 1998 World Cup final?"):
        answer = await engine.query(question, scope)
        print(f"\nQ: {question}")
        print(f"A: {answer.text}  [{answer.outcome}, grounded={answer.grounded}]")
        for citation in answer.citations:
            print(f"   [{citation.marker}] {citation.document_id} page {citation.page}")


if __name__ == "__main__":
    asyncio.run(main())
