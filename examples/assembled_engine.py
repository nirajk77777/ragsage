"""RagSage.from_config: the whole engine from one config object.

The other three examples hand-wire ports, because that is what they are about.
This one is the opposite: the shortest path from a DSN and two API keys to a
grounded answer.

Unlike its siblings this one needs a real Postgres, so it *skips* rather than fails
when there isn't one — which is why CI can run all four unconditionally. Point it at
a database with the `vector` extension available:

    TEST_DATABASE_URL=postgresql://user:pw@localhost:5432/ragsage \\
        python examples/assembled_engine.py

Nothing here reads the environment on ragsage's behalf: this script reads it and
passes values in, because the library deliberately never does (ADR-0002).
"""

from __future__ import annotations

import asyncio
import os
import sys

from ragsage.fakes import FakeEmbedder, FakeEngineKit
from ragsage.models import RawSource
from ragsage.providers import ProviderConfig
from ragsage.sage import ComputeKit, RagSage, RagSageConfig
from ragsage.scope import Scope
from ragsage.storage import PostgresConfig

# Small enough to read; the real default is 1024, matching voyage-3-large.
EMBEDDING_DIM = 8

CORPUS = b"""# Ancient technology

## The Antikythera mechanism

It modelled the motion of the planets and predicted eclipses decades ahead.
"""


def offline_compute() -> ComputeKit:
    """Swap the model adapters for fakes so this runs without API keys.

    This *is* the overridability claim, exercised: `from_config` builds Voyage and
    OpenAI adapters, and one `ComputeKit` argument replaces them wholesale. Real
    usage omits this entirely and passes the keys instead.
    """
    kit = FakeEngineKit()
    return ComputeKit(
        parser=kit.parser,
        classifier=kit.classifier,
        chunker=kit.chunker,
        contextualizer=kit.contextualizer,
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        reranker=kit.reranker,
        llm=kit.llm,
    )


async def main() -> int:
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        print("skipped: set TEST_DATABASE_URL to a Postgres with the vector extension")
        return 0

    config = RagSageConfig(
        postgres=PostgresConfig(dsn=dsn, embedding_dim=EMBEDDING_DIM),
        providers=ProviderConfig(
            # Real usage: os.environ["OPENAI_API_KEY"], os.environ["VOYAGE_API_KEY"].
            openai_api_key="",
            voyage_api_key="",
            embedding_model="fake-embedder",
        ),
    )
    sage = RagSage.from_config(config, compute=offline_compute())
    scope = Scope(namespace="examples")

    try:
        # Idempotent: safe on every start, not just the first.
        await sage.migrate()
        # Keep the example repeatable — it is the same namespace every run.
        await sage.purge(scope)

        result = await sage.ingest(RawSource(name="devices.md", content=CORPUS), scope)
        print(f"ingested {result.document.source}: {result.chunk_count} chunks")

        answer = await sage.query("What did the Antikythera mechanism model?", scope)
        print(f"A: {answer.text}  [{answer.outcome.value}]")
        for citation in answer.citations:
            print(f"   [{citation.marker}] {citation.document_id} page {citation.page}")

        # purge is load-bearing, not cleanup: ADR-0002 dropped the FK that used to
        # cascade chunk deletion from the consumer's user row, so this is now the
        # only thing that removes a namespace's corpus.
        await sage.purge(scope)
    finally:
        # Close the pool. ragsage owns it, so ragsage has to be told to let go.
        await sage.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
