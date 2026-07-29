"""Bring your own embedder — the ports are ``Protocol``s, so nothing is imported.

    python examples/custom_embedder.py

``TrigramEmbedder`` below is a plain class in your codebase. It subclasses
nothing, registers nowhere, and imports nothing from ``ragsage`` — it satisfies
the :class:`ragsage.ports.Embedder` port purely by having an ``async embed`` of
the right shape. That is the whole point of structural typing here: the
dependency arrow points inward, so your adapter may know about ragsage but
ragsage never has to know about your adapter. Swap Voyage for a local model, a
different vendor, or a hash trick like this one, and no ragsage code changes.

The one thing you do owe the engine is self-consistency: the same embedder must
produce the query vector and the stored vectors, at one fixed width. Real stores
pin that width in a column type; the in-memory fake here doesn't care what it is.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence

from ragsage import IngestionPipeline, QueryEngine, RawSource, Scope
from ragsage.fakes import FakeEngineKit
from ragsage.ports import Embedder

DOCUMENT = b"""The Antikythera mechanism is an ancient Greek analogue computer.
It was recovered from a shipwreck in 1901 and modelled the motion of the planets.
"""


class TrigramEmbedder:
    """Embeds text as an L2-normalised bag of character trigrams.

    Deliberately not a neural model: character trigrams need no weights, no
    network, and no API key, which keeps the example about the seam rather than
    about the vendor. A real adapter would batch these texts into one provider
    call — the port is batched for exactly that reason.
    """

    def __init__(self, *, dim: int = 512) -> None:
        self._dim = dim

    async def embed(self, texts: Sequence[str]) -> Sequence[tuple[float, ...]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> tuple[float, ...]:
        counts = [0.0] * self._dim
        lowered = f" {text.lower()} "
        for i in range(len(lowered) - 2):
            digest = hashlib.blake2b(lowered[i : i + 3].encode(), digest_size=8).digest()
            counts[int.from_bytes(digest, "big") % self._dim] += 1.0
        norm = sum(c * c for c in counts) ** 0.5
        return tuple(c / norm for c in counts) if norm else tuple(counts)


async def main() -> None:
    embedder = TrigramEmbedder()

    # The ports are runtime-checkable, so conformance is observable rather than
    # a matter of faith — and it holds despite TrigramEmbedder never naming it.
    print(f"conforms to the Embedder port: {isinstance(embedder, Embedder)}")

    kit = FakeEngineKit()
    scope = Scope(namespace="local")
    pipeline = IngestionPipeline(
        parser=kit.parser,
        classifier=kit.classifier,
        chunker=kit.chunker,
        contextualizer=kit.contextualizer,
        embedder=embedder,  # the only line that differs from the fakes example
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        document_store=kit.document_store,
        llm=kit.llm,
        cache=kit.cache,
    )
    engine = QueryEngine(
        embedder=embedder,  # the same instance: a corpus is only searchable by
        vector_store=kit.vector_store,  # the embedder that wrote it
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
    )

    result = await pipeline.ingest(RawSource(name="antikythera.txt", content=DOCUMENT), scope)
    print(f"ingested {result.document.source}: {result.chunk_count} chunks")

    answer = await engine.query("What was recovered from a shipwreck?", scope)
    print(f"A: {answer.text}")


if __name__ == "__main__":
    asyncio.run(main())
