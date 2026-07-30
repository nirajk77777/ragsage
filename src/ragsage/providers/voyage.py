"""Voyage-backed adapters for the retrieval-side model ports.

Embedding and reranking are hard-bound to Voyage (ADR-0002) and reach it directly
through LangChain — there is no gateway in the path. Each adapter conforms *by
shape* to a port in :mod:`ragsage.ports` (``Embedder``, ``Reranker``) and holds
one of the singleton clients built in :mod:`ragsage.providers.clients`.

Neither takes a :data:`~ragsage.providers.config.CallConfig`: the Voyage
integration accepts no LangChain callbacks, so there is nothing to thread through
and pretending otherwise would be a lie in the signature. A consumer that needs
these calls observed has to wrap the adapter itself.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank

from ragsage.models import ScoredChunk, Vector


class VoyageEmbedder:
    """``Embedder`` over Voyage's batch embeddings endpoint.

    The vectors it returns are as wide as the configured embedding model makes
    them, which is why the model is pinned per corpus: a store's vector column is
    sized to one model's output and re-indexing is the only way to change it.
    """

    def __init__(self, client: VoyageAIEmbeddings) -> None:
        self._client = client

    async def embed(self, texts: Sequence[str]) -> Sequence[Vector]:
        if not texts:
            return []
        rows = await self._client.aembed_documents(list(texts))
        return [tuple(row) for row in rows]


class VoyageReranker:
    """``Reranker`` over Voyage's rerank endpoint (a cross-encoder).

    Each candidate is handed to Voyage as a :class:`~langchain_core.documents.Document`
    tagged with its position in the input list; the reranked replies carry that tag
    and a ``relevance_score`` back, so the adapter rebuilds the
    :class:`ScoredChunk` list from its *own* candidates by index — the model only
    ever sees text, never the engine's chunk identities. A reply whose tag is
    missing or out of range is dropped rather than trusted, so a malformed
    response can never mislabel a citation. The client scores every candidate (it was built with ``top_k=None``);
    the port's per-call ``top_k`` is applied here.
    """

    def __init__(self, client: VoyageAIRerank) -> None:
        self._client = client

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int
    ) -> Sequence[ScoredChunk]:
        if not candidates:
            return []
        documents = [
            Document(page_content=c.chunk.text, metadata={"index": i})
            for i, c in enumerate(candidates)
        ]
        reranked = await self._client.acompress_documents(documents, query)
        scored: list[ScoredChunk] = []
        for doc in reranked:
            index = doc.metadata.get("index")
            if not isinstance(index, int) or not (0 <= index < len(candidates)):
                continue
            score = float(doc.metadata.get("relevance_score", 0.0))
            scored.append(ScoredChunk(chunk=candidates[index].chunk, score=score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
