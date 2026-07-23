"""Tuning knobs for the two pipelines, kept out of the domain models.

These are *policy*, not data: how big chunks are, whether to contextualise,
how many candidates to retrieve before reranking. They live in their own file
so the façade signatures (`ingest(source, scope, config)`, `query(question,
scope, options)`) read cleanly and defaults live in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionConfig:
    """How a document is turned into stored, retrievable chunks.

    ``contextualize`` toggles Anthropic-style contextual retrieval — prepending
    an LLM-written sentence situating each chunk in its document before
    embedding. It is on by default (better retrieval) with a per-call off
    switch, matching the per-workspace toggle the backend exposes.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    contextualize: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")


@dataclass(frozen=True)
class QueryOptions:
    """How a question is answered: the retrieve -> rerank -> generate funnel.

    Defaults follow the spec's ratio — retrieve a wide candidate set, rerank
    with a cross-encoder, and hand only the top few to the model. ``min_score``
    is the floor below which retrieval is treated as empty, which is what turns
    a weak match into an honest "not found" instead of a hallucination.
    """

    retrieve_k: int = 20
    rerank_k: int = 5
    context_k: int = 3
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if min(self.retrieve_k, self.rerank_k, self.context_k) <= 0:
            raise ValueError("retrieve_k, rerank_k and context_k must be positive")
