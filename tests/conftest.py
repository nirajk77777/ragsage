"""Shared Seam-1 fixtures: a fully-wired engine over in-memory fakes.

Every ragsage test drives the public façades — :class:`IngestionPipeline`,
:class:`QueryEngine` — with injected fakes and asserts on what comes back out.
These fixtures assemble that wiring once. The ``pipeline`` and ``engine`` share a
single :class:`FakeEngineKit`, so a document ingested through one is queryable
through the other, exactly as the real backend wires them.
"""

from __future__ import annotations

import pytest

from ragsage import IngestionPipeline, QueryEngine, Scope
from ragsage.fakes import FakeEngineKit


@pytest.fixture
def kit() -> FakeEngineKit:
    return FakeEngineKit()


@pytest.fixture
def scope() -> Scope:
    return Scope(namespace="local")


@pytest.fixture
def pipeline(kit: FakeEngineKit) -> IngestionPipeline:
    return IngestionPipeline(
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


@pytest.fixture
def engine(kit: FakeEngineKit) -> QueryEngine:
    return QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
        tracer=kit.tracer,
    )
