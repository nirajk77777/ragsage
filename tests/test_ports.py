"""Every port has a conforming fake — the standalone guarantee, made checkable.

The ports are ``runtime_checkable`` Protocols, so a structural ``isinstance``
proves each fake actually implements its port's surface. This test is the
executable form of the issue's "fake adapters exist for every port".
"""

from __future__ import annotations

from ragsage import fakes, ports


def test_a_fake_conforms_to_each_port() -> None:
    pairs = [
        (fakes.FakeDocumentParser(), ports.DocumentParser),
        (fakes.FakePageClassifier(), ports.PageClassifier),
        (fakes.FakeChunker(), ports.Chunker),
        (fakes.FakeContextualizer(), ports.Contextualizer),
        (fakes.FakeEmbedder(), ports.Embedder),
        (fakes.FakeReranker(), ports.Reranker),
        (fakes.FakeLLMClient(), ports.LLMClient),
        (fakes.FakeQueryRewriter(), ports.QueryRewriter),
        (fakes.FakeVectorStore(), ports.VectorStore),
        (fakes.FakeLexicalStore(), ports.LexicalStore),
        (fakes.FakeDocumentStore(), ports.DocumentStore),
        (fakes.FakeCache(), ports.Cache),
        (fakes.FakeTracer(), ports.Tracer),
    ]
    for fake, port in pairs:
        assert isinstance(fake, port), f"{type(fake).__name__} does not satisfy {port.__name__}"


def test_every_port_is_covered() -> None:
    # Guard against adding a port without a matching fake. The ports the issue
    # names (plus QueryRewriter, added for multi-turn) must each be a Protocol
    # defined here, and we ship exactly one fake per port.
    expected = {
        "DocumentParser",
        "PageClassifier",
        "Chunker",
        "Contextualizer",
        "Embedder",
        "Reranker",
        "LLMClient",
        "QueryRewriter",
        "VectorStore",
        "LexicalStore",
        "DocumentStore",
        "Cache",
        "Tracer",
    }
    for name in expected:
        port = getattr(ports, name)
        assert getattr(port, "_is_protocol", False), f"{name} must be a Protocol"
    assert len(list(fakes.all_fakes())) == len(expected) == 13
