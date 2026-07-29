"""The shipped provider adapters, exercised against stubbed provider clients.

No network. Each adapter holds one LangChain client, so a stub standing in for
that client is enough to assert the two things that actually matter: that the
adapter conforms to its port by shape, and that it builds the *right request* and
maps the reply back into domain objects. Prompt wording is asserted only where it
is load-bearing (the document body must be the prompt prefix for OpenAI's cache
to hit; a rerank reply must be rejoined to candidates by the adapter's own index
tag), never word-for-word.

The live smoke test that does touch the real APIs is in ``test_providers_live``
and is skipped unless explicitly enabled.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from ragsage import ports
from ragsage.models import Chunk, Document, PageImage, ScoredChunk, Turn
from ragsage.providers import (
    OpenAIContextualizer,
    OpenAILLMClient,
    OpenAIQueryRewriter,
    ProviderConfig,
    VoyageEmbedder,
    VoyageReranker,
    build_provider_clients,
)

# --------------------------------------------------------------------------- #
# Stubs standing in for the LangChain clients
# --------------------------------------------------------------------------- #


class StubChat:
    """A ``ChatOpenAI`` stand-in that records calls and replays canned replies."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies) or [""]
        self.calls: list[tuple[list[BaseMessage], Any]] = []

    async def ainvoke(self, messages: list[BaseMessage], config: Any = None) -> BaseMessage:
        self.calls.append((messages, config))
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        return AIMessage(content=reply)

    async def astream(
        self, messages: list[BaseMessage], config: Any = None
    ) -> AsyncIterator[BaseMessage]:
        self.calls.append((messages, config))
        for reply in self._replies:
            yield AIMessageChunk(content=reply)


class StubEmbeddings:
    """A ``VoyageAIEmbeddings`` stand-in returning fixed rows."""

    def __init__(self, *rows: list[float]) -> None:
        self._rows = list(rows)
        self.calls: list[list[str]] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self._rows[: len(texts)]


class StubRerank:
    """A ``VoyageAIRerank`` stand-in replaying a canned, out-of-order ranking."""

    def __init__(self, *ranked: LCDocument) -> None:
        self._ranked = list(ranked)
        self.calls: list[tuple[list[LCDocument], str]] = []

    async def acompress_documents(
        self, documents: Sequence[LCDocument], query: str
    ) -> Sequence[LCDocument]:
        self.calls.append((list(documents), query))
        return self._ranked


def _chunk(text: str, ordinal: int = 0) -> Chunk:
    return Chunk(id=f"doc:{ordinal}", document_id="doc", text=text, page=1, ordinal=ordinal)


def _scored(text: str, ordinal: int) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(text, ordinal), score=0.0)


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)  # type: ignore[union-attr]
    )


# --------------------------------------------------------------------------- #
# Port conformance
# --------------------------------------------------------------------------- #


def test_each_adapter_conforms_to_its_port() -> None:
    chat = StubChat()
    pairs = [
        (VoyageEmbedder(StubEmbeddings()), ports.Embedder),  # type: ignore[arg-type]
        (VoyageReranker(StubRerank()), ports.Reranker),  # type: ignore[arg-type]
        (OpenAIContextualizer(chat), ports.Contextualizer),  # type: ignore[arg-type]
        (OpenAIQueryRewriter(chat), ports.QueryRewriter),  # type: ignore[arg-type]
        (OpenAILLMClient(generation=chat, vision=chat), ports.LLMClient),  # type: ignore[arg-type]
    ]
    for adapter, port in pairs:
        assert isinstance(adapter, port), (
            f"{type(adapter).__name__} does not satisfy {port.__name__}"
        )


def test_no_langfuse_import_anywhere_in_ragsage() -> None:
    # ADR-0003's Langfuse threading stays in the consuming backend. ragsage takes
    # an opaque per-call config and knows nothing about what is inside it; tracing
    # proper arrives through the injected Tracer/Span ports.
    src = Path(__file__).resolve().parents[1] / "src" / "ragsage"
    importer = re.compile(r"^\s*(?:import|from)\s+langfuse", re.MULTILINE)
    offenders = sorted(p.name for p in src.rglob("*.py") if importer.search(p.read_text()))
    assert not offenders, f"langfuse is imported in {offenders}; it must stay in the consumer"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_config_is_frozen_and_takes_keys_from_the_caller() -> None:
    config = ProviderConfig(openai_api_key="sk-openai", voyage_api_key="pa-voyage")
    assert config.openai_api_key == "sk-openai"
    with pytest.raises(Exception):  # noqa: B017 — dataclasses raise FrozenInstanceError
        config.openai_api_key = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_model": ""},
        {"rerank_model": "   "},
        {"contextualize_model": ""},
        {"generation_model": ""},
        {"vision_model": ""},
        {"timeout_seconds": 0.0},
        {"timeout_seconds": -1.0},
        {"context_max_tokens": 0},
        {"rewrite_max_tokens": -5},
        {"vision_max_tokens": 0},
    ],
)
def test_config_rejects_nonsense(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ProviderConfig(openai_api_key="k", voyage_api_key="k", **overrides)  # type: ignore[arg-type]


def test_empty_keys_still_build_clients() -> None:
    # Dev, CI, and any harness that swaps fakes in at the ports boot with no keys.
    # Construction must succeed (the SDKs reject an empty key outright), leaving a
    # real call to fail with an auth error at request time instead.
    clients = build_provider_clients(ProviderConfig(openai_api_key="", voyage_api_key=""))
    assert clients.generation.openai_api_key.get_secret_value() == "unset"


def test_clients_are_built_per_role() -> None:
    config = ProviderConfig(
        openai_api_key="sk-openai",
        voyage_api_key="pa-voyage",
        embedding_model="voyage-3-large",
        rerank_model="rerank-2",
        contextualize_model="cheap-model",
        generation_model="big-model",
        vision_model="eyes-model",
        timeout_seconds=12.0,
        context_max_tokens=64,
        rewrite_max_tokens=128,
        vision_max_tokens=2048,
    )
    clients = build_provider_clients(config)

    assert clients.embeddings.model == "voyage-3-large"
    # The port passes top_k per call, so the client must not fix one per deploy.
    assert clients.rerank.top_k is None
    assert clients.generation.model_name == "big-model"
    assert clients.generation.streaming is True
    assert clients.contextualize.model_name == "cheap-model"
    # ChatOpenAI normalises ``max_completion_tokens`` onto its ``max_tokens`` field.
    assert clients.contextualize.max_tokens == 64
    # Rewriting shares the cheap tier but gets its own budget.
    assert clients.rewrite.model_name == "cheap-model"
    assert clients.rewrite.max_tokens == 128
    assert clients.vision.model_name == "eyes-model"
    assert clients.vision.max_tokens == 2048
    # Generation is uncapped: the answer's length is the model's to decide.
    assert clients.generation.max_tokens is None
    assert clients.generation.request_timeout == 12.0


# --------------------------------------------------------------------------- #
# OpenAI adapters
# --------------------------------------------------------------------------- #


async def test_contextualizer_sends_the_document_as_the_prompt_prefix() -> None:
    chat = StubChat("A pricing table from the Acme plan.")
    adapter = OpenAIContextualizer(chat, config={"tags": ["caller-supplied"]})  # type: ignore[arg-type]

    embed_text = await adapter.contextualize(
        Document(id="doc", source="acme.pdf", content_hash="h"),
        _chunk("Standard tier costs $10."),
        full_text="Acme pricing, in full.",
    )

    messages, config = chat.calls[0]
    # The whole document goes first, as the literal prefix, so OpenAI's automatic
    # prefix caching hits across every chunk of the same document.
    assert "Acme pricing, in full." in _text_of(messages[0])
    assert "Standard tier costs $10." in _text_of(messages[1])
    # The opaque per-call config is threaded through untouched.
    assert config == {"tags": ["caller-supplied"]}
    assert embed_text == "A pricing table from the Acme plan.\n\nStandard tier costs $10."


async def test_contextualizer_falls_back_to_the_raw_chunk() -> None:
    adapter = OpenAIContextualizer(StubChat("   "))
    embed_text = await adapter.contextualize(
        Document(id="doc", source="acme.pdf", content_hash="h"),
        _chunk("Standard tier costs $10."),
        full_text="Acme pricing.",
    )
    # An empty context must not pollute the vector with a leading blank line.
    assert embed_text == "Standard tier costs $10."


async def test_rewriter_skips_the_call_with_no_history() -> None:
    chat = StubChat("never used")
    assert await OpenAIQueryRewriter(chat).rewrite("What is the price?", []) == "What is the price?"
    assert chat.calls == []


async def test_rewriter_sends_the_transcript_and_returns_the_standalone_question() -> None:
    chat = StubChat("What is the price of the Acme plan?")
    adapter = OpenAIQueryRewriter(chat)

    rewritten = await adapter.rewrite(
        "What about its price?",
        [Turn(question="Tell me about Acme.", answer="Acme is a plan.")],
    )

    prompt = _text_of(chat.calls[0][0][0])
    assert "Tell me about Acme." in prompt
    assert "Acme is a plan." in prompt
    assert "What about its price?" in prompt
    assert rewritten == "What is the price of the Acme plan?"


async def test_rewriter_keeps_the_question_when_the_model_says_nothing() -> None:
    rewritten = await OpenAIQueryRewriter(StubChat("")).rewrite(
        "What about its price?", [Turn(question="Acme?", answer="A plan.")]
    )
    assert rewritten == "What about its price?"


async def test_llm_client_streams_tokens_and_drops_empty_deltas() -> None:
    chat = StubChat("Acme ", "", "costs $10.")
    tokens = [t async for t in OpenAILLMClient(generation=chat, vision=chat).generate("prompt")]
    assert tokens == ["Acme ", "costs $10."]


async def test_llm_client_transcribes_an_image_as_a_data_uri() -> None:
    vision = StubChat("The scanned page said this.")
    adapter = OpenAILLMClient(generation=StubChat(), vision=vision, config={"tags": ["job"]})  # type: ignore[arg-type]

    text = await adapter.transcribe(PageImage(ref="p1", data=b"PNGBYTES"))

    message, config = vision.calls[0]
    blocks = message[0].content
    assert isinstance(blocks, list)
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")  # type: ignore[index]
    assert config == {"tags": ["job"]}
    assert text == "The scanned page said this."


async def test_llm_client_transcribes_nothing_when_the_page_has_no_bytes() -> None:
    # The parser couldn't render the page; there is nothing to send a vision model.
    vision = StubChat("should not be called")
    text = await OpenAILLMClient(generation=StubChat(), vision=vision).transcribe(
        PageImage(ref="p1", data=None)
    )
    assert text == ""
    assert vision.calls == []


# --------------------------------------------------------------------------- #
# Voyage adapters
# --------------------------------------------------------------------------- #


async def test_embedder_returns_vectors_as_tuples() -> None:
    client = StubEmbeddings([0.1, 0.2], [0.3, 0.4])
    vectors = await VoyageEmbedder(client).embed(["one", "two"])  # type: ignore[arg-type]
    assert vectors == [(0.1, 0.2), (0.3, 0.4)]
    assert client.calls == [["one", "two"]]


async def test_embedder_short_circuits_on_no_texts() -> None:
    client = StubEmbeddings()
    assert await VoyageEmbedder(client).embed([]) == []  # type: ignore[arg-type]
    assert client.calls == []


async def test_reranker_rejoins_candidates_by_index_and_applies_top_k() -> None:
    candidates = [_scored("alpha", 0), _scored("beta", 1), _scored("gamma", 2)]
    client = StubRerank(
        LCDocument(page_content="beta", metadata={"index": 1, "relevance_score": 0.4}),
        LCDocument(page_content="gamma", metadata={"index": 2, "relevance_score": 0.9}),
        LCDocument(page_content="alpha", metadata={"index": 0, "relevance_score": 0.1}),
    )

    result = await VoyageReranker(client).rerank("q", candidates, top_k=2)  # type: ignore[arg-type]

    sent, query = client.calls[0]
    assert query == "q"
    # Only text crosses the wire; identity is the adapter's own positional tag.
    assert [d.page_content for d in sent] == ["alpha", "beta", "gamma"]
    assert [d.metadata["index"] for d in sent] == [0, 1, 2]
    # Reordered by the model's score, truncated to the port's per-call top_k, and
    # each result carries the engine's original chunk, not a reconstructed one.
    assert [c.chunk.text for c in result] == ["gamma", "beta"]
    assert result[0].chunk is candidates[2].chunk
    assert result[0].score == 0.9


async def test_reranker_drops_replies_whose_index_it_cannot_trust() -> None:
    # A missing or out-of-range tag would otherwise attach a score — and a citation
    # — to the wrong chunk. Dropping is the only safe reading.
    candidates = [_scored("alpha", 0), _scored("beta", 1)]
    client = StubRerank(
        LCDocument(page_content="beta", metadata={"index": 1, "relevance_score": 0.4}),
        LCDocument(page_content="?", metadata={"relevance_score": 0.99}),
        LCDocument(page_content="?", metadata={"index": 17, "relevance_score": 0.98}),
    )
    result = await VoyageReranker(client).rerank("q", candidates, top_k=5)  # type: ignore[arg-type]
    assert [c.chunk.text for c in result] == ["beta"]


async def test_reranker_short_circuits_on_no_candidates() -> None:
    client = StubRerank()
    assert await VoyageReranker(client).rerank("q", [], top_k=3) == []  # type: ignore[arg-type]
    assert client.calls == []
