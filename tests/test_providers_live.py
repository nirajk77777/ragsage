"""Opt-in smoke test against the *real* Voyage and OpenAI APIs.

Everything else in the suite is hermetic; this file is the one place that spends
money and needs a network, so it is skipped unless a human turns it on:

    RAGSAGE_LIVE_PROVIDERS=1 OPENAI_API_KEY=... VOYAGE_API_KEY=... uv run pytest -m live_providers

It exists because stubs prove the adapter's *shape* and nothing about whether the
request the SDK builds is one the provider accepts — a renamed parameter or a
retired model id is invisible to every other test here. It reads keys from the
environment, which the library itself never does: a test *is* the caller, and the
caller is exactly who is allowed to decide where credentials come from.
"""

from __future__ import annotations

import os

import pytest

from ragsage.models import Turn
from ragsage.providers import (
    OpenAIQueryRewriter,
    ProviderConfig,
    VoyageEmbedder,
    build_provider_clients,
)

pytestmark = [
    pytest.mark.live_providers,
    pytest.mark.skipif(
        os.getenv("RAGSAGE_LIVE_PROVIDERS") != "1"
        or not os.getenv("OPENAI_API_KEY")
        or not os.getenv("VOYAGE_API_KEY"),
        reason="live provider calls are opt-in: set RAGSAGE_LIVE_PROVIDERS=1 and both API keys",
    ),
]


@pytest.fixture
def clients() -> object:
    return build_provider_clients(
        ProviderConfig(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            voyage_api_key=os.environ["VOYAGE_API_KEY"],
        )
    )


async def test_voyage_embeds_for_real(clients: object) -> None:
    vectors = await VoyageEmbedder(clients.embeddings).embed(["hello world"])  # type: ignore[attr-defined]
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


async def test_openai_rewrites_for_real(clients: object) -> None:
    rewritten = await OpenAIQueryRewriter(clients.rewrite).rewrite(  # type: ignore[attr-defined]
        "What about its price?",
        [Turn(question="Tell me about the Acme plan.", answer="Acme is our standard plan.")],
    )
    assert "acme" in rewritten.lower()
