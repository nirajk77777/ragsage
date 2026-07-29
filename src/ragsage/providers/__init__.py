"""The shipped production adapters: Voyage for retrieval, OpenAI for generation.

Five adapters cover the five model-shaped ports — ``Embedder``, ``Reranker``,
``LLMClient``, ``Contextualizer``, ``QueryRewriter`` — so ``pip install ragsage``
plus two API keys is a working engine, not a pile of interfaces waiting for the
consumer to write 750 lines of adapter code (ADR-0002). They are *core*, not an
extra: an engine you have to assemble before it does anything is reusable in
principle and unusable in practice.

They remain adapters, not the engine. Each conforms to its port by shape, the
ports stay public, and the in-memory :mod:`ragsage.fakes` are still a complete
alternative — swap one of these for a local embedder or a different provider and
nothing upstream changes.

**Importing this package pulls the provider SDKs**, and with ``voyageai`` comes
``numpy`` and HuggingFace ``tokenizers``. That is why the adapters live behind
this namespace and are *not* re-exported from the top-level ``ragsage`` package:
``import ragsage`` and ``import ragsage.parsing`` stay free of that stack, which
is the portability promise ADR-0001 kept and ``tests/test_dependency_guard.py``
enforces. Import ``ragsage.providers`` only where you actually intend to talk to
a provider.
"""

from ragsage.providers.clients import ProviderClients, build_provider_clients
from ragsage.providers.config import CallConfig, ProviderConfig
from ragsage.providers.openai import (
    OpenAIContextualizer,
    OpenAILLMClient,
    OpenAIQueryRewriter,
)
from ragsage.providers.voyage import VoyageEmbedder, VoyageReranker

__all__ = [
    "CallConfig",
    "OpenAIContextualizer",
    "OpenAILLMClient",
    "OpenAIQueryRewriter",
    "ProviderClients",
    "ProviderConfig",
    "VoyageEmbedder",
    "VoyageReranker",
    "build_provider_clients",
]
