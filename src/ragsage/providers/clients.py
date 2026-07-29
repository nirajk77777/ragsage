"""The provider clients — built once, shared across every request.

There is no gateway. Each role's model is called directly through LangChain and
hard-bound to its provider: embedding and reranking go to Voyage, generation /
contextualization / query rewriting / vision go to OpenAI. The underlying
LangChain clients are stateless with respect to the caller and hold their own
connection pools, so they are constructed once at a composition root
(:func:`build_provider_clients`) and reused for the life of the process — a query
or an ingestion job binds a slim adapter around them rather than building a new
client.

The clients need no explicit teardown: LangChain manages the provider SDK's
transport internally, so there is nothing to ``aclose`` on shutdown.

Importing this module pulls the provider SDKs (and, through ``voyageai``,
``numpy`` and HuggingFace ``tokenizers``). That is deliberate and is why the
package is ``ragsage.providers`` rather than something the top-level ``ragsage``
namespace imports: ``import ragsage`` and ``import ragsage.parsing`` must stay
free of that stack (ADR-0001's dependency invariant, enforced by
``tests/test_dependency_guard.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from pydantic import SecretStr

from ragsage.providers.config import ProviderConfig

# The provider SDKs reject an empty API key at *construction*. A caller may
# legitimately have none (dev, CI, a harness that swaps fakes in at the ports),
# so an unset key becomes this placeholder: the clients construct and the process
# boots, while any real provider call fails with an auth error at request time.
_UNSET_KEY = "unset"


@dataclass(frozen=True)
class ProviderClients:
    """The LangChain clients, one per model role, built once as singletons.

    Held together so a composition root can construct them in one call and hand
    the whole bundle to the adapters that bind roles around them. Generation and
    vision are distinct :class:`ChatOpenAI` instances because they differ in
    whether they stream and how many tokens they may emit; contextualize and
    rewrite share the cheap tier but cap output differently.
    """

    embeddings: VoyageAIEmbeddings
    rerank: VoyageAIRerank
    generation: ChatOpenAI
    contextualize: ChatOpenAI
    rewrite: ChatOpenAI
    vision: ChatOpenAI


def build_provider_clients(config: ProviderConfig) -> ProviderClients:
    """Construct the per-role provider clients from config (opens no socket).

    Every OpenAI role shares one key and one timeout; likewise every Voyage role.
    The rerank client is built with ``top_k=None`` so it scores and returns *all*
    candidates — :class:`~ragsage.providers.voyage.VoyageReranker` honours the
    per-call ``top_k`` itself, since the ``Reranker`` port passes it per request
    rather than fixing it per deployment.
    """
    openai_key = SecretStr(config.openai_api_key or _UNSET_KEY)
    voyage_key = SecretStr(config.voyage_api_key or _UNSET_KEY)

    def chat(model: str, *, max_tokens: int | None = None, streaming: bool = False) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            api_key=openai_key,
            timeout=config.timeout_seconds,
            max_completion_tokens=max_tokens,
            streaming=streaming,
        )

    return ProviderClients(
        embeddings=VoyageAIEmbeddings(
            model=config.embedding_model,
            api_key=voyage_key,
        ),
        rerank=VoyageAIRerank(
            model=config.rerank_model,
            api_key=voyage_key,
            top_k=None,
        ),
        generation=chat(config.generation_model, streaming=True),
        contextualize=chat(config.contextualize_model, max_tokens=config.context_max_tokens),
        rewrite=chat(config.contextualize_model, max_tokens=config.rewrite_max_tokens),
        vision=chat(config.vision_model, max_tokens=config.vision_max_tokens),
    )
