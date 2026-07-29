"""Provider credentials and per-role model choices, as data the caller owns.

The library never reads ``os.environ``. Every key, model id, and timeout arrives
in this frozen dataclass, constructed by whoever embeds ragsage — a backend
reading its own settings, a CLI reading flags, a test passing literals. That is
the rule ADR-0002 states as "configuration is a frozen dataclass passed by the
caller — never environment variables read at class-definition time": a library
that reaches for the environment is a library you cannot instantiate twice, test
without monkeypatching, or reason about from the call site.

This config is deliberately small and provider-shaped. Ticket 05 folds it into
one top-level ``RagSageConfig`` alongside the storage config; until then it
stands alone so the provider adapters have exactly one place to be configured
from.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig

#: An opaque per-call configuration threaded, untouched, into the underlying
#: LangChain call.
#:
#: ragsage does not read it, build it, or depend on what is inside it. A consumer
#: that wants its model calls attributed to a user — the backend binds Langfuse
#: callbacks this way (its ADR-0003) — constructs one and hands it to the adapter;
#: ragsage passes it straight through. That observability concern stays entirely
#: outside this library: there is no Langfuse import anywhere in ragsage, and
#: tracing proper reaches the engine through the injected ``Tracer``/``Span``
#: ports instead. The alias is LangChain's own type because these adapters *are*
#: LangChain clients, so naming it precisely costs nothing and keeps the pass-
#: through type-checked.
type CallConfig = RunnableConfig


@dataclass(frozen=True)
class ProviderConfig:
    """Keys, per-role model ids, and call limits for the Voyage/OpenAI adapters.

    The roles are split because they are priced and shaped differently, not for
    symmetry: embedding and reranking go to Voyage; generation, contextualization,
    query rewriting, and vision go to OpenAI, with contextualization and rewriting
    sharing a cheap tier. ``embedding_model`` is pinned per corpus — changing it
    invalidates every stored vector — so it is named here rather than defaulted at
    each call site.

    An empty API key is permitted and means "no calls to this provider will
    succeed". The provider SDKs reject an empty key at *construction*, so
    :func:`~ragsage.providers.clients.build_provider_clients` substitutes a
    placeholder: the clients build and the process boots (dev, CI, and any harness
    that swaps fakes in at the ports never touches a model), while a real call
    fails with an auth error at request time rather than at import time.
    """

    openai_api_key: str
    voyage_api_key: str

    embedding_model: str = "voyage-3-large"
    rerank_model: str = "rerank-2"
    contextualize_model: str = "gpt-4o-mini"
    generation_model: str = "gpt-4o"
    vision_model: str = "gpt-4o"

    timeout_seconds: float = 60.0

    # A situating context is a sentence or two, and a rewritten question a single
    # sentence; cap each cheap role's output so it cannot run away. Generation is
    # uncapped (the answer's length is the model's to decide); vision transcribes
    # a whole page, so it gets the largest budget.
    context_max_tokens: int = 128
    rewrite_max_tokens: int = 256
    vision_max_tokens: int = 4096

    def __post_init__(self) -> None:
        for field_name in (
            "embedding_model",
            "rerank_model",
            "contextualize_model",
            "generation_model",
            "vision_model",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be a non-empty model id")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for field_name in ("context_max_tokens", "rewrite_max_tokens", "vision_max_tokens"):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
