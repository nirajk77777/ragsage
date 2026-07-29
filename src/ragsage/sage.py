"""``RagSage`` — one entry point that assembles the engine from a config object.

Before this, a consumer had to hand-wire a parser, a classifier, a chunker, a
contextualizer, an embedder, a reranker, an LLM client, a rewriter, three stores
and a cache before it could ingest a single document. That wiring is the same
every time, so it belongs in the library.

**This is an assembler, not a replacement.** The ports stay public,
:class:`~ragsage.ingestion.IngestionPipeline` and :class:`~ragsage.query.QueryEngine`
stay directly constructible, and every adapter :meth:`RagSage.from_config` builds
can be swapped. :meth:`RagSage.pipeline_for` and :meth:`RagSage.engine_for` are
public for exactly that reason — a consumer that outgrows the façade reaches past
it rather than forking it. The fakes, the CLI and the offline test suite never touch
this module, which is the standing evidence that it stayed optional.

**Not re-exported from the ``ragsage`` namespace**, deliberately: importing this
pulls SQLAlchemy, asyncpg and the provider SDKs (and through ``voyageai``, numpy
and HuggingFace ``tokenizers``). ``import ragsage`` and ``import ragsage.parsing``
must stay free of that stack — see ``tests/test_dependency_guard.py``. Reach for it
explicitly::

    from ragsage.sage import RagSage, RagSageConfig

Why the stores are built per operation rather than once: they hold a *session*, and
a session carries the namespace binding that makes Row-Level Security work. A
long-lived store would either pin one namespace forever or need its session swapped
underneath it. So each call opens a scoped session, builds the two façades around
it, and closes it — the compute adapters, which are stateless with respect to the
caller, are built once and reused.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace

from sqlalchemy.ext.asyncio import AsyncSession

from ragsage.config import IngestionConfig, QueryOptions
from ragsage.contextualizing import HeadingWindowContextualizer
from ragsage.ingestion import IngestionPipeline, IngestionResult
from ragsage.models import Answer, AnswerEvent, RawSource, Turn
from ragsage.parsing import HeuristicBackend, LayoutPageClassifier
from ragsage.ports import (
    Cache,
    Chunker,
    Contextualizer,
    DocumentParser,
    DocumentStore,
    Embedder,
    LexicalStore,
    LLMClient,
    PageClassifier,
    QueryRewriter,
    Reranker,
    Tracer,
    VectorStore,
)
from ragsage.providers import (
    OpenAIContextualizer,
    OpenAILLMClient,
    OpenAIQueryRewriter,
    ProviderConfig,
    VoyageEmbedder,
    VoyageReranker,
    build_provider_clients,
)
from ragsage.query import QueryEngine
from ragsage.scope import Scope
from ragsage.storage import (
    Database,
    PgDocumentStore,
    PgLexicalStore,
    PgVectorStore,
    PostgresConfig,
    purge_namespace,
)


class _NullCache:
    """The default cache: correct, and caches nothing.

    :class:`~ragsage.ports.Cache` is an optimisation the pipeline is correct
    without, so the honest default for a library is a no-op rather than a
    process-local dict that silently does nothing useful across workers. A
    consumer wanting real caching injects a Redis adapter.
    """

    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str) -> None:
        return None


@dataclass(frozen=True)
class RagSageConfig:
    """Everything :meth:`RagSage.from_config` needs, composed from the two halves.

    Composed rather than flattened on purpose. ``PostgresConfig`` (ticket 01) and
    ``ProviderConfig`` (ticket 04) already validate their own fields and are
    independently useful — a consumer injecting a custom embedder still wants the
    storage config. Flattening would mean two declarations of every setting and a
    second place for them to drift.

    Neither half reads the environment, here or anywhere: wiring env vars to fields
    is the consumer's job at its own composition root (ADR-0002).

    ``ingestion`` and ``query`` carry the *policy* defaults — chunk size, whether to
    contextualize, how many candidates to retrieve — so a caller can set them once
    instead of passing them to every call. Both remain overridable per call.
    """

    postgres: PostgresConfig
    providers: ProviderConfig
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    query: QueryOptions = field(default_factory=QueryOptions)

    def __post_init__(self) -> None:
        # The two halves each validate themselves; this checks the seam between
        # them. Note what it cannot check: whether ``providers.embedding_model``
        # actually emits ``postgres.embedding_dim`` dimensions. No lookup table maps
        # a model name to a width, so that invariant is enforced by Postgres at the
        # first insert — a wrong pairing fails loudly there rather than here.
        if not self.providers.embedding_model.strip():
            raise ValueError("RagSageConfig.providers.embedding_model must be set")


@dataclass(frozen=True)
class ComputeKit:
    """The adapters that do not depend on a database session.

    Stateless with respect to the caller, so one instance serves every operation.
    Frozen, so overriding one is :func:`dataclasses.replace` rather than a
    thirteen-argument constructor call::

        kit = replace(ComputeKit.from_config(config), embedder=MyEmbedder())
        sage = RagSage.from_config(config, compute=kit)
    """

    parser: DocumentParser
    classifier: PageClassifier
    chunker: Chunker
    contextualizer: Contextualizer
    embedder: Embedder
    reranker: Reranker
    llm: LLMClient
    rewriter: QueryRewriter | None = None
    cache: Cache = field(default_factory=_NullCache)
    tracer: Tracer | None = None

    @classmethod
    def from_config(cls, config: RagSageConfig) -> ComputeKit:
        """Build the default adapters: ragsage's own parser, Voyage, and OpenAI."""
        clients = build_provider_clients(config.providers)
        # One backend satisfies both the parser and the chunker port: it parses a
        # document once and hands its structure to the chunk step.
        backend = HeuristicBackend()
        return cls(
            parser=backend,
            classifier=LayoutPageClassifier(),
            chunker=backend,
            # The deterministic contextualizer, not the OpenAI one: it costs no
            # model call and needs no network, which is the better default. A
            # caller who wants the model-written version overrides this with
            # OpenAIContextualizer(clients.contextualize).
            contextualizer=HeadingWindowContextualizer(),
            embedder=VoyageEmbedder(clients.embeddings),
            reranker=VoyageReranker(clients.rerank),
            llm=OpenAILLMClient(generation=clients.generation, vision=clients.vision),
            rewriter=OpenAIQueryRewriter(clients.rewrite),
        )

    @classmethod
    def with_model_contextualizer(cls, config: RagSageConfig) -> ComputeKit:
        """The same kit, but contextualizing with OpenAI instead of headings.

        Offered because it is the one default a quality-focused consumer is most
        likely to want changed, and discovering the right client to pass otherwise
        means reading this module.
        """
        clients = build_provider_clients(config.providers)
        return replace(
            cls.from_config(config),
            contextualizer=OpenAIContextualizer(clients.contextualize),
        )


@dataclass(frozen=True)
class StoreKit:
    """How to build the three session-bound stores.

    Factories rather than instances, because a store holds a session and a session
    holds the namespace binding RLS enforces. Overriding one — a consumer with its
    own vector database — means supplying a factory that ignores the session.
    """

    vector_store: Callable[[AsyncSession], VectorStore]
    lexical_store: Callable[[AsyncSession], LexicalStore]
    document_store: Callable[[AsyncSession], DocumentStore]

    @classmethod
    def from_config(cls, config: RagSageConfig) -> StoreKit:
        embedding_model = config.providers.embedding_model
        text_search_config = config.postgres.text_search_config
        return cls(
            vector_store=lambda session: PgVectorStore(session, embedding_model=embedding_model),
            lexical_store=lambda session: PgLexicalStore(
                session, text_search_config=text_search_config
            ),
            document_store=PgDocumentStore,
        )


class RagSage:
    """The assembled engine: migrate, ingest, query, stream, purge."""

    def __init__(
        self,
        *,
        database: Database,
        compute: ComputeKit,
        stores: StoreKit,
        ingestion: IngestionConfig | None = None,
        query: QueryOptions | None = None,
    ) -> None:
        self._database = database
        self._compute = compute
        self._stores = stores
        self._ingestion = ingestion or IngestionConfig()
        self._query = query or QueryOptions()

    @classmethod
    def from_config(
        cls,
        config: RagSageConfig,
        *,
        database: Database | None = None,
        compute: ComputeKit | None = None,
        stores: StoreKit | None = None,
    ) -> RagSage:
        """Assemble the default engine, with any layer replaceable.

        Each argument defaults to the standard build. Pass ``compute`` to swap model
        adapters, ``stores`` to swap persistence, ``database`` to share a pool with
        the consumer's own.
        """
        return cls(
            database=database or Database(config.postgres),
            compute=compute or ComputeKit.from_config(config),
            stores=stores or StoreKit.from_config(config),
            ingestion=config.ingestion,
            query=config.query,
        )

    # -- composition, exposed on purpose -----------------------------------

    def pipeline_for(self, session: AsyncSession) -> IngestionPipeline:
        """The ingestion pipeline bound to one scoped session.

        Public so a consumer can drive the pipeline directly — inside its own
        transaction, say — without reimplementing this wiring.
        """
        return IngestionPipeline(
            parser=self._compute.parser,
            classifier=self._compute.classifier,
            chunker=self._compute.chunker,
            contextualizer=self._compute.contextualizer,
            embedder=self._compute.embedder,
            vector_store=self._stores.vector_store(session),
            lexical_store=self._stores.lexical_store(session),
            document_store=self._stores.document_store(session),
            llm=self._compute.llm,
            cache=self._compute.cache,
            tracer=self._compute.tracer,
        )

    def engine_for(self, session: AsyncSession) -> QueryEngine:
        """The query engine bound to one scoped session."""
        return QueryEngine(
            embedder=self._compute.embedder,
            vector_store=self._stores.vector_store(session),
            lexical_store=self._stores.lexical_store(session),
            reranker=self._compute.reranker,
            llm=self._compute.llm,
            rewriter=self._compute.rewriter,
            tracer=self._compute.tracer,
        )

    # -- lifecycle ---------------------------------------------------------

    async def migrate(self) -> None:
        """Create ragsage's table, indexes, role and RLS policy. Idempotent."""
        await self._database.migrate()

    async def dispose(self) -> None:
        """Close the connection pool. Call on shutdown."""
        await self._database.dispose()

    # -- the write path ----------------------------------------------------

    async def ingest(
        self, source: RawSource, scope: Scope, config: IngestionConfig | None = None
    ) -> IngestionResult:
        """Ingest one document into ``scope``, in a single scoped transaction."""
        async with self._database.session(scope) as session:
            return await self.pipeline_for(session).ingest(source, scope, config or self._ingestion)

    # -- the read path -----------------------------------------------------

    async def query(
        self,
        question: str,
        scope: Scope,
        options: QueryOptions | None = None,
        *,
        history: Sequence[Turn] = (),
    ) -> Answer:
        """Answer ``question`` from ``scope``'s corpus."""
        async with self._database.session(scope) as session:
            return await self.engine_for(session).query(
                question, scope, options or self._query, history=history
            )

    async def stream(
        self,
        question: str,
        scope: Scope,
        options: QueryOptions | None = None,
        *,
        history: Sequence[Turn] = (),
    ) -> AsyncIterator[AnswerEvent]:
        """Stream the answer, holding the scoped session open for the whole stream.

        The session must outlive the generator — retrieval happens as the stream is
        consumed, not before it starts — so the transaction closes when the consumer
        stops iterating. A caller that abandons the stream early still releases the
        connection, because the ``async with`` unwinds on generator close.
        """
        async with self._database.session(scope) as session:
            engine = self.engine_for(session)
            async for event in engine.stream(
                question, scope, options or self._query, history=history
            ):
                yield event

    # -- deletion ----------------------------------------------------------

    async def delete_document(self, scope: Scope, document_id: str) -> None:
        """Remove one document's chunks from ``scope``. Idempotent.

        The consumer's own delete path needs this: its lifecycle row and the chunks
        now live in different tables on different connections, so deleting the row
        no longer takes the chunks with it. All three stores address the same rows,
        so this drives them together rather than leaving the caller to guess which
        one owns the purge.
        """
        async with self._database.session(scope) as session:
            await self._stores.vector_store(session).delete(scope, document_id)
            await self._stores.lexical_store(session).delete(scope, document_id)
            await self._stores.document_store(session).delete(scope, document_id)

    async def purge(self, scope: Scope) -> None:
        """Delete every chunk in ``scope``'s namespace. Idempotent.

        Load-bearing rather than convenience. ADR-0002 dropped the
        ``chunks.user_id -> users.id ON DELETE CASCADE`` foreign key, so deleting a
        consumer's user no longer purges their chunks for free. Nothing deletes
        users today; this exists so the path is available the day account deletion
        ships, instead of being discovered missing then.
        """
        async with self._database.session(scope) as session:
            await purge_namespace(session)
