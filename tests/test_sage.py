"""``RagSage``: that it assembles the right thing, and that it stayed optional.

The assembler's risk is not that it fails to work — it is that it quietly becomes
the *only* way to use the library, and the ports stop being a real seam. So most of
what is asserted here is negative: the fakes still satisfy every port, the CLI still
runs with no database, and both façades are still constructible by hand.

The end-to-end tests need a real Postgres and are marked ``integration``. They wire
:class:`~ragsage.fakes.FakeEngineKit`'s *model* adapters into the real Postgres
stores — the same trick the consumer's Seam-2 harness uses — so the full
ingest-then-query loop is exercised over real SQL without a network call.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from conftest import TEST_EMBEDDING_DIM
from ragsage import IngestionPipeline, QueryEngine
from ragsage.config import IngestionConfig, QueryOptions
from ragsage.fakes import FakeEmbedder, FakeEngineKit
from ragsage.models import Answer, Outcome, RawSource
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
from ragsage.providers import ProviderConfig
from ragsage.sage import ComputeKit, RagSage, RagSageConfig, StoreKit
from ragsage.scope import Scope
from ragsage.storage import Database, PostgresConfig

_DSN = "postgresql://rag:rag@localhost:1/unused"


def _config(**postgres_overrides: object) -> RagSageConfig:
    return RagSageConfig(
        postgres=PostgresConfig(dsn=_DSN, **postgres_overrides),  # type: ignore[arg-type]
        providers=ProviderConfig(openai_api_key="", voyage_api_key=""),
    )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_the_config_is_frozen() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.postgres = PostgresConfig(dsn=_DSN)  # type: ignore[misc]


def test_the_config_reads_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every value comes from the caller — ADR-0002 rules out ambient defaults."""
    for name in (
        "OPENAI_API_KEY",
        "VOYAGE_API_KEY",
        "DATABASE_URL",
        "RAGSAGE_DSN",
        "TEST_DATABASE_URL",
    ):
        monkeypatch.setenv(name, "from-the-environment")

    config = _config()

    assert config.providers.openai_api_key == ""
    assert config.providers.voyage_api_key == ""
    assert "from-the-environment" not in config.postgres.dsn


def test_an_empty_embedding_model_is_refused() -> None:
    with pytest.raises(ValueError, match="embedding_model"):
        RagSageConfig(
            postgres=PostgresConfig(dsn=_DSN),
            providers=ProviderConfig(openai_api_key="", voyage_api_key="", embedding_model="  "),
        )


def test_policy_defaults_carry_onto_the_sage() -> None:
    config = replace(
        _config(),
        ingestion=IngestionConfig(chunk_size=128),
        query=QueryOptions(retrieve_k=7, context_k=2),
    )
    sage = RagSage.from_config(config, compute=_fake_compute(), stores=_fake_stores())

    assert sage._ingestion.chunk_size == 128
    assert sage._query.retrieve_k == 7
    assert sage._query.context_k == 2


# --------------------------------------------------------------------------- #
# What from_config builds, and that all of it is replaceable
# --------------------------------------------------------------------------- #


def test_from_config_builds_every_port() -> None:
    """No network: the provider clients construct lazily and open no socket."""
    kit = ComputeKit.from_config(_config())

    assert isinstance(kit.parser, DocumentParser)
    assert isinstance(kit.classifier, PageClassifier)
    assert isinstance(kit.chunker, Chunker)
    assert isinstance(kit.contextualizer, Contextualizer)
    assert isinstance(kit.embedder, Embedder)
    assert isinstance(kit.reranker, Reranker)
    assert isinstance(kit.llm, LLMClient)
    assert isinstance(kit.rewriter, QueryRewriter)
    assert isinstance(kit.cache, Cache)


def test_the_default_contextualizer_needs_no_model_call() -> None:
    """A default that costs a model call per chunk would be a surprising default."""
    from ragsage.contextualizing import HeadingWindowContextualizer

    assert isinstance(ComputeKit.from_config(_config()).contextualizer, HeadingWindowContextualizer)


def test_the_model_contextualizer_is_one_call_away() -> None:
    from ragsage.providers import OpenAIContextualizer

    kit = ComputeKit.with_model_contextualizer(_config())

    assert isinstance(kit.contextualizer, OpenAIContextualizer)
    # …and nothing else changed.
    assert isinstance(kit.embedder, Embedder)


def test_the_default_store_factories_build_the_postgres_stores() -> None:
    from ragsage.storage import PgDocumentStore, PgLexicalStore, PgVectorStore

    stores = StoreKit.from_config(_config())
    session = object()  # never touched: constructing a store opens nothing

    assert isinstance(stores.vector_store(session), PgVectorStore)  # type: ignore[arg-type]
    assert isinstance(stores.lexical_store(session), PgLexicalStore)  # type: ignore[arg-type]
    assert isinstance(stores.document_store(session), PgDocumentStore)  # type: ignore[arg-type]


def test_any_single_adapter_can_be_replaced() -> None:
    """The overridability claim, via dataclasses.replace rather than a wide ctor."""
    kit = FakeEngineKit()
    overridden = replace(ComputeKit.from_config(_config()), embedder=kit.embedder)

    assert overridden.embedder is kit.embedder
    # Replacing one leaves the rest as built.
    assert isinstance(overridden.reranker, Reranker)


def test_the_whole_engine_can_run_on_injected_adapters() -> None:
    sage = RagSage.from_config(_config(), compute=_fake_compute(), stores=_fake_stores())

    pipeline = sage.pipeline_for(object())  # type: ignore[arg-type]
    engine = sage.engine_for(object())  # type: ignore[arg-type]

    assert isinstance(pipeline, IngestionPipeline)
    assert isinstance(engine, QueryEngine)


def test_a_caller_can_share_its_own_pool() -> None:
    database = Database(PostgresConfig(dsn=_DSN))
    sage = RagSage.from_config(
        _config(), database=database, compute=_fake_compute(), stores=_fake_stores()
    )

    assert sage._database is database


# --------------------------------------------------------------------------- #
# The façade must not become the only way in
# --------------------------------------------------------------------------- #


def test_both_facades_remain_constructible_by_hand(kit: FakeEngineKit) -> None:
    """No hidden state: the ports are still the contract."""
    pipeline = IngestionPipeline(
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
    engine = QueryEngine(
        embedder=kit.embedder,
        vector_store=kit.vector_store,
        lexical_store=kit.lexical_store,
        reranker=kit.reranker,
        llm=kit.llm,
        tracer=kit.tracer,
    )

    assert isinstance(pipeline, IngestionPipeline)
    assert isinstance(engine, QueryEngine)


def test_the_fakes_still_satisfy_every_port(kit: FakeEngineKit) -> None:
    assert isinstance(kit.parser, DocumentParser)
    assert isinstance(kit.classifier, PageClassifier)
    assert isinstance(kit.chunker, Chunker)
    assert isinstance(kit.contextualizer, Contextualizer)
    assert isinstance(kit.embedder, Embedder)
    assert isinstance(kit.reranker, Reranker)
    assert isinstance(kit.llm, LLMClient)
    assert isinstance(kit.vector_store, VectorStore)
    assert isinstance(kit.lexical_store, LexicalStore)
    assert isinstance(kit.document_store, DocumentStore)
    assert isinstance(kit.cache, Cache)
    assert isinstance(kit.tracer, Tracer)


def test_importing_ragsage_does_not_pull_the_assembler() -> None:
    """The façade must stay opt-in, or `import ragsage` inherits the whole stack.

    Checked in a clean subprocess for the same reason the dependency guard is.
    """
    probe = (
        "import ragsage, sys, json; "
        "print(json.dumps('ragsage.sage' in sys.modules "
        "or 'sqlalchemy' in sys.modules or 'voyageai' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "false", (
        "`import ragsage` pulled in the assembler or its dependencies; "
        "ragsage.sage must stay explicitly imported"
    )


def test_the_cli_still_runs_the_whole_loop_with_no_database(tmp_path: object) -> None:
    """The standing proof that the library works without any of this module.

    Drives the installed console script end to end in a subprocess — ingest, then
    query, in two separate processes as a user would.
    """
    import pathlib

    workdir = pathlib.Path(str(tmp_path))
    (workdir / "doc.md").write_text("# Physics\n\nThe mitochondrion produces ATP.\n")
    store = workdir / "state.json"

    ingest = subprocess.run(
        [sys.executable, "-m", "ragsage.cli", "ingest", str(workdir), "--store", str(store)],
        capture_output=True,
        text=True,
    )
    query = subprocess.run(
        [
            sys.executable,
            "-m",
            "ragsage.cli",
            "query",
            "What produces ATP?",
            "--store",
            str(store),
        ],
        capture_output=True,
        text=True,
    )

    assert ingest.returncode == 0, ingest.stderr
    assert query.returncode == 0, query.stderr
    assert "ATP" in query.stdout


# --------------------------------------------------------------------------- #
# End to end, over real SQL
# --------------------------------------------------------------------------- #


def _fake_compute(*, embedding_dim: int = TEST_EMBEDDING_DIM) -> ComputeKit:
    """Model adapters from the fakes; nothing here touches a network.

    The embedder is sized to the migrated column: ``FakeEmbedder`` defaults to 256
    and the test schema is deliberately narrow, so leaving it alone would fail on
    width rather than on whatever the test is about.
    """
    kit = FakeEngineKit()
    return ComputeKit(
        parser=kit.parser,
        classifier=kit.classifier,
        chunker=kit.chunker,
        contextualizer=kit.contextualizer,
        embedder=FakeEmbedder(dim=embedding_dim),
        reranker=kit.reranker,
        llm=kit.llm,
        tracer=kit.tracer,
    )


def _fake_stores() -> StoreKit:
    kit = FakeEngineKit()
    return StoreKit(
        vector_store=lambda _session: kit.vector_store,
        lexical_store=lambda _session: kit.lexical_store,
        document_store=lambda _session: kit.document_store,
    )


def _sage_over(database: Database) -> RagSage:
    """Fake models, real Postgres stores — the consumer's Seam-2 wiring."""
    config = RagSageConfig(
        postgres=database.config,
        providers=ProviderConfig(
            openai_api_key="", voyage_api_key="", embedding_model="fake-embedder"
        ),
    )
    return RagSage.from_config(config, database=database, compute=_fake_compute())


@pytest.mark.integration
async def test_migrate_then_ingest_then_query_end_to_end(database: Database) -> None:
    sage = _sage_over(database)
    scope = Scope(namespace="acme")

    await sage.migrate()  # idempotent: the fixture already ran it
    result = await sage.ingest(
        RawSource(name="physics.md", content=b"The mitochondrion produces ATP in the cell.\n"),
        scope,
    )
    answer = await sage.query("What produces ATP?", scope)

    assert result.chunk_count > 0
    assert isinstance(answer, Answer)
    assert answer.outcome is Outcome.ANSWERED
    assert "mitochondrion" in answer.text.lower()


@pytest.mark.integration
async def test_streaming_holds_the_session_open(database: Database) -> None:
    sage = _sage_over(database)
    scope = Scope(namespace="acme")
    await sage.ingest(
        RawSource(name="physics.md", content=b"The mitochondrion produces ATP in the cell.\n"),
        scope,
    )

    events = [event async for event in sage.stream("What produces ATP?", scope)]

    assert events, "the stream must yield events, not close on an expired session"


@pytest.mark.integration
async def test_abandoning_a_stream_early_still_releases_the_connection(
    database: Database,
) -> None:
    """A consumer that stops iterating must not leak a pooled connection."""
    sage = _sage_over(database)
    scope = Scope(namespace="acme")
    await sage.ingest(RawSource(name="a.md", content=b"The mitochondrion produces ATP.\n"), scope)

    stream = sage.stream("What produces ATP?", scope)
    async for _ in stream:
        break
    await stream.aclose()

    # The pool is still usable, which it would not be if the session leaked.
    assert await sage.query("What produces ATP?", scope) is not None


@pytest.mark.integration
async def test_purge_removes_a_namespace_and_is_idempotent(database: Database) -> None:
    sage = _sage_over(database)
    acme, globex = Scope(namespace="acme"), Scope(namespace="globex")
    body = b"The mitochondrion produces ATP in the cell.\n"

    await sage.ingest(RawSource(name="a.md", content=body), acme)
    await sage.ingest(RawSource(name="b.md", content=body), globex)

    await sage.purge(acme)
    await sage.purge(acme)  # idempotent

    assert (await sage.query("What produces ATP?", acme)).outcome is Outcome.NOT_FOUND
    assert (await sage.query("What produces ATP?", globex)).outcome is Outcome.ANSWERED


@pytest.mark.integration
async def test_one_namespace_cannot_read_another_through_the_facade(database: Database) -> None:
    sage = _sage_over(database)
    await sage.ingest(
        RawSource(name="secret.md", content=b"The mitochondrion produces ATP.\n"),
        Scope(namespace="acme"),
    )

    answer = await sage.query("What produces ATP?", Scope(namespace="globex"))

    assert answer.outcome is Outcome.NOT_FOUND


@pytest.mark.integration
async def test_an_injected_embedder_of_the_wrong_width_fails_loudly(database: Database) -> None:
    """A dimension mismatch is the likeliest wiring error, and it must not pass.

    ``embedding_dim`` and the embedder are configured separately and nothing can
    cross-check them statically, so the database is the backstop.
    """
    compute = _fake_compute(embedding_dim=TEST_EMBEDDING_DIM + 1)
    config = RagSageConfig(
        postgres=database.config,
        providers=ProviderConfig(
            openai_api_key="", voyage_api_key="", embedding_model="wrong-width"
        ),
    )
    sage = RagSage.from_config(config, database=database, compute=compute)

    with pytest.raises(Exception, match=r"dimension|expected|vector"):
        await sage.ingest(
            RawSource(name="a.md", content=b"The mitochondrion produces ATP.\n"),
            Scope(namespace="acme"),
        )
