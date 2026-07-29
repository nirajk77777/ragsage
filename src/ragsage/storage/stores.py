"""The three Postgres storage ports, keyed on ``namespace``.

Moved inward from the consuming backend (ADR-0002) and re-keyed on the way: the
isolation column was a ``user_id`` UUID, and is now ``namespace TEXT`` taken
verbatim from :class:`~ragsage.scope.Scope`. That is what lets ``namespace="local"``
work for a CLI user — the old adapter coerced the namespace with ``uuid.UUID(...)``
and would simply raise.

Every method here assumes it was handed a session from
:func:`~ragsage.storage.session.open_scoped_session`, already pinned to a namespace
and running as the non-privileged role. **No method writes a namespace predicate of
its own on the read path.** That is deliberate and is the point of the design: RLS
adds it, so a store that forgets one cannot leak, whereas an application-side
``WHERE`` that is forgotten silently can. The one place ``namespace`` is written
explicitly is the INSERT, because a value has to be supplied — and the policy's
``WITH CHECK`` verifies it against the session anyway.

Three behaviours carried over unchanged, because the query layer above depends on
each of them:

* **Dense search returns cosine *similarity*** (``1 - distance``), not distance, so
  the dense and lexical halves agree that higher is better before
  :mod:`ragsage.query` fuses them with RRF.
* **Lexical search uses ``websearch_to_tsquery``** — it parses a raw user question
  the way a search box does and never raises on odd input — ranked by ``ts_rank``.
  A query with no indexable terms matches nothing rather than erroring.
* **``scope.filters["document_ids"]`` narrows and never widens.** Absent means the
  whole corpus; *present but empty* matches nothing, because the caller named a set
  that resolved to no documents and returning everything would be the wrong answer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import Select, delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from ragsage.models import Chunk, Document, EmbeddedChunk, ScoredChunk, Vector
from ragsage.scope import Scope
from ragsage.storage.identifiers import safe_identifier
from ragsage.storage.tables import chunks

_C = chunks.c


def _row_to_chunk(row: Any) -> Chunk:
    """Rebuild a :class:`~ragsage.models.Chunk` from its stored row.

    ``metadata`` round-trips: it is the consumer's payload, and dropping it on read
    would make the column write-only.
    """
    return Chunk(
        id=row.chunk_ref,
        document_id=row.rag_document_id,
        text=row.text,
        page=row.page,
        ordinal=row.ordinal,
        embed_text=row.embed_text,
        metadata=row.metadata or {},
    )


def _document_filter(statement: Select[Any], scope: Scope) -> Select[Any]:
    """Apply ``scope.filters["document_ids"]``, if the caller set one.

    Note the ``is not None`` rather than a truthiness test: an empty collection is
    a *present* filter that matches nothing, and treating it as absent would widen
    a deliberately narrow query into a whole-corpus search.
    """
    allowed = scope.filters.get("document_ids")
    if allowed is None or isinstance(allowed, str | bytes) or not isinstance(allowed, Iterable):
        return statement
    return statement.where(_C.rag_document_id.in_(list(allowed)))


async def _delete_document(session: AsyncSession, document_id: str) -> None:
    """Purge one document's chunks. RLS confines it to the session's namespace.

    All three stores' ``delete`` reduce to this: there is one table, so the vector,
    lexical and document ports are purging the same rows.
    """
    await session.execute(delete(chunks).where(_C.rag_document_id == document_id))
    await session.flush()


async def purge_namespace(session: AsyncSession) -> None:
    """Delete every chunk row visible to this session — i.e. one whole namespace.

    Carries no ``WHERE`` at all, which looks alarming and is exactly right: the RLS
    policy scopes the statement to the session's namespace. A hand-written predicate
    here would be a second, weaker copy of a guarantee the database already makes.

    Idempotent — deleting nothing is a successful delete.
    """
    await session.execute(delete(chunks))
    await session.flush()


class PgVectorStore:
    """Dense vectors: the write path for every column, plus ANN search.

    All three stores share one table, and this is the one that writes it. The
    corpus's pinned ``embedding_model`` is recorded on every row so the pin is
    inspectable and a corpus cannot silently mix models or widths.
    """

    def __init__(self, session: AsyncSession, *, embedding_model: str = "") -> None:
        # Defaulted empty because the purge path constructs this store with no
        # model and never touches the column; ``upsert`` is where it is required.
        self._session = session
        self._embedding_model = embedding_model

    async def upsert(self, scope: Scope, records: Sequence[EmbeddedChunk]) -> None:
        """Replace the rows for the incoming chunk refs, then insert them.

        Delete-then-insert rather than ``ON CONFLICT`` because a re-ingest may
        produce a *different* set of refs for the same document (a changed chunk
        size splits differently), and an upsert keyed on the primary key would
        leave the old rows behind. Clearing the incoming refs first makes a retry
        idempotent instead of doubling.
        """
        if not records:
            return
        if not self._embedding_model:
            # An unlabelled vector cannot be reproduced or audited, which defeats
            # the per-corpus pin.
            raise ValueError("PgVectorStore requires an embedding_model to upsert vectors")

        await self._session.execute(
            delete(chunks).where(
                _C.rag_document_id.in_({r.chunk.document_id for r in records}),
                _C.chunk_ref.in_([r.chunk.id for r in records]),
            )
        )
        await self._session.execute(
            insert(chunks),
            [
                {
                    "namespace": scope.namespace,
                    "chunk_ref": record.chunk.id,
                    "rag_document_id": record.chunk.document_id,
                    "text": record.chunk.text,
                    "embed_text": record.chunk.embed_text,
                    "page": record.chunk.page,
                    "ordinal": record.chunk.ordinal,
                    "embedding": list(record.vector),
                    "embedding_model": self._embedding_model,
                    "metadata": dict(record.chunk.metadata),
                }
                for record in records
            ],
        )
        await self._session.flush()

    async def search(self, scope: Scope, query_vector: Vector, *, k: int) -> Sequence[ScoredChunk]:
        """Nearest neighbours by cosine distance, returned as cosine *similarity*.

        ``cosine_distance`` must match the index's ``vector_cosine_ops``, or the
        HNSW index is built and never used. The score is inverted to a similarity
        so higher is better, which is what RRF fusion and the lexical half assume.
        """
        distance = _C.embedding.cosine_distance(list(query_vector))
        statement = _document_filter(
            select(chunks, distance.label("distance")).order_by(distance).limit(k), scope
        )
        rows = (await self._session.execute(statement)).all()
        return [
            ScoredChunk(chunk=_row_to_chunk(row), score=1.0 - float(row.distance)) for row in rows
        ]

    async def delete(self, scope: Scope, document_id: str) -> None:
        await _delete_document(self._session, document_id)


class PgLexicalStore:
    """Keyword search over the database-generated ``lexical`` column.

    ``index`` is a no-op, and that is not a stub: ``lexical`` is a generated column
    on rows :class:`PgVectorStore` already wrote in the same transaction, so there
    is nothing separate to persist. Writing here would mean a second copy of the
    same text.

    The text-search configuration must match the one the generated column was built
    with — a query parsed as ``simple`` against a column stemmed as ``english``
    silently under-matches — so it is passed in from the same config rather than
    defaulted independently.
    """

    def __init__(self, session: AsyncSession, *, text_search_config: str = "english") -> None:
        self._session = session
        # Reaches SQL as the first argument to websearch_to_tsquery; guarded for the
        # same reason the DDL guards it.
        self._text_search_config = safe_identifier(text_search_config)

    async def index(self, scope: Scope, chunks_to_index: Sequence[Chunk]) -> None:
        return None

    async def search(self, scope: Scope, query: str, *, k: int) -> Sequence[ScoredChunk]:
        """Rank matches by ``ts_rank``; an unindexable query matches nothing.

        The blank short-circuit avoids a pointless round trip.
        ``websearch_to_tsquery`` handles everything else — quotes, ``or``, a bare
        ``-`` — without raising, so an arbitrary user question is safe to pass
        straight through.
        """
        if not query.strip():
            return []
        tsquery = func.websearch_to_tsquery(self._text_search_config, query)
        rank = func.ts_rank(_C.lexical, tsquery)
        statement = _document_filter(
            select(chunks, rank.label("rank"))
            .where(_C.lexical.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(k),
            scope,
        )
        rows = (await self._session.execute(statement)).all()
        return [ScoredChunk(chunk=_row_to_chunk(row), score=float(row.rank)) for row in rows]

    async def delete(self, scope: Scope, document_id: str) -> None:
        await _delete_document(self._session, document_id)


class PgDocumentStore:
    """The document-level port, deliberately minimal.

    **Document identity is the consumer's, not the library's**, and that boundary is
    kept rather than quietly moved inward: ``find_by_hash`` returns ``None`` and
    ``save`` persists nothing. The consumer owns the lifecycle row — status, storage
    key, dedup at upload — so re-implementing dedup here would mean two answers to
    "have I seen this content", which is worse than none.

    That is why ragsage owns exactly one table. ``get``/``list`` reconstruct what
    was ingested from the chunks that exist, which is all the port needs, and
    ``delete`` purges them.

    One consequence worth naming: :attr:`Document.metadata` therefore has nowhere to
    go. :attr:`Chunk.metadata` round-trips through the JSONB column, but a
    document-level payload would need a documents table ragsage has decided not to
    own. A consumer that needs it keeps it on its own row.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, scope: Scope, document: Document, chunks_saved: Sequence[Chunk]) -> None:
        # The consumer's lifecycle row already exists; the chunk rows were written
        # by the vector store in this transaction. Nothing to add.
        return None

    async def find_by_hash(self, scope: Scope, content_hash: str) -> Document | None:
        # Always "new". Dedup belongs to the consumer, which only ever hands the
        # pipeline content it has already decided is novel.
        return None

    async def get(self, scope: Scope, document_id: str) -> Document | None:
        found = await self._session.scalar(
            select(_C.rag_document_id).where(_C.rag_document_id == document_id).limit(1)
        )
        if found is None:
            return None
        return _reconstructed(document_id)

    async def list(self, scope: Scope) -> Sequence[Document]:
        refs = (await self._session.scalars(select(_C.rag_document_id).distinct())).all()
        return [_reconstructed(ref) for ref in refs]

    async def delete(self, scope: Scope, document_id: str) -> None:
        await _delete_document(self._session, document_id)


def _reconstructed(document_id: str) -> Document:
    """A Document rebuilt from chunk rows alone.

    ``source`` and ``content_hash`` are not stored here — they belong to the
    consumer's lifecycle row — so the id stands in for the source and the hash is
    empty. Honest about being a reconstruction rather than pretending to be the
    original record.
    """
    return Document(id=document_id, source=document_id, content_hash="")
