"""A SQLAlchemy Core description of :data:`~ragsage.storage.schema.TABLE`.

The DDL in :mod:`ragsage.storage.schema` is the source of truth — it is
hand-written because RLS policies, roles, a ``FORCE ROW LEVEL SECURITY``, an HNSW
index and a generated column are not all expressible through SQLAlchemy's DDL
layer, and the half that is expressible is not worth splitting from the half that
is not.

This module exists so the *stores* can build queries as expressions rather than
f-strings: ``embedding.cosine_distance(...)``, ``lexical.op("@@")(...)``, a bound
``IN`` over document ids. That keeps user data in bind parameters everywhere and
keeps the read path readable.

Two descriptions of one table is a drift risk, and the risk is answered rather than
accepted: an integration test compares every column here against
``information_schema`` on a migrated database, so a column added to the DDL and
forgotten here (or vice versa) fails. ``embedding``'s width is deliberately left
unconstrained in this description — the real column is sized from config, and the
type is only used to render bind parameters.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from ragsage.storage.schema import TABLE

#: Private registry — ragsage never runs ``create_all``, so this holds exactly one
#: table and shares nothing with a consumer's own SQLAlchemy metadata.
metadata_registry = MetaData()

chunks = Table(
    TABLE,
    metadata_registry,
    Column("namespace", Text, primary_key=True, nullable=False),
    Column("chunk_ref", Text, primary_key=True, nullable=False),
    Column("rag_document_id", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("embed_text", Text, nullable=False),
    Column("page", Integer, nullable=False),
    Column("ordinal", Integer, nullable=False),
    # Unsized on purpose: the migrated column carries the configured width, and
    # this description is only used to adapt a Python list into a bind parameter.
    Column("embedding", Vector(), nullable=False),
    Column("embedding_model", Text, nullable=False),
    # JSONB with a JSON fallback keeps the type portable while still rendering as
    # jsonb on Postgres, which is the only dialect ragsage targets.
    Column("metadata", JSONB().with_variant(JSON(), "sqlite"), nullable=False, server_default="{}"),
    # Database-generated; never written from Python. Present here so the lexical
    # store can build its @@ match against it.
    Column("lexical", TSVECTOR, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

#: The columns a store writes. ``lexical`` and ``created_at`` are the database's to
#: fill, and naming the writable set once stops an INSERT from drifting from it.
WRITABLE_COLUMNS = (
    "namespace",
    "chunk_ref",
    "rag_document_id",
    "text",
    "embed_text",
    "page",
    "ordinal",
    "embedding",
    "embedding_model",
    "metadata",
)
