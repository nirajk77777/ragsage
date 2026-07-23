"""The opaque retrieval scope handed to the engine by its caller.

``ragsage`` is deliberately tenancy-agnostic: it must work unchanged whether it
is driven single-tenant from a CLI or multi-tenant from the SaaS backend. The
backend maps ``tenant_id -> Scope``; the library only ever sees a namespace
string plus optional metadata filters (e.g. a subset of document ids). A
pricing/auth/tenancy change must never require editing this file — that is the
litmus test for the boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Scope:
    """An opaque isolation boundary for ingestion and retrieval.

    Parameters
    ----------
    namespace:
        The hard partition every store keys on. The backend passes the
        tenant id here; a CLI user might pass ``"local"``. Must be non-empty.
    filters:
        Optional metadata narrowing *within* a namespace — for example
        restricting a query to a chosen subset of documents. Never widens
        access beyond ``namespace``.
    """

    namespace: str
    filters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.namespace or not self.namespace.strip():
            raise ValueError("Scope.namespace must be a non-empty string")
        # Freeze the mapping so a Scope is genuinely immutable and hashable in
        # spirit — callers can't mutate filters out from under the engine.
        object.__setattr__(self, "filters", MappingProxyType(dict(self.filters)))

    def with_filters(self, **extra: object) -> Scope:
        """Return a copy narrowed by additional metadata filters."""
        return Scope(namespace=self.namespace, filters={**self.filters, **extra})
