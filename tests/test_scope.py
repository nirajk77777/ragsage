"""Tests for the opaque retrieval Scope — ragsage's public value object.

These exercise the boundary the whole library rests on: a namespace plus
immutable metadata filters, with no knowledge of tenants or auth.
"""

from dataclasses import FrozenInstanceError

import pytest

from ragsage import Scope, __version__


def test_version_is_exposed() -> None:
    assert __version__ == "0.1.0"


def test_scope_defaults_to_empty_filters() -> None:
    scope = Scope(namespace="tenant-abc")
    assert scope.namespace == "tenant-abc"
    assert dict(scope.filters) == {}


def test_scope_rejects_empty_namespace() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Scope(namespace="   ")


def test_filters_are_immutable() -> None:
    scope = Scope(namespace="t", filters={"doc_id": [1, 2]})
    with pytest.raises(TypeError):
        scope.filters["doc_id"] = [3]  # type: ignore[index]


def test_with_filters_narrows_without_mutating_original() -> None:
    base = Scope(namespace="t", filters={"doc_id": [1]})
    narrowed = base.with_filters(lang="en")

    assert dict(base.filters) == {"doc_id": [1]}
    assert dict(narrowed.filters) == {"doc_id": [1], "lang": "en"}
    assert narrowed.namespace == base.namespace


def test_scope_is_frozen() -> None:
    scope = Scope(namespace="t")
    with pytest.raises(FrozenInstanceError):
        scope.namespace = "other"  # type: ignore[misc]
