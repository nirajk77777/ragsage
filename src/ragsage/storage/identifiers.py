"""Guards for the two SQL names Postgres will not let us bind as parameters.

A role name in ``SET LOCAL ROLE`` and a setting name inside a Row-Level Security
policy are *identifiers*, and Postgres has no bind-parameter form for one: they
have to be interpolated into the statement text. Both come from trusted
configuration rather than from a request, but interpolation is a footgun
regardless of provenance, so anything that is not a plain identifier fails loudly
— at config construction, and again at the interpolation site.

This mirrors the consuming backend's ``safe_identifier`` deliberately. ADR-0002
moves the isolation posture inward into ``ragsage``; the guard that keeps that
posture honest moves with it, rather than being left behind in the consumer.
"""

from __future__ import annotations

import re

# A plain, unquoted-safe SQL identifier: leading letter or underscore, then
# letters, digits, underscores. No quotes, no dots, no whitespace — anything
# needing more than this is not a name we are willing to interpolate.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_identifier(name: str) -> str:
    """Return ``name`` if it is a plain SQL identifier, else raise ``ValueError``.

    Used for the app role, which reaches ``SET LOCAL ROLE "<role>"`` as
    interpolated text. Returning the name (rather than just asserting) keeps the
    guard at the interpolation site, where a reader can see that nothing
    unchecked reaches the SQL string.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def safe_setting_name(name: str) -> str:
    """Return ``name`` if it is a valid custom Postgres setting, else raise.

    A run-time-defined setting must be *qualified* — ``prefix.name``, exactly one
    dot — because Postgres only accepts assignments to unknown parameters when
    they carry an extension-style prefix. Requiring the prefix here means a
    typo'd isolation variable fails when the config is built, not on the first
    query against a policy that then silently matches nothing.
    """
    prefix, dot, suffix = name.partition(".")
    if not dot:
        raise ValueError(
            f"unsafe Postgres setting name: {name!r} — a custom setting must be "
            f"qualified with a prefix, e.g. 'ragsage.namespace'"
        )
    if not _IDENTIFIER_RE.match(prefix) or not _IDENTIFIER_RE.match(suffix):
        raise ValueError(f"unsafe Postgres setting name: {name!r}")
    return name
