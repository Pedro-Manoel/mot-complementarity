"""Canonical serialisation, content-addressed identities and specification errors.

Every identity (configuration, detection cache, tracking cell, evaluation) is the SHA-256 of the canonical JSON of
the content that determines it.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

SCHEMA_VERSION = 1

_PENDING_MARKERS = frozenset({"tbd", "todo", "pending", "?", ""})


class SpecError(ValueError):
    """Invalid specification, raised before any execution starts."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def stable_id(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def is_unresolved(value: Any) -> bool:
    """True for `None`, NaN and pending markers such as "TBD"."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in _PENDING_MARKERS


def require_fields(d: dict, fields: Iterable[str], where: str) -> None:
    missing = [c for c in fields if c not in d or is_unresolved(d[c])]
    if missing:
        raise SpecError(f"{where}: missing or unresolved fields: {', '.join(missing)}")
