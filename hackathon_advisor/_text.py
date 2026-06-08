"""Shared, dependency-free text and timestamp helpers.

Kept stdlib-only so it is safe to import from any runtime (Modal containers,
embedding subprocesses) and from the export modules without creating cycles.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def clean(value: Any) -> str:
    """Collapse whitespace to single spaces; ``None`` becomes an empty string."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def list_of_dicts(value: Any) -> list[dict[str, Any]]:
    """Return only the ``dict`` items of ``value`` when it is a list, else ``[]``."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string at second resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
