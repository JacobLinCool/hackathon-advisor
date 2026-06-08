"""Central configuration accessors.

Every accessor reads ``os.environ`` (or an explicit mapping) **live** on each call,
so lazily-built runtimes and tests that monkeypatch the environment always observe
the current value — there is no import-time snapshot. Stdlib-only, so this module is
safe to import from any runtime, embedding subprocess, or Modal container.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


class ConfigError(RuntimeError, ValueError):
    """Invalid configuration value.

    Subclasses both ``RuntimeError`` and ``ValueError`` so existing
    ``except RuntimeError`` handlers and ``pytest.raises`` checks keep working
    regardless of which base they expect.
    """


def _source(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def str_env(name: str, default: str = "", *, env: Mapping[str, str] | None = None) -> str:
    """Raw environment string, or ``default`` when unset."""
    return _source(env).get(name, default)


def bool_env(name: str, default: bool = False, *, env: Mapping[str, str] | None = None) -> bool:
    """Boolean flag. Empty or unrecognised values fall back to ``default``."""
    raw = _source(env).get(name, "").strip().lower()
    if not raw:
        return default
    if raw in TRUE_VALUES:
        return True
    if raw in FALSE_VALUES:
        return False
    return default


def tri_state_env(name: str, *, env: Mapping[str, str] | None = None) -> bool | None:
    """``True``/``False`` for recognised boolean strings, ``None`` when unset/unrecognised."""
    raw = _source(env).get(name, "").strip().lower()
    if raw in TRUE_VALUES:
        return True
    if raw in FALSE_VALUES:
        return False
    return None


def int_env(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Integer with optional bounds. Empty falls back to ``default``; out-of-range raises ConfigError."""
    raw = _source(env).get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} {_below_message(minimum)}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be at most {maximum}.")
    return value


def optional_int_env(
    name: str,
    *,
    minimum: int = 1,
    env: Mapping[str, str] | None = None,
) -> int | None:
    """Integer or ``None`` when unset. Values below ``minimum`` raise ConfigError."""
    raw = _source(env).get(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < minimum:
        raise ConfigError(f"{name} {_below_message(minimum)}")
    return value


def first_nonempty_env(*names: str, default: str = "", env: Mapping[str, str] | None = None) -> str:
    """First non-empty (stripped) value among ``names``, else ``default``."""
    source = _source(env)
    for name in names:
        value = source.get(name, "").strip()
        if value:
            return value
    return default


def _below_message(minimum: int) -> str:
    if minimum == 1:
        return "must be a positive integer."
    if minimum == 0:
        return "must be a non-negative integer."
    return f"must be at least {minimum}."
