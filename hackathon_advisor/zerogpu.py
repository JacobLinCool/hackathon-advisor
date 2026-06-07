from __future__ import annotations

import os
from collections.abc import Callable
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_GPU_DURATION_SECONDS = 60
MAX_GPU_DURATION_SECONDS = 120


def zero_gpu_enabled() -> bool:
    return os.environ.get("ADVISOR_ZERO_GPU", "").strip().lower() in TRUE_VALUES


def zero_gpu_duration_seconds() -> int:
    raw = os.environ.get("ADVISOR_ZERO_GPU_DURATION", "").strip()
    if not raw:
        return DEFAULT_GPU_DURATION_SECONDS
    duration = int(raw)
    if duration <= 0:
        raise RuntimeError("ADVISOR_ZERO_GPU_DURATION must be a positive integer.")
    if duration > MAX_GPU_DURATION_SECONDS:
        raise RuntimeError(f"ADVISOR_ZERO_GPU_DURATION must be at most {MAX_GPU_DURATION_SECONDS} seconds.")
    return duration


def gpu_task(function: Callable[P, R]) -> Callable[P, R]:
    if not zero_gpu_enabled():
        return function
    try:
        import spaces
    except ImportError as error:
        raise RuntimeError(
            "ADVISOR_ZERO_GPU=1 requires the Hugging Face `spaces` package. "
            "Install runtime requirements before enabling ZeroGPU."
        ) from error
    return spaces.GPU(duration=zero_gpu_duration_seconds())(function)


QUOTA_ERROR_HINTS = ("quota", "gpu task aborted", "no gpu", "exceeded", "gpu is not available")


def is_gpu_quota_error(error: BaseException) -> bool:
    """Heuristically detect a ZeroGPU allocation/quota failure so the caller can fall back to
    a CPU run. ZeroGPU raises before the wrapped function body executes, so this is checked
    against the exception that surfaces from the first pull of the GPU generator."""
    name = type(error).__name__.lower()
    if "quota" in name or "gpu" in name:
        return True
    message = str(error).lower()
    return any(hint in message for hint in QUOTA_ERROR_HINTS)
