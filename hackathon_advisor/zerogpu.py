from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from hackathon_advisor.config import bool_env, int_env

P = ParamSpec("P")
R = TypeVar("R")


DEFAULT_GPU_DURATION_SECONDS = 60
MAX_GPU_DURATION_SECONDS = 120


def zero_gpu_enabled() -> bool:
    return bool_env("ADVISOR_ZERO_GPU")


def gpu_device() -> str:
    """torch device for the GPU path: 'cuda' under ZeroGPU, else 'local' (auto-resolved at load)."""
    return "cuda" if zero_gpu_enabled() else "local"


def zero_gpu_duration_seconds() -> int:
    return int_env(
        "ADVISOR_ZERO_GPU_DURATION",
        DEFAULT_GPU_DURATION_SECONDS,
        minimum=1,
        maximum=MAX_GPU_DURATION_SECONDS,
    )


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
