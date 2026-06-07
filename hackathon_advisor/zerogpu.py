from __future__ import annotations

import os
from collections.abc import Callable
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_GPU_DURATION_SECONDS = 180


def zero_gpu_enabled() -> bool:
    return os.environ.get("ADVISOR_ZERO_GPU", "").strip().lower() in TRUE_VALUES


def zero_gpu_duration_seconds() -> int:
    raw = os.environ.get("ADVISOR_ZERO_GPU_DURATION", "").strip()
    if not raw:
        return DEFAULT_GPU_DURATION_SECONDS
    duration = int(raw)
    if duration <= 0:
        raise RuntimeError("ADVISOR_ZERO_GPU_DURATION must be a positive integer.")
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
