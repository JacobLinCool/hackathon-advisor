"""Lightweight logging and per-turn profiling for the advisor runtime.

The numbers here are debug/operations signal only — they are written to logs, never to the
UI. Stage timings are measured by *observing the turn event stream from the main process*, so
they stay correct even when the model itself runs inside a ZeroGPU fork (where a module-global
counter would reset on every call).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import platform
import sys
import threading
import time
from typing import Any

logger = logging.getLogger("hackathon_advisor")

_counter_lock = threading.Lock()
_messages_processed = 0


def configure_logging() -> None:
    """Attach a stream handler once, honoring ADVISOR_LOG_LEVEL (default INFO)."""
    level_name = os.environ.get("ADVISOR_LOG_LEVEL", "INFO").strip().upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def next_message_index() -> int:
    """Increment and return the lifetime count of processed advisor messages (main process)."""
    global _messages_processed
    with _counter_lock:
        _messages_processed += 1
        return _messages_processed


def messages_processed() -> int:
    return _messages_processed


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


def resource_snapshot() -> dict[str, Any]:
    """Best-effort process resource usage via the stdlib plus torch device memory if torch is
    already imported. Returns whatever could be sampled; never raises."""
    snapshot: dict[str, Any] = {}
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
        snapshot["rss_mb"] = round(usage.ru_maxrss / divisor, 1)
        snapshot["cpu_user_s"] = round(usage.ru_utime, 3)
        snapshot["cpu_sys_s"] = round(usage.ru_stime, 3)
    except Exception:  # pragma: no cover - platform dependent
        pass
    snapshot.update(_torch_memory_snapshot())
    return snapshot


def _torch_memory_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    torch = sys.modules.get("torch")  # do not import torch just to profile
    if torch is None:
        return out
    try:
        if torch.cuda.is_available():
            out["cuda_alloc_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            out["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:  # pragma: no cover - device dependent
        pass
    try:
        mps = getattr(torch, "mps", None)
        current = getattr(mps, "current_allocated_memory", None)
        if current is not None:
            out["mps_alloc_mb"] = round(current() / 1e6, 1)
    except Exception:  # pragma: no cover - device dependent
        pass
    return out


@dataclass
class TurnProfiler:
    """Times a single advisor turn by observing its event stream. Drive it by calling
    ``observe(event)`` for every emitted event dict, then ``log_summary()`` when the turn
    ends (in a finally block, so partial turns still get logged)."""

    message_index: int
    compute: str
    backend: str
    device: str = ""
    message_chars: int = 0
    started: float = field(default_factory=time.perf_counter)
    stage_at: dict[str, float] = field(default_factory=dict)
    ended: float | None = None
    tokens: int = 0
    tool_count: int = 0
    fell_back: bool = False
    logged: bool = False

    def log_start(self) -> None:
        logger.info(
            "turn #%d start | compute=%s backend=%s message_chars=%d",
            self.message_index,
            self.compute,
            self.backend,
            self.message_chars,
        )

    def observe(self, event: dict[str, Any]) -> None:
        now = time.perf_counter()
        event_type = event.get("type")
        if event_type == "stage":
            self.stage_at.setdefault(str(event.get("stage")), now)
        elif event_type == "model_progress":
            self.tokens = max(self.tokens, int(event.get("tokens") or 0))
        elif event_type == "tool_event":
            self.tool_count += 1
        elif event_type == "fallback":
            self.fell_back = True
        elif event_type == "done":
            self.ended = now

    def durations(self) -> dict[str, float]:
        end = self.ended if self.ended is not None else time.perf_counter()
        out: dict[str, float] = {"total_ms": _ms(end - self.started)}
        planning = self.stage_at.get("planning")
        running = self.stage_at.get("running_tool")
        writing = self.stage_at.get("writing")
        if planning is not None and running is not None:
            out["decode_ms"] = _ms(running - planning)
        if running is not None and writing is not None:
            out["tools_ms"] = _ms(writing - running)
        if writing is not None:
            out["write_ms"] = _ms(end - writing)
        return out

    def log_summary(self, error: BaseException | None = None) -> None:
        if self.logged:
            return
        self.logged = True
        durations = self.durations()
        timing = " ".join(f"{key}={value}" for key, value in durations.items())
        resources = " ".join(f"{key}={value}" for key, value in resource_snapshot().items())
        status = "error" if error is not None else "done"
        message = (
            f"turn #{self.message_index} {status} | {timing} | "
            f"tokens={self.tokens} tools={self.tool_count} compute={self.compute} "
            f"device={self.device or '?'} backend={self.backend} fallback={self.fell_back} | {resources}"
        )
        if error is not None:
            logger.warning("%s | exception=%r", message, error)
        else:
            logger.info(message)
