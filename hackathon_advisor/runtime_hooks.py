from __future__ import annotations

import sys
from typing import Any


_HOOK_INSTALLED = False


def install_asyncio_cleanup_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    previous_hook = sys.unraisablehook

    def hook(args: Any) -> None:
        if _is_asyncio_invalid_fd_cleanup(args):
            return
        previous_hook(args)

    sys.unraisablehook = hook
    _HOOK_INSTALLED = True


def _is_asyncio_invalid_fd_cleanup(args: Any) -> bool:
    if getattr(args, "exc_type", None) is not ValueError:
        return False
    if str(getattr(args, "exc_value", "")) != "Invalid file descriptor: -1":
        return False
    owner = getattr(args, "object", None)
    return getattr(owner, "__qualname__", "") == "BaseEventLoop.__del__"
