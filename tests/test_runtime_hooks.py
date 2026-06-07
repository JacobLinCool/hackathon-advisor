from types import SimpleNamespace

from hackathon_advisor.runtime_hooks import _is_asyncio_invalid_fd_cleanup


def test_asyncio_invalid_fd_cleanup_hook_matches_only_event_loop_destructor() -> None:
    def event_loop_del() -> None:
        pass

    event_loop_del.__qualname__ = "BaseEventLoop.__del__"

    def other_function() -> None:
        pass

    assert _is_asyncio_invalid_fd_cleanup(
        SimpleNamespace(
            exc_type=ValueError,
            exc_value=ValueError("Invalid file descriptor: -1"),
            object=event_loop_del,
        )
    )
    assert not _is_asyncio_invalid_fd_cleanup(
        SimpleNamespace(
            exc_type=ValueError,
            exc_value=ValueError("Invalid file descriptor: -1"),
            object=other_function,
        )
    )
    assert not _is_asyncio_invalid_fd_cleanup(
        SimpleNamespace(
            exc_type=RuntimeError,
            exc_value=RuntimeError("Invalid file descriptor: -1"),
            object=event_loop_del,
        )
    )
