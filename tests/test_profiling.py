import logging

from hackathon_advisor.profiling import (
    TurnProfiler,
    configure_logging,
    messages_processed,
    next_message_index,
    resource_snapshot,
)


def _turn_events() -> list[dict]:
    return [
        {"type": "start"},
        {"type": "stage", "stage": "planning"},
        {"type": "model_progress", "tokens": 5, "max_tokens": 180},
        {"type": "model_progress", "tokens": 12, "max_tokens": 180},
        {"type": "stage", "stage": "running_tool"},
        {"type": "tool_event", "name": "save_idea"},
        {"type": "tool_event", "name": "score_idea"},
        {"type": "stage", "stage": "writing"},
        {"type": "token", "text": "hello "},
        {"type": "done"},
    ]


def test_profiler_observes_tokens_tools_and_stage_durations() -> None:
    profiler = TurnProfiler(message_index=1, compute="cpu", backend="minicpm-transformers")
    for event in _turn_events():
        profiler.observe(event)

    durations = profiler.durations()

    assert profiler.tokens == 12
    assert profiler.tool_count == 2
    assert profiler.fell_back is False
    assert set(durations) >= {"total_ms", "decode_ms", "tools_ms", "write_ms"}
    assert all(value >= 0 for value in durations.values())


def test_profiler_logs_start_and_summary() -> None:
    configure_logging()  # the advisor logger does not propagate, so capture it directly
    logger = logging.getLogger("hackathon_advisor")
    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    logger.addHandler(handler)
    try:
        profiler = TurnProfiler(message_index=7, compute="gpu", backend="rules", message_chars=42)
        profiler.log_start()
        for event in _turn_events():
            profiler.observe(event)
        profiler.log_summary()
        profiler.log_summary()  # idempotent: a second call must not log again
    finally:
        logger.removeHandler(handler)

    summaries = [message for message in messages if "turn #7" in message]
    assert any("start" in message for message in summaries)
    assert sum("done" in message for message in summaries) == 1  # log_summary is idempotent


def test_profiler_marks_fallback() -> None:
    profiler = TurnProfiler(message_index=2, compute="gpu", backend="minicpm-transformers")
    profiler.observe({"type": "fallback", "to": "cpu"})

    assert profiler.fell_back is True


def test_resource_snapshot_is_best_effort_dict() -> None:
    snapshot = resource_snapshot()

    assert isinstance(snapshot, dict)
    # rss is available on the platforms we run on; never raises regardless.
    assert "rss_mb" in snapshot


def test_message_counter_increments() -> None:
    start = messages_processed()
    first = next_message_index()
    second = next_message_index()

    assert second == first + 1
    assert messages_processed() >= start + 2
