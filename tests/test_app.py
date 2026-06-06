import json

from app import bootstrap, engine, health, index, trace_artifact


def test_health_exposes_index_metadata() -> None:
    payload = health()

    assert payload["ok"] is True
    assert payload["projects"] == len(index.projects)
    assert payload["index_algorithm"] == "tfidf-sparse-v1"
    assert len(payload["snapshot_digest"]) == 64


def test_bootstrap_exposes_index_metadata() -> None:
    payload = bootstrap()

    assert payload["index_algorithm"] == "tfidf-sparse-v1"
    assert payload["index_generated_at"]
    assert payload["snapshot_digest"]
    assert payload["top_projects"]


def test_trace_artifact_endpoint_exports_jsonl() -> None:
    state = engine.turn("A local-first archive cartographer for family photos", {}).state
    payload = trace_artifact(json.dumps(state))
    lines = [json.loads(line) for line in payload.splitlines()]

    assert lines[0]["type"] == "trace_manifest"
    assert lines[0]["turn_count"] == 1
    assert lines[1]["type"] == "agent_turn"
