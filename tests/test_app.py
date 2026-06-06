import json

from app import bootstrap, engine, health, index, runtime, tool_contract_check, tool_contracts, trace_artifact


def test_health_exposes_index_metadata() -> None:
    payload = health()

    assert payload["ok"] is True
    assert payload["projects"] == len(index.projects)
    assert payload["index_algorithm"] == "tfidf-sparse-v1"
    assert payload["runtime"]["backend"] == "rules"
    assert len(payload["snapshot_digest"]) == 64


def test_bootstrap_exposes_index_metadata() -> None:
    payload = bootstrap()

    assert payload["index_algorithm"] == "tfidf-sparse-v1"
    assert payload["index_generated_at"]
    assert payload["snapshot_digest"]
    assert payload["runtime"]["tool_count"] >= 8
    assert payload["top_projects"]


def test_trace_artifact_endpoint_exports_jsonl() -> None:
    state = engine.turn("A local-first archive cartographer for family photos", {}).state
    payload = trace_artifact(json.dumps(state))
    lines = [json.loads(line) for line in payload.splitlines()]

    assert lines[0]["type"] == "trace_manifest"
    assert lines[0]["turn_count"] == 1
    assert lines[1]["type"] == "agent_turn"


def test_tool_contracts_endpoint_exposes_schemas() -> None:
    payload = tool_contracts()

    assert payload["tool_count"] >= 8
    assert any(tool["function"]["name"] == "search_projects" for tool in payload["tools"])


def test_tool_contract_check_endpoint_defaults_safely() -> None:
    payload = tool_contract_check("broken", "family archive")

    assert payload["status"] == "defaulted"
    assert payload["call"]["name"] == "search_projects"


def test_runtime_endpoint_reports_planner() -> None:
    payload = runtime()

    assert payload["backend"] == "rules"
    assert payload["model_id"] == "deterministic-tool-router"
    assert payload["loaded"] is True
