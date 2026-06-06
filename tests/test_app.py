import json

from app import (
    bootstrap,
    chapter_artifact,
    engine,
    field_notes_artifact,
    health,
    index,
    lora_dataset_artifact,
    prize_ledger_endpoint,
    runtime,
    tool_contract_check,
    tool_contracts,
    trace_artifact,
)


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
    assert payload["default_targets"] == payload["target_options"][:3]
    assert "skills" in payload["profile_fields"]
    assert payload["prize_ledger"]["tiny_titan_eligible"] is True


def test_trace_artifact_endpoint_exports_jsonl() -> None:
    state = engine.turn("A local-first archive cartographer for family photos", {}).state
    payload = trace_artifact(json.dumps(state))
    lines = [json.loads(line) for line in payload.splitlines()]

    assert lines[0]["type"] == "trace_manifest"
    assert lines[0]["turn_count"] == 1
    assert lines[1]["type"] == "agent_turn"


def test_field_notes_endpoint_exports_markdown() -> None:
    state = engine.turn(
        "A local-first archive cartographer for family photos",
        {"profile": {"skills": "frontend"}, "targets": ["Field Notes"]},
    ).state
    state = engine.turn("make a build plan", state).state

    payload = field_notes_artifact(json.dumps(state))

    assert payload.startswith("# Hackathon Advisor Field Notes")
    assert "Skills: frontend" in payload
    assert "Targets: Field Notes" in payload
    assert "## Turn Trace" in payload
    assert "Record the trace and write Field Notes" in payload


def test_chapter_endpoint_exports_markdown() -> None:
    state = engine.turn("A local-first archive cartographer for family photos", {}).state
    state = engine.turn("write bolder and find whitespace", state).state

    payload = chapter_artifact(json.dumps(state))

    assert payload.startswith("# The Unwritten Almanac Chapter")
    assert "## Page 1:" in payload
    assert "## Page 2:" in payload
    assert "Closest inked pages:" in payload


def test_lora_dataset_endpoint_exports_sft_jsonl() -> None:
    state = engine.turn(
        "A local-first archive cartographer for family photos",
        {"targets": ["Well-Tuned"]},
    ).state
    state = engine.turn("make a build plan", state).state

    payload = lora_dataset_artifact(json.dumps(state))
    lines = [json.loads(line) for line in payload.splitlines()]

    assert lines[0]["type"] == "lora_sft_manifest"
    assert lines[0]["example_count"] == len(lines) - 1
    assert lines[1]["example_kind"] == "tool_call"
    assert lines[1]["base_model"] == "openbmb/MiniCPM5-1B"
    assert lines[2]["example_kind"] == "advisor_response"


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


def test_prize_ledger_endpoint_reports_submission_evidence() -> None:
    payload = prize_ledger_endpoint()

    assert payload["runtime"]["backend"] == "rules"
    assert payload["tiny_titan_eligible"] is True
    assert any(badge["name"] == "Sharing is Caring" for badge in payload["badges"])
    assert payload["training_artifacts"][0]["endpoint"] == "lora_dataset"
