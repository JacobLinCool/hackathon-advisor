import json
from io import BytesIO
from zipfile import ZipFile

from app import (
    artifact_png,
    bootstrap,
    chapter_artifact,
    demo_bundle,
    demo_session,
    engine,
    field_notes_artifact,
    health,
    index,
    lora_dataset_artifact,
    lora_training_kit,
    prize_ledger_endpoint,
    runtime,
    submission_packet_artifact,
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
    assert payload["default_goals"] == payload["goal_options"][:3]
    assert [goal["id"] for goal in payload["goal_profiles"]] == payload["goal_options"]
    assert payload["goal_profiles"][0]["label"] == "Local-first"
    assert "description" in payload["goal_profiles"][0]
    assert "skills" in payload["profile_fields"]
    assert "prize_ledger" not in payload
    assert all("trace" not in goal["description"].lower() for goal in payload["goal_profiles"])


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
        {"profile": {"skills": "frontend"}, "goals": ["Field Notes"]},
    ).state
    state = engine.turn("make a build plan", state).state

    payload = field_notes_artifact(json.dumps(state))

    assert payload.startswith("# Hackathon Advisor Field Notes")
    assert "Skills: frontend" in payload
    assert "Goals: Build notes" in payload
    assert "Targets: Field Notes" not in payload
    assert "## Turn Trace" in payload
    assert "Write build notes from the exact decisions" in payload


def test_chapter_endpoint_exports_markdown() -> None:
    state = engine.turn("A local-first archive cartographer for family photos", {}).state
    state = engine.turn("write bolder and find whitespace", state).state

    payload = chapter_artifact(json.dumps(state))

    assert payload.startswith("# The Unwritten Almanac Chapter")
    assert "## Page 1:" in payload
    assert "## Page 2:" in payload
    assert "Goals:" in payload
    assert "Targets:" not in payload
    assert "Closest cited pages:" in payload


def test_lora_dataset_endpoint_exports_sft_jsonl() -> None:
    state = engine.turn(
        "A local-first archive cartographer for family photos",
        {"goals": ["Well-Tuned"]},
    ).state
    state = engine.turn("make a build plan", state).state

    payload = lora_dataset_artifact(json.dumps(state))
    lines = [json.loads(line) for line in payload.splitlines()]

    assert lines[0]["type"] == "lora_sft_manifest"
    assert lines[0]["example_count"] == len(lines) - 1
    assert lines[1]["example_kind"] == "tool_call"
    assert lines[1]["base_model"] == "openbmb/MiniCPM5-1B"
    assert lines[2]["example_kind"] == "advisor_response"


def test_submission_packet_endpoint_exports_markdown() -> None:
    state = engine.turn(
        "A local-first archive cartographer for family photos",
        {"goals": ["Field Notes"]},
    ).state
    state = engine.turn("make a build plan", state).state

    payload = submission_packet_artifact(json.dumps(state))

    assert payload.startswith("# Hackathon Advisor Submission Packet")
    assert "## Demo Script" in payload
    assert "## Prize Evidence" in payload
    assert "Live Space:" in payload


def test_tool_contracts_endpoint_exposes_schemas() -> None:
    payload = tool_contracts()

    assert payload["tool_count"] >= 8
    assert any(tool["function"]["name"] == "search_projects" for tool in payload["tools"])


def test_demo_session_endpoint_returns_export_ready_state() -> None:
    payload = demo_session()

    assert payload["turn_count"] == 2
    assert payload["session"]["trace"]
    assert payload["session"]["ideas"]
    assert payload["plan"]
    assert payload["artifact"]["wood_map"]["dots"]
    assert payload["export_ready"]["submission_packet"] is True


def test_demo_bundle_endpoint_returns_zip_attachment() -> None:
    response = demo_bundle()

    assert response.media_type == "application/zip"
    assert "hackathon-advisor-demo-bundle.zip" in response.headers["content-disposition"]
    with ZipFile(BytesIO(response.body)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))

    assert "submission-packet.md" in names
    assert "lora-sft.jsonl" in names
    assert "lora-training-kit.zip" in names
    assert "archive-cartographer.png" in names
    assert manifest["turn_count"] == 2


def test_artifact_png_endpoint_returns_png_attachment() -> None:
    state = engine.turn("A local-first archive cartographer for family photos", {}).state
    response = artifact_png(state["last_artifact"])

    assert response.media_type == "image/png"
    assert 'filename="a-local-first-archive-cartographer-for-family-photos.png"' in response.headers[
        "content-disposition"
    ]
    assert response.body.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(response.body) > 10_000


def test_lora_training_kit_endpoint_returns_zip_attachment() -> None:
    response = lora_training_kit()

    assert response.media_type == "application/zip"
    assert "hackathon-advisor-lora-training-kit.zip" in response.headers["content-disposition"]
    with ZipFile(BytesIO(response.body)) as archive:
        names = set(archive.namelist())
        recipe = json.loads(archive.read("training-recipe.json"))

    assert "adapter-model-card.md" in names
    assert "train-command.txt" in names
    assert recipe["publish_status"] == "not-published"


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
    assert payload["training_artifacts"][1]["endpoint"] == "/api/lora-training-kit.zip"
