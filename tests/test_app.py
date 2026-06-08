import json
import asyncio
import time
from io import BytesIO
from zipfile import ZipFile

import app as app_module
from app import (
    agent_turn_stream,
    artifact_png,
    bootstrap,
    chapter_api,
    chapter_artifact,
    dashboard,
    dashboard_refresh_start,
    dashboard_refresh_status,
    demo_bundle,
    demo_session,
    engine,
    field_notes_api,
    field_notes_artifact,
    health,
    index,
    lora_dataset_artifact,
    lora_training_kit,
    prize_ledger_endpoint,
    runtime,
    submission_packet_artifact,
    transcribe_audio,
    tool_contract_check,
    tool_contracts,
    trace_artifact,
)
from hackathon_advisor.dashboard import build_dashboard_payload
from hackathon_advisor.data import Project, ProjectIndex


async def _read_streaming_response(response) -> str:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


class DummyUpload:
    def __init__(
        self,
        content: bytes,
        filename: str = "voice.wav",
        content_type: str = "audio/wav",
    ) -> None:
        self._content = content
        self._offset = 0
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._content):
            return b""
        if size is None or size < 0:
            size = len(self._content) - self._offset
        start = self._offset
        self._offset = min(len(self._content), self._offset + size)
        return self._content[start : self._offset]


def _reset_refresh_state(status: str = "idle") -> None:
    with app_module._refresh_lock:
        app_module._refresh_state.update(
            {
                "status": status,
                "run_id": "test-run" if status == "running" else "",
                "stage": "crawling" if status == "running" else "",
                "stage_label": "Fetching public Spaces" if status == "running" else "",
                "started_at": "",
                "finished_at": "",
                "error": "",
                "result": None,
            }
        )


def _wait_for_refresh(timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    state = dashboard_refresh_status()
    while state["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        state = dashboard_refresh_status()
    return state


def test_health_exposes_index_metadata() -> None:
    payload = health()

    assert payload["ok"] is True
    assert payload["projects"] == len(index.projects)
    assert payload["index_algorithm"] == "llama-cpp-embedding-v1"
    assert payload["runtime"]["backend"] == "rules"
    assert payload["voice"]["model_id"] == "nvidia/nemotron-speech-streaming-en-0.6b"
    assert len(payload["snapshot_digest"]) == 64


def test_bootstrap_exposes_index_metadata(monkeypatch) -> None:
    def fail_query_embedder(_: str) -> tuple[float, ...]:
        raise AssertionError("bootstrap should not load the runtime query embedder")

    monkeypatch.setattr(index, "_query_embedder", fail_query_embedder)

    payload = bootstrap()

    assert payload["index_algorithm"] == "llama-cpp-embedding-v1"
    assert payload["index_generated_at"]
    assert payload["snapshot_digest"]
    assert payload["runtime"]["tool_count"] >= 8
    assert payload["voice"]["backend"] == "nemo-asr"
    assert payload["top_projects"]
    assert payload["whitespace"]
    assert payload["default_goals"] == payload["goal_options"][:3]
    assert [goal["id"] for goal in payload["goal_profiles"]] == payload["goal_options"]
    assert payload["goal_profiles"][0]["label"] == "Local-first"
    assert "description" in payload["goal_profiles"][0]
    assert "skills" in payload["profile_fields"]
    assert "prize_ledger" not in payload
    assert all("trace" not in goal["description"].lower() for goal in payload["goal_profiles"])


def test_dashboard_endpoint_exposes_atlas_payload() -> None:
    payload = dashboard()

    assert payload["layout"]["algorithm"] == "tsne"
    assert payload["project_count"] == len(payload["points"])
    assert payload["clusters"]
    assert payload["links"]
    assert payload["quest_report"]["status"] in {"analyzed", "not_analyzed"}
    assert payload["refresh"]["status"] in {"idle", "running", "succeeded", "failed"}


def test_dashboard_refresh_requires_bucket(monkeypatch) -> None:
    _reset_refresh_state()
    monkeypatch.delenv("ADVISOR_CACHE_DIR", raising=False)

    try:
        dashboard_refresh_start()
    except Exception as error:
        assert getattr(error, "status_code", None) == 400
    else:
        raise AssertionError("dashboard refresh should require ADVISOR_CACHE_DIR")


def test_dashboard_refresh_rejects_concurrent_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ADVISOR_CACHE_DIR", str(tmp_path))
    _reset_refresh_state(status="running")

    try:
        dashboard_refresh_start()
    except Exception as error:
        assert getattr(error, "status_code", None) == 409
    else:
        raise AssertionError("concurrent dashboard refresh should fail")
    finally:
        _reset_refresh_state()


def test_dashboard_refresh_embedding_build_runs_in_subprocess(monkeypatch, tmp_path) -> None:
    project_path = tmp_path / "projects.json"
    index_path = tmp_path / "project_index.json"
    reuse_index_path = tmp_path / "reuse_project_index.json"
    project_path.write_text(
        json.dumps({"generated_at": "2026-06-08T00:00:00+00:00", "source": "test", "projects": []}),
        encoding="utf-8",
    )
    reuse_index_path.write_text(json.dumps({"documents": []}), encoding="utf-8")
    monkeypatch.setenv("ADVISOR_EMBEDDING_MODEL_REPO", "test/repo")
    monkeypatch.setenv("ADVISOR_EMBEDDING_MODEL_FILE", "model.gguf")
    monkeypatch.setenv("ADVISOR_EMBEDDING_MODEL_PATH", "/tmp/model.gguf")
    captured = {}

    def fake_run_refresh_index_command(command):
        captured["command"] = command
        index_path.write_text(json.dumps({"schema": "ok"}), encoding="utf-8")

    monkeypatch.setattr(app_module, "_run_refresh_index_command", fake_run_refresh_index_command)

    payload = app_module._build_refresh_index_payload(project_path, index_path, reuse_index_path=reuse_index_path)

    command = captured["command"]
    assert payload == {"schema": "ok"}
    assert command[1].endswith("scripts/build_project_index.py")
    assert command[command.index("--model-repo") + 1] == "test/repo"
    assert command[command.index("--model-file") + 1] == "model.gguf"
    assert command[command.index("--model-path") + 1] == "/tmp/model.gguf"
    assert command[command.index("--reuse-index") + 1] == str(reuse_index_path)
    assert command[command.index("--build-source") + 1] == "space dashboard refresh"
    assert command[command.index("--builder") + 1] == "app.py:/api/dashboard/refresh"


def test_refresh_subprocess_env_uses_cache_dir_for_hf_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ADVISOR_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)

    env = app_module._refresh_subprocess_env()

    assert env["HF_HOME"] == str(tmp_path / "huggingface")
    assert (tmp_path / "huggingface").is_dir()


def test_refresh_embedding_timeout_rejects_non_positive_env(monkeypatch) -> None:
    monkeypatch.setenv("ADVISOR_REFRESH_EMBEDDING_TIMEOUT_SECONDS", "0")

    try:
        app_module._refresh_embedding_timeout_seconds()
    except RuntimeError as error:
        assert "must be a positive integer" in str(error)
    else:
        raise AssertionError("non-positive refresh embedding timeout should fail")


def test_dashboard_refresh_persists_and_swaps_latest(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ADVISOR_CACHE_DIR", str(tmp_path))
    _reset_refresh_state()

    def fake_refresh_payloads(run_id: str) -> tuple[dict, dict, dict]:
        projects_payload = json.loads(app_module.DATA_PATH.read_text(encoding="utf-8"))
        index_payload = json.loads(app_module.INDEX_PATH.read_text(encoding="utf-8"))
        refreshed_index = ProjectIndex.from_files(app_module.DATA_PATH, app_module.INDEX_PATH)
        refreshed_dashboard = build_dashboard_payload(
            refreshed_index,
            generated_at="2026-06-08T00:00:00+00:00",
        )
        return projects_payload, index_payload, refreshed_dashboard

    monkeypatch.setattr(app_module, "_build_refresh_payloads", fake_refresh_payloads)
    response = dashboard_refresh_start()
    assert response.status_code == 202

    state = _wait_for_refresh()
    assert state["status"] == "succeeded"
    assert (tmp_path / "latest.json").is_file()
    assert state["result"]["project_count"] == len(app_module.index.projects)
    assert dashboard()["provenance"]["snapshot_digest"] == state["result"]["snapshot_digest"]


def test_dashboard_refresh_quest_analysis_uses_minicpm_analyzer(monkeypatch) -> None:
    project = Project(
        id="build-small-hackathon/minicpm-refresh-smoke",
        title="MiniCPM Refresh Smoke",
        summary="A local llama.cpp project that exports field notes.",
        tags=("local-first", "gradio"),
        models=("tinyllama-gguf",),
        datasets=("examples",),
        likes=1,
        sdk="gradio",
        license="mit",
        created_at="2026-06-01T00:00:00+00:00",
        last_modified="2026-06-08T00:00:00+00:00",
        host="https://minicpm-refresh-smoke.hf.space",
        url="https://huggingface.co/spaces/build-small-hackathon/minicpm-refresh-smoke",
        app_file="app.py",
        app_file_embedding_text="download artifact trace report lora training local model",
    )

    class FakeMiniCPMAnalyzer:
        source = "minicpm-json-quest-analyzer"

        def analyze(self, projects):
            assert [item.id for item in projects] == [project.id]
            return {
                project.id: [
                    {
                        "quest": "Off the Grid",
                        "confidence": 0.82,
                        "evidence": "local llama.cpp project",
                        "source": "readme",
                    },
                    {
                        "quest": "Field Notes",
                        "confidence": 0.78,
                        "evidence": "exports field notes",
                        "source": "readme",
                    },
                ]
            }

    monkeypatch.setattr(app_module, "create_quest_analyzer", lambda device: FakeMiniCPMAnalyzer())

    result = app_module._analyze_dashboard_quests([project.to_refresh_snapshot_dict()])

    quests = {match["quest"] for match in result["matches_by_project"][project.id]}
    assert result["source"] == "minicpm-json-quest-analyzer"
    assert quests == {"Off the Grid", "Field Notes"}


def test_dashboard_refresh_quest_analysis_batches_minicpm(monkeypatch) -> None:
    projects = [
        Project(
            id=f"build-small-hackathon/batched-{index}",
            title=f"Batched {index}",
            summary="Small local demo",
            tags=("gradio",),
            models=(),
            datasets=(),
            likes=0,
            sdk="gradio",
            license="mit",
            created_at="2026-06-01T00:00:00+00:00",
            last_modified="2026-06-08T00:00:00+00:00",
            host=f"https://batched-{index}.hf.space",
            url=f"https://huggingface.co/spaces/build-small-hackathon/batched-{index}",
            readme_body="README evidence",
            app_file_source="import gradio as gr",
        )
        for index in range(3)
    ]
    calls = []

    class FakeMiniCPMAnalyzer:
        source = "minicpm-json-quest-analyzer"

        def analyze(self, batch):
            calls.append([project.id for project in batch])
            return {project.id: [] for project in batch}

    monkeypatch.setenv("ADVISOR_QUEST_ANALYSIS_BATCH_SIZE", "2")
    monkeypatch.setattr(app_module, "create_quest_analyzer", lambda device: FakeMiniCPMAnalyzer())

    result = app_module._analyze_dashboard_quests([project.to_refresh_snapshot_dict() for project in projects])

    assert calls == [
        ["build-small-hackathon/batched-0", "build-small-hackathon/batched-1"],
        ["build-small-hackathon/batched-2"],
    ]
    assert set(result["matches_by_project"]) == {project.id for project in projects}


def test_dashboard_refresh_quest_analysis_requires_two_segment_snapshot() -> None:
    project = Project(
        id="build-small-hackathon/missing-evidence",
        title="Missing Evidence",
        summary="summary is not enough",
        tags=("gradio",),
        models=(),
        datasets=(),
        likes=0,
        sdk="gradio",
        license="mit",
        created_at="2026-06-01T00:00:00+00:00",
        last_modified="2026-06-08T00:00:00+00:00",
        host="https://missing-evidence.hf.space",
        url="https://huggingface.co/spaces/build-small-hackathon/missing-evidence",
        app_file="app.py",
        app_file_embedding_text="signals are not enough",
    )
    row = project.to_refresh_snapshot_dict()
    del row["readme_body"]

    try:
        app_module._analyze_dashboard_quests([row])
    except RuntimeError as error:
        assert "readme_body and app_file_source" in str(error)
    else:
        raise AssertionError("quest analysis should require the two-segment refresh snapshot")


def test_agent_turn_stream_endpoint_exports_ndjson_events() -> None:
    response = agent_turn_stream(
        {
            "message": "A local-first archive cartographer for family photos",
            "session_json": "{}",
        }
    )
    payload = asyncio.run(_read_streaming_response(response))
    lines = [json.loads(line) for line in payload.splitlines()]

    assert response.media_type == "application/x-ndjson"
    assert lines[0]["type"] == "start"
    assert any(line["type"] == "token" for line in lines)
    assert lines[-1]["type"] == "done"
    assert lines[-1]["state"]["ideas"]


def test_agent_turn_stream_streams_stage_and_tool_events() -> None:
    response = agent_turn_stream(
        {
            "message": "A local-first archive cartographer for family photos",
            "session_json": "{}",
        }
    )
    payload = asyncio.run(_read_streaming_response(response))
    lines = [json.loads(line) for line in payload.splitlines()]
    types = [line["type"] for line in lines]

    assert "stage" in types
    assert any(line["type"] == "tool_event" and line.get("name") for line in lines)
    assert types.index("stage") < types.index("token")


def test_agent_turn_stream_runs_on_cpu_compute() -> None:
    response = agent_turn_stream(
        {
            "message": "A local-first archive cartographer for family photos",
            "session_json": "{}",
            "compute": "cpu",
        }
    )
    payload = asyncio.run(_read_streaming_response(response))
    lines = [json.loads(line) for line in payload.splitlines()]

    assert lines[0]["type"] == "start"
    assert lines[-1]["type"] == "done"
    assert lines[-1]["state"]["ideas"]


def test_transcribe_audio_endpoint_saves_audio(monkeypatch) -> None:
    captured = {}

    def fake_transcribe(path: str) -> dict:
        captured["path"] = path
        return {
            "transcript": "A local-first memory archive.",
            "model_id": "nvidia/nemotron-speech-streaming-en-0.6b",
            "backend": "nemo-asr",
            "sample_rate": 16000,
        }

    monkeypatch.setattr("app._transcribe_voice", fake_transcribe)

    payload = asyncio.run(transcribe_audio(DummyUpload(b"RIFF....WAVE")))

    assert payload["transcript"] == "A local-first memory archive."
    assert captured["path"].endswith(".wav")


def test_transcribe_audio_endpoint_accepts_octet_stream_audio(monkeypatch) -> None:
    monkeypatch.setattr(
        "app._transcribe_voice",
        lambda path: {
            "transcript": "A local-first memory archive.",
            "model_id": "nvidia/nemotron-speech-streaming-en-0.6b",
            "backend": "nemo-asr",
            "sample_rate": 16000,
        },
    )

    payload = asyncio.run(
        transcribe_audio(
            DummyUpload(b"RIFF....WAVE", filename="idea.wav", content_type="application/octet-stream")
        )
    )

    assert payload["transcript"] == "A local-first memory archive."


def test_transcribe_audio_endpoint_rejects_non_audio() -> None:
    upload = DummyUpload(b"hello", filename="note.txt", content_type="text/plain")

    try:
        asyncio.run(transcribe_audio(upload))
    except Exception as error:
        assert getattr(error, "status_code", None) == 415
    else:
        raise AssertionError("non-audio upload should fail")


def test_transcribe_audio_endpoint_rejects_empty_audio() -> None:
    upload = DummyUpload(b"", filename="empty.wav", content_type="audio/wav")

    try:
        asyncio.run(transcribe_audio(upload))
    except Exception as error:
        assert getattr(error, "status_code", None) == 400
    else:
        raise AssertionError("empty audio upload should fail")


def test_markdown_api_endpoints_return_plain_markdown() -> None:
    state = engine.turn("A local-first archive cartographer for family photos", {}).state

    notes = field_notes_api({"session_json": json.dumps(state)})
    chapter = chapter_api({"session_json": json.dumps(state)})

    assert notes.media_type == "text/markdown; charset=utf-8"
    assert notes.body.decode("utf-8").startswith("# Hackathon Advisor Field Notes")
    assert chapter.media_type == "text/markdown; charset=utf-8"
    assert chapter.body.decode("utf-8").startswith("# The Unwritten Almanac Chapter")


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
    assert "## Session Decisions" in payload
    assert "## Turn Trace" not in payload
    assert "Planner call" not in payload
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
    assert recipe["publish_status"] == "published"
    assert recipe["adapter_repo"] == "build-small-hackathon/hackathon-advisor-minicpm5-lora"


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
    assert payload["voice"]["model_id"] == "nvidia/nemotron-speech-streaming-en-0.6b"
    assert any(badge["name"] == "Sharing is Caring" for badge in payload["badges"])
    assert {badge["name"]: badge["status"] for badge in payload["badges"]}["Llama Champion"] == "ready"
    assert {item["role"]: item["status"] for item in payload["model_stack"]}["Voice input"] == "deployed"
    assert payload["retrieval_index"]["index_algorithm"] == "llama-cpp-embedding-v1"
    assert payload["retrieval_index"]["embedding_runtime"] == "llama.cpp via llama-cpp-python"
    assert payload["training_artifacts"][0]["endpoint"] == "lora_dataset"
    assert payload["training_artifacts"][1]["endpoint"] == "/api/lora-training-kit.zip"
