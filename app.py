from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
from threading import Lock, Thread
import time
from typing import Any, Iterator
from uuid import uuid4

from fastapi import Body, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from gradio import Server

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.artifact_bundle import BUNDLE_FILENAME, build_demo_bundle_zip
from hackathon_advisor.asr_runtime import create_asr_transcriber
from hackathon_advisor.chapter import build_chapter_markdown
from hackathon_advisor.dashboard import build_dashboard_payload
from hackathon_advisor.dashboard_storage import (
    DashboardStorageError,
    cache_dir_from_env,
    load_latest_artifacts,
    persist_refresh_artifacts,
    require_writable_cache_dir,
)
from hackathon_advisor.data import DEFAULT_EMBEDDING_MODEL_FILE, DEFAULT_EMBEDDING_MODEL_REPO, Project, ProjectIndex
from hackathon_advisor.demo_rehearsal import build_demo_rehearsal
from hackathon_advisor.model_runtime import create_tool_planner
from hackathon_advisor.profiling import (
    TurnProfiler,
    configure_logging,
    next_message_index,
)
from hackathon_advisor.field_notes import build_field_notes_markdown
from hackathon_advisor.lora_dataset import build_lora_dataset_jsonl
from hackathon_advisor.lora_training_kit import TRAINING_KIT_FILENAME, build_lora_training_kit_zip
from hackathon_advisor.png_export import artifact_png_filename, render_artifact_png
from hackathon_advisor.prize_ledger import prize_ledger
from hackathon_advisor.quest_analysis import create_quest_analyzer, validate_matches_by_project
from hackathon_advisor.runtime_hooks import install_asyncio_cleanup_hook
from hackathon_advisor.submission_packet import build_submission_packet_markdown
from hackathon_advisor.tool_contracts import resolve_tool_call, tool_schemas
from hackathon_advisor.tools import GOALS, goal_profiles
from hackathon_advisor.trace_export import build_trace_jsonl, trace_metadata
from hackathon_advisor.zerogpu import gpu_task, is_gpu_quota_error, zero_gpu_enabled


configure_logging()
install_asyncio_cleanup_hook()

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
DATA_PATH = ROOT / "data" / "projects.json"
INDEX_PATH = ROOT / "data" / "project_index.json"
PROFILE_FIELDS = ["skills", "time", "preferences", "constraints"]
MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
AUDIO_UPLOAD_SUFFIXES = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus", ".wav", ".webm"}
DEFAULT_HF_ORG = "build-small-hackathon"
DEFAULT_REFRESH_EMBEDDING_TIMEOUT_SECONDS = 1800
REFRESH_SUBPROCESS_LOG_TAIL_LINES = 80
REFRESH_STAGE_LABELS = {
    "crawling": "Fetching public Spaces",
    "embedding": "Rebuilding the embedding index",
    "quest_analysis": "Classifying quest coverage",
    "atlas": "Projecting the atlas",
    "persisting": "Writing dashboard artifacts",
    "swapping": "Activating the latest dashboard",
}

_runtime_lock = Lock()
_refresh_lock = Lock()


def _load_initial_runtime() -> tuple[ProjectIndex, dict[str, Any]]:
    artifacts = load_latest_artifacts(cache_dir_from_env())
    if artifacts is not None:
        loaded_index = ProjectIndex.from_files(artifacts.projects_path, artifacts.index_path)
        return loaded_index, artifacts.dashboard
    loaded_index = ProjectIndex.from_files(DATA_PATH, INDEX_PATH)
    return loaded_index, build_dashboard_payload(loaded_index)


index, dashboard_payload = _load_initial_runtime()

# Acceleration is automatic: on a ZeroGPU Space the GPU path uses accelerate device_map inside
# the @spaces.GPU fork; locally the device resolves CUDA -> Apple MPS -> CPU. CPU is only used
# as an explicit override or a quota fallback.
engine = AdvisorEngine(index, create_tool_planner(device="cuda" if zero_gpu_enabled() else "local"))
voice_transcriber = create_asr_transcriber()
app = Server()

_cpu_engine: AdvisorEngine | None = None
_refresh_state: dict[str, Any] = {
    "status": "idle",
    "run_id": "",
    "stage": "",
    "stage_label": "",
    "started_at": "",
    "finished_at": "",
    "error": "",
    "result": None,
}


def _json_event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _cpu_engine_instance() -> AdvisorEngine:
    """A CPU-pinned advisor engine used for the explicit CPU override and for the automatic
    fallback when a ZeroGPU allocation is denied. Loaded lazily so the CPU model only enters
    memory when CPU is actually used."""
    global _cpu_engine
    if _cpu_engine is None:
        _cpu_engine = AdvisorEngine(index, create_tool_planner(device="cpu"))
    return _cpu_engine


@gpu_task
def _engine_turn_stream_gpu(message: str, session: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield from engine.turn_stream(message, session)


@gpu_task
def _transcribe_voice(audio_path: str) -> dict[str, Any]:
    return voice_transcriber.transcribe(Path(audio_path)).to_dict()


@gpu_task
def _analyze_dashboard_quests(project_rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_evidence_keys = [
        str(item.get("id") or index)
        for index, item in enumerate(project_rows)
        if "readme_body" not in item or "app_file_source" not in item
    ]
    if missing_evidence_keys:
        raise RuntimeError(
            "dashboard quest analysis requires refresh snapshots with readme_body and app_file_source; "
            f"missing evidence keys for {len(missing_evidence_keys)} projects"
        )
    projects = [Project.from_dict(item) for item in project_rows]
    analyzer = create_quest_analyzer(device="cuda" if zero_gpu_enabled() else "local")
    matches = analyzer.analyze(projects)
    source = getattr(analyzer, "source", "quest-analyzer")
    validated = validate_matches_by_project(matches, projects, source=source)
    return {
        "source": validated.source,
        "matches_by_project": validated.matches_by_project,
    }


def _refresh_public_state() -> dict[str, Any]:
    with _refresh_lock:
        return dict(_refresh_state)


def _set_refresh_state(**updates: Any) -> None:
    with _refresh_lock:
        _refresh_state.update(updates)
        stage = str(_refresh_state.get("stage") or "")
        _refresh_state["stage_label"] = REFRESH_STAGE_LABELS.get(stage, "")


def _start_refresh_thread(cache_dir: Path) -> dict[str, Any]:
    with _refresh_lock:
        if _refresh_state.get("status") == "running":
            raise HTTPException(status_code=409, detail="Dashboard refresh is already running.")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        _refresh_state.update(
            {
                "status": "running",
                "run_id": run_id,
                "stage": "crawling",
                "stage_label": REFRESH_STAGE_LABELS["crawling"],
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "finished_at": "",
                "error": "",
                "result": None,
            }
        )
    thread = Thread(target=_run_refresh_job, args=(run_id, cache_dir), daemon=True)
    thread.start()
    return _refresh_public_state()


def _run_refresh_job(run_id: str, cache_dir: Path) -> None:
    try:
        projects_payload, index_payload, refreshed_dashboard = _build_refresh_payloads(run_id)
        _set_refresh_state(stage="persisting")
        artifacts = persist_refresh_artifacts(
            cache_dir,
            run_id,
            projects_payload=projects_payload,
            index_payload=index_payload,
            dashboard_payload=refreshed_dashboard,
        )
        _set_refresh_state(stage="swapping")
        _replace_runtime_from_files(artifacts.projects_path, artifacts.index_path, artifacts.dashboard)
        _set_refresh_state(
            status="succeeded",
            stage="",
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            result={
                "run_id": run_id,
                "project_count": refreshed_dashboard["project_count"],
                "snapshot_digest": refreshed_dashboard["provenance"]["snapshot_digest"],
                "dashboard_generated_at": refreshed_dashboard["generated_at"],
            },
        )
    except Exception as error:  # noqa: BLE001 - background job must report every failure as state
        _set_refresh_state(
            status="failed",
            stage="",
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            error=str(error),
            result=None,
        )


def _build_refresh_payloads(run_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts.crawl_hf_spaces import API, crawl_projects

    org = os.environ.get("ADVISOR_HF_ORG", DEFAULT_HF_ORG).strip() or DEFAULT_HF_ORG
    _set_refresh_state(stage="crawling")
    project_rows = sorted(crawl_projects(org), key=lambda project: project["id"].lower())
    projects_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{API}/spaces?author={org}",
        "projects": project_rows,
    }

    _set_refresh_state(stage="embedding")
    with tempfile.TemporaryDirectory(prefix="advisor-refresh-") as directory:
        project_path = Path(directory) / "projects.json"
        project_path.write_text(json.dumps(projects_payload, ensure_ascii=False), encoding="utf-8")
        reuse_index_path = Path(directory) / "reuse_project_index.json"
        with _runtime_lock:
            reuse_index_path.write_text(json.dumps(index.index_payload, ensure_ascii=False), encoding="utf-8")
        index_payload = _build_refresh_index_payload(
            project_path,
            Path(directory) / "project_index.json",
            reuse_index_path=reuse_index_path,
        )

    projects = [Project.from_dict(item) for item in projects_payload["projects"]]
    refreshed_index = ProjectIndex(
        projects=projects,
        generated_at=str(projects_payload["generated_at"]),
        source=str(projects_payload["source"]),
        index_payload=index_payload,
    )

    _set_refresh_state(stage="quest_analysis")
    quest_analysis = _analyze_dashboard_quests([project.to_refresh_snapshot_dict() for project in projects])
    _set_refresh_state(stage="atlas")
    refreshed_dashboard = build_dashboard_payload(
        refreshed_index,
        quest_matches=quest_analysis["matches_by_project"],
        quest_source=str(quest_analysis["source"]),
    )
    return projects_payload, index_payload, refreshed_dashboard


def _build_refresh_index_payload(
    project_path: Path,
    index_path: Path,
    *,
    reuse_index_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_project_index.py"),
        "--projects",
        str(project_path),
        "--out",
        str(index_path),
        "--model-repo",
        os.environ.get("ADVISOR_EMBEDDING_MODEL_REPO", DEFAULT_EMBEDDING_MODEL_REPO),
        "--model-file",
        os.environ.get("ADVISOR_EMBEDDING_MODEL_FILE", DEFAULT_EMBEDDING_MODEL_FILE),
        "--build-source",
        "space dashboard refresh",
        "--builder",
        "app.py:/api/dashboard/refresh",
    ]
    if reuse_index_path is not None:
        command.extend(["--reuse-index", str(reuse_index_path)])
    model_path = os.environ.get("ADVISOR_EMBEDDING_MODEL_PATH", "").strip()
    if model_path:
        command.extend(["--model-path", model_path])
    n_ctx = os.environ.get("ADVISOR_EMBEDDING_N_CTX", "").strip()
    if n_ctx:
        command.extend(["--n-ctx", n_ctx])
    n_threads = os.environ.get("ADVISOR_EMBEDDING_THREADS", "").strip()
    if n_threads:
        command.extend(["--n-threads", n_threads])

    _run_refresh_index_command(command)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"refresh embedding index build did not write valid JSON: {index_path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("refresh embedding index build returned a non-object JSON payload")
    return payload


def _run_refresh_index_command(command: list[str]) -> None:
    timeout_seconds = _refresh_embedding_timeout_seconds()
    output_tail: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=_refresh_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    started = time.monotonic()
    try:
        while process.poll() is None:
            for key, _event in selector.select(timeout=1):
                line = key.fileobj.readline()
                if line:
                    _record_refresh_subprocess_line(output_tail, line)
            if time.monotonic() - started > timeout_seconds:
                process.kill()
                process.wait(timeout=5)
                raise RuntimeError(
                    "refresh embedding index build timed out "
                    f"after {timeout_seconds} seconds. Last output:\n{_format_output_tail(output_tail)}"
                )
        for line in process.stdout:
            _record_refresh_subprocess_line(output_tail, line)
    finally:
        selector.close()
        process.stdout.close()
    if process.returncode != 0:
        raise RuntimeError(
            "refresh embedding index build failed "
            f"with exit code {process.returncode}. Last output:\n{_format_output_tail(output_tail)}"
        )


def _refresh_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("HF_HOME"):
        cache_dir = cache_dir_from_env()
        if cache_dir is not None:
            hf_home = cache_dir / "huggingface"
            hf_home.mkdir(parents=True, exist_ok=True)
            env["HF_HOME"] = str(hf_home)
    return env


def _refresh_embedding_timeout_seconds() -> int:
    raw = os.environ.get("ADVISOR_REFRESH_EMBEDDING_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_REFRESH_EMBEDDING_TIMEOUT_SECONDS
    timeout = int(raw)
    if timeout <= 0:
        raise RuntimeError("ADVISOR_REFRESH_EMBEDDING_TIMEOUT_SECONDS must be a positive integer.")
    return timeout


def _record_refresh_subprocess_line(output_tail: list[str], raw_line: str) -> None:
    line = raw_line.rstrip()
    if not line:
        return
    print(f"[dashboard-refresh embedding] {line}", flush=True)
    output_tail.append(line)
    del output_tail[:-REFRESH_SUBPROCESS_LOG_TAIL_LINES]


def _format_output_tail(output_tail: list[str]) -> str:
    return "\n".join(output_tail) if output_tail else "(no output)"


def _replace_runtime_from_files(projects_path: Path, index_path: Path, refreshed_dashboard: dict[str, Any]) -> None:
    global index, engine, _cpu_engine, dashboard_payload
    new_index = ProjectIndex.from_files(projects_path, index_path)
    with _runtime_lock:
        index = new_index
        engine = AdvisorEngine(new_index, engine.planner)
        if _cpu_engine is not None:
            _cpu_engine = AdvisorEngine(new_index, _cpu_engine.planner)
        dashboard_payload = refreshed_dashboard


def _session_from_json(session_json: str = "{}") -> dict[str, Any]:
    try:
        session = json.loads(session_json or "{}")
    except json.JSONDecodeError:
        return {}
    return session if isinstance(session, dict) else {}


def _session_from_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    return _session_from_json(str(payload.get("session_json") or "{}"))


def _primary_turn_stream(message: str, session: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if zero_gpu_enabled():
        yield from _engine_turn_stream_gpu(message, session)
    else:
        yield from engine.turn_stream(message, session)


def _agent_turn_events(
    message: str,
    session_json: str = "{}",
    compute: str = "gpu",
) -> Iterator[str]:
    profiler = TurnProfiler(
        message_index=next_message_index(),
        compute=compute,
        backend=str(engine.runtime_status().get("backend", "")),
        message_chars=len(message),
    )
    profiler.log_start()
    try:
        for event in _profiled_turn_events(message, session_json, compute):
            profiler.observe(event)
            yield _json_event(event)
        profiler.device = _active_device(compute)
        profiler.log_summary()
    except Exception as error:  # noqa: BLE001 - log timing/resources even when a turn fails
        profiler.device = _active_device(compute)
        profiler.log_summary(error)
        raise


def _active_device(compute: str) -> str:
    """The torch device the turn actually resolved to (e.g. mps/cuda/cpu), read after the run
    so the lazy model has reported its resolved device."""
    active = _cpu_engine if compute == "cpu" else engine
    try:
        return str(active.runtime_status().get("device", "")) if active is not None else ""
    except Exception:  # noqa: BLE001 - profiling must never break a turn
        return ""


def _profiled_turn_events(
    message: str,
    session_json: str,
    compute: str,
) -> Iterator[dict[str, Any]]:
    session = _session_from_json(session_json)
    if compute != "cpu":
        produced = False
        try:
            for event in _primary_turn_stream(message, session):
                produced = True
                yield event
            return
        except Exception as error:  # noqa: BLE001 - fall back to local on a clean quota failure
            if produced or not is_gpu_quota_error(error):
                raise
            yield {
                "type": "fallback",
                "to": "cpu",
                "reason": "ZeroGPU quota reached — running this turn locally (slower).",
            }

    for event in _cpu_engine_instance().turn_stream(message, session):
        yield event


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/{path:path}")
def static_file(path: str) -> FileResponse:
    target = (STATIC_DIR / path).resolve()
    if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target)


@app.get("/api/dashboard")
def dashboard() -> dict:
    with _runtime_lock:
        payload = dict(dashboard_payload)
    payload["refresh"] = _refresh_public_state()
    return payload


@app.post("/api/dashboard/refresh")
def dashboard_refresh_start() -> JSONResponse:
    try:
        cache_dir = require_writable_cache_dir()
    except DashboardStorageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return JSONResponse(_start_refresh_thread(cache_dir), status_code=202)


@app.get("/api/dashboard/refresh")
def dashboard_refresh_status() -> dict:
    return _refresh_public_state()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "projects": len(index.projects),
        "runtime": engine.runtime_status(),
        "voice": voice_transcriber.status().to_dict(),
        **trace_metadata(index),
    }


@app.get("/api/bootstrap")
def bootstrap() -> dict:
    runtime_status = engine.runtime_status()
    return {
        "project_count": len(index.projects),
        "runtime": runtime_status,
        "voice": voice_transcriber.status().to_dict(),
        **trace_metadata(index),
        "top_projects": [project.to_public_dict() for project in index.top_projects(limit=8)],
        "whitespace": [item.to_dict() for item in index.starter_directions(limit=5)],
        "goal_options": GOALS,
        "goal_profiles": goal_profiles(),
        "default_goals": GOALS[:3],
        "profile_fields": PROFILE_FIELDS,
    }


@app.get("/api/runtime")
def runtime() -> dict:
    return engine.runtime_status()


@app.get("/api/prize-ledger")
def prize_ledger_endpoint() -> dict:
    return prize_ledger(engine.runtime_status(), trace_metadata(index), voice_transcriber.status().to_dict())


@app.get("/api/tool-contracts")
def tool_contracts() -> dict:
    return {
        "tool_count": len(tool_schemas()),
        "tools": tool_schemas(),
    }


@app.get("/api/demo-session")
def demo_session() -> dict:
    return build_demo_rehearsal(engine)


@app.get("/api/demo-bundle.zip")
def demo_bundle() -> Response:
    runtime_status = engine.runtime_status()
    ledger = prize_ledger(runtime_status, trace_metadata(index), voice_transcriber.status().to_dict())
    metadata = {
        **trace_metadata(index),
        "project_count": len(index.projects),
    }
    content = build_demo_bundle_zip(build_demo_rehearsal(engine), metadata, ledger)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{BUNDLE_FILENAME}"'},
    )


@app.post("/api/artifact.png")
def artifact_png(artifact: dict[str, Any] | None = Body(default=None)) -> Response:
    artifact = artifact or {}
    filename = artifact_png_filename(artifact)
    return Response(
        content=render_artifact_png(artifact),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/agent-turn")
def agent_turn_stream(payload: dict[str, Any] | None = Body(default=None)) -> StreamingResponse:
    payload = payload or {}
    message = str(payload.get("message") or "")
    session_json = str(payload.get("session_json") or "{}")
    compute = _normalize_compute(payload.get("compute"))

    def stream() -> Iterator[str]:
        for event in _agent_turn_events(message, session_json, compute):
            yield f"{event}\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def _normalize_compute(value: Any) -> str:
    # Acceleration is automatic; "cpu" is the only manual override (not surfaced in the UI).
    return "cpu" if str(value or "").strip().lower() == "cpu" else "gpu"


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> dict[str, Any]:
    content_type = str(audio.content_type or "")
    filename = Path(str(audio.filename or "voice-note")).name
    suffix = Path(filename).suffix.lower() or ".audio"
    if not _is_audio_upload(content_type, suffix):
        raise HTTPException(status_code=415, detail="Voice input must be an audio file.")
    with tempfile.TemporaryDirectory(prefix="advisor-upload-") as directory:
        source = Path(directory) / f"voice{suffix}"
        await _save_audio_upload(audio, source)
        return _transcribe_voice(str(source))


def _is_audio_upload(content_type: str, suffix: str) -> bool:
    if content_type.startswith("audio/"):
        return True
    if content_type in {"", "application/octet-stream"} and suffix in AUDIO_UPLOAD_SUFFIXES:
        return True
    return False


async def _save_audio_upload(upload: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_AUDIO_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Voice note is too large.")
            handle.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Voice note is empty.")


@app.post("/api/field-notes")
def field_notes_api(payload: dict[str, Any] | None = Body(default=None)) -> Response:
    session = _session_from_payload(payload)
    content = build_field_notes_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )
    return Response(content=content, media_type="text/markdown; charset=utf-8")


@app.post("/api/chapter")
def chapter_api(payload: dict[str, Any] | None = Body(default=None)) -> Response:
    session = _session_from_payload(payload)
    content = build_chapter_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )
    return Response(content=content, media_type="text/markdown; charset=utf-8")


@app.get("/api/lora-training-kit.zip")
def lora_training_kit() -> Response:
    runtime_status = engine.runtime_status()
    ledger = prize_ledger(runtime_status, trace_metadata(index), voice_transcriber.status().to_dict())
    metadata = {
        **trace_metadata(index),
        "project_count": len(index.projects),
    }
    demo = build_demo_rehearsal(engine)
    session = demo.get("session") if isinstance(demo.get("session"), dict) else {}
    content = build_lora_training_kit_zip(session, metadata, ledger)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{TRAINING_KIT_FILENAME}"'},
    )


@app.api(name="tool_contract_check", concurrency_limit=8)
def tool_contract_check(model_output: str, fallback_query: str = "") -> dict:
    return resolve_tool_call(model_output, fallback_query=fallback_query).to_dict()


@app.api(name="trace_artifact", concurrency_limit=8)
def trace_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    return build_trace_jsonl(session, trace_metadata(index))


@app.api(name="field_notes", concurrency_limit=8)
def field_notes_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    return build_field_notes_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )


@app.api(name="chapter", concurrency_limit=8)
def chapter_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    return build_chapter_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )


@app.api(name="lora_dataset", concurrency_limit=8)
def lora_dataset_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    return build_lora_dataset_jsonl(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )


@app.api(name="submission_packet", concurrency_limit=8)
def submission_packet_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    runtime_status = engine.runtime_status()
    return build_submission_packet_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
        prize_ledger(runtime_status, trace_metadata(index), voice_transcriber.status().to_dict()),
    )


@app.api(name="agent_turn", concurrency_limit=4, stream_every=0.04)
def agent_turn(message: str, session_json: str = "{}", compute: str = "gpu") -> Iterator[str]:
    yield from _agent_turn_events(message, session_json, _normalize_compute(compute))


if __name__ == "__main__":
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
