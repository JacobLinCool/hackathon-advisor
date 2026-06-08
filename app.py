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
import traceback
from typing import Any, Iterator
from uuid import uuid4

from fastapi import Body, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from gradio import Server

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.artifact_bundle import BUNDLE_FILENAME, build_demo_bundle_zip
from hackathon_advisor.asr_runtime import create_asr_transcriber
from hackathon_advisor.chapter import build_chapter_markdown
from hackathon_advisor.config import int_env
from hackathon_advisor.dashboard import build_dashboard_payload
from hackathon_advisor.dashboard_storage import (
    DashboardStorageError,
    cache_dir_from_env,
    load_latest_artifacts,
    persist_refresh_artifacts,
    require_writable_cache_dir,
)
from hackathon_advisor.dashboard_search import (
    DEFAULT_SEARCH_LIMIT,
    DashboardSearchIndex,
    normalize_query,
    normalize_search_limit,
)
from hackathon_advisor.data import (
    DEFAULT_EMBEDDING_MODEL_FILE,
    DEFAULT_EMBEDDING_MODEL_REPO,
    Project,
    ProjectIndex,
    normalize_project_tags,
)
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
from hackathon_advisor.quest_cache import (
    build_quest_analysis_run_payload,
    quest_analyzer_fingerprint_from_env,
    quest_cache_run_record,
    read_quest_cache_entry,
    write_quest_cache_entry,
)
from hackathon_advisor.quest_analysis import create_quest_analyzer, validate_matches_by_project
from hackathon_advisor.runtime_hooks import install_asyncio_cleanup_hook
from hackathon_advisor.submission_packet import build_submission_packet_markdown
from hackathon_advisor.tool_contracts import resolve_tool_call, tool_schemas
from hackathon_advisor.tools import GOALS, goal_profiles
from hackathon_advisor.trace_export import build_trace_jsonl, trace_metadata
from hackathon_advisor.zerogpu import gpu_device, gpu_task, is_gpu_quota_error, zero_gpu_enabled


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
DEFAULT_QUEST_ANALYSIS_BATCH_SIZE = 8
DEFAULT_REFRESH_COMPUTE = "cpu"
DEFAULT_SCHEDULED_REFRESH_INTERVAL_SECONDS = 3600
DEFAULT_SCHEDULED_REFRESH_INITIAL_DELAY_SECONDS = 300
DEFAULT_REFRESH_LOCK_TTL_SECONDS = 7200
REFRESH_LOCK_FILENAME = "refresh.lock"
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
_scheduler_lock = Lock()
_scheduler_started = False


def _empty_quest_cache_progress() -> dict[str, Any]:
    return {
        "project_count": 0,
        "hit_count": 0,
        "miss_count": 0,
        "analyzed_count": 0,
        "remaining_count": 0,
        "last_project_id": "",
    }


def _load_initial_runtime() -> tuple[ProjectIndex, dict[str, Any]]:
    artifacts = load_latest_artifacts(cache_dir_from_env())
    if artifacts is not None:
        loaded_index = ProjectIndex.from_files(artifacts.projects_path, artifacts.index_path)
        return loaded_index, artifacts.dashboard
    loaded_index = ProjectIndex.from_files(DATA_PATH, INDEX_PATH)
    return loaded_index, build_dashboard_payload(loaded_index)


index, dashboard_payload = _load_initial_runtime()
dashboard_search_index = DashboardSearchIndex(index.projects, dashboard_payload)

# Acceleration is automatic: on a ZeroGPU Space the GPU path uses accelerate device_map inside
# the @spaces.GPU fork; locally the device resolves CUDA -> Apple MPS -> CPU. CPU is only used
# as an explicit override or a quota fallback.
engine = AdvisorEngine(index, create_tool_planner(device=gpu_device()))
voice_transcriber = create_asr_transcriber()
app = Server()

_cpu_engine: AdvisorEngine | None = None
_refresh_state: dict[str, Any] = {
    "status": "idle",
    "run_id": "",
    "compute": "",
    "reason": "",
    "stage": "",
    "stage_label": "",
    "started_at": "",
    "finished_at": "",
    "error": "",
    "result": None,
    "quest_cache": _empty_quest_cache_progress(),
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


def _analyze_dashboard_quests(
    project_rows: list[dict[str, Any]],
    *,
    cache_dir: Path,
    compute: str,
    run_id: str,
) -> dict[str, Any]:
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
    analyzer_fingerprint = quest_analyzer_fingerprint_from_env()
    matches_by_project: dict[str, list[dict[str, Any]]] = {}
    record_by_project: dict[str, dict[str, Any]] = {}
    misses: list[tuple[Project, dict[str, Any]]] = []
    hit_count = 0
    miss_count = 0
    analyzed_count = 0
    source = str(analyzer_fingerprint["source"])
    batch_size = _quest_analysis_batch_size()
    _set_quest_cache_progress(
        project_count=len(projects),
        hit_count=0,
        miss_count=0,
        analyzed_count=0,
        remaining_count=len(projects),
        last_project_id="",
    )
    _refresh_lease_heartbeat(cache_dir, run_id)

    for project in projects:
        lookup = read_quest_cache_entry(cache_dir, project, analyzer_fingerprint)
        if lookup.entry is not None:
            hit_count += 1
            matches_by_project[project.id] = lookup.entry.matches
            record_by_project[project.id] = quest_cache_run_record(
                project=project,
                identity=lookup.identity,
                matches=lookup.entry.matches,
                status="cached",
                source=lookup.entry.source,
                path=lookup.entry.path,
            )
            print(
                f"[quest-cache] hit {project.id} key={lookup.identity.cache_key[:12]} "
                f"matches={len(lookup.entry.matches)}",
                flush=True,
            )
        else:
            miss_count += 1
            misses.append((project, lookup.identity.to_dict()))
            print(
                f"[quest-cache] miss {project.id} key={lookup.identity.cache_key[:12]} "
                f"reason={lookup.reason}",
                flush=True,
            )
        _set_quest_cache_progress(
            project_count=len(projects),
            hit_count=hit_count,
            miss_count=miss_count,
            analyzed_count=analyzed_count,
            remaining_count=len(projects) - hit_count - analyzed_count,
            last_project_id=project.id,
        )
        _refresh_lease_heartbeat(cache_dir, run_id)

    for start in range(0, len(misses), batch_size):
        batch = misses[start : start + batch_size]
        batch_projects = [item[0] for item in batch]
        batch_rows = [project.to_refresh_snapshot_dict() for project in batch_projects]
        result = _analyze_dashboard_quest_batch(batch_rows, compute=compute)
        source = str(result["source"])
        validated_batch = validate_matches_by_project(
            result["matches_by_project"],
            batch_projects,
            source=source,
        )
        for project, _identity_row in batch:
            entry = write_quest_cache_entry(
                cache_dir,
                project,
                analyzer_fingerprint,
                validated_batch.matches_by_project[project.id],
                source=source,
            )
            analyzed_count += 1
            matches_by_project[project.id] = entry.matches
            record_by_project[project.id] = quest_cache_run_record(
                project=project,
                identity=entry.identity,
                matches=entry.matches,
                status="analyzed",
                source=entry.source,
                path=entry.path,
            )
            print(
                f"[quest-cache] analyzed {project.id} key={entry.identity.cache_key[:12]} "
                f"matches={len(entry.matches)}",
                flush=True,
            )
            _set_quest_cache_progress(
                project_count=len(projects),
                hit_count=hit_count,
                miss_count=miss_count,
                analyzed_count=analyzed_count,
                remaining_count=len(projects) - hit_count - analyzed_count,
                last_project_id=project.id,
            )
            _refresh_lease_heartbeat(cache_dir, run_id)
    validated = validate_matches_by_project(matches_by_project, projects, source=source)
    summary = {
        "project_count": len(projects),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "analyzed_count": analyzed_count,
        "remaining_count": 0,
        "compute": compute,
    }
    project_records = [record_by_project[project.id] for project in projects]
    return {
        "source": validated.source,
        "matches_by_project": validated.matches_by_project,
        "quest_analysis_payload": build_quest_analysis_run_payload(
            run_id=run_id,
            analyzer_fingerprint=analyzer_fingerprint,
            summary=summary,
            project_records=project_records,
        ),
    }


@gpu_task
def _analyze_dashboard_quest_batch_gpu(project_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _analyze_dashboard_quest_batch_with_device(
        project_rows,
        device=gpu_device(),
    )


def _analyze_dashboard_quest_batch_cpu(project_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _analyze_dashboard_quest_batch_with_device(project_rows, device="cpu")


def _analyze_dashboard_quest_batch(project_rows: list[dict[str, Any]], *, compute: str) -> dict[str, Any]:
    if compute == "gpu":
        return _analyze_dashboard_quest_batch_gpu(project_rows)
    return _analyze_dashboard_quest_batch_cpu(project_rows)


def _analyze_dashboard_quest_batch_with_device(project_rows: list[dict[str, Any]], *, device: str) -> dict[str, Any]:
    projects = [Project.from_dict(item) for item in project_rows]
    analyzer = create_quest_analyzer(device=device)
    matches = analyzer.analyze(projects)
    source = getattr(analyzer, "source", "quest-analyzer")
    validated = validate_matches_by_project(matches, projects, source=source)
    return {
        "source": validated.source,
        "matches_by_project": validated.matches_by_project,
    }


def _quest_analysis_batch_size() -> int:
    return int_env(
        "ADVISOR_QUEST_ANALYSIS_BATCH_SIZE",
        DEFAULT_QUEST_ANALYSIS_BATCH_SIZE,
        minimum=1,
    )


def _refresh_public_state() -> dict[str, Any]:
    with _refresh_lock:
        state = dict(_refresh_state)
        state["quest_cache"] = dict(_refresh_state.get("quest_cache") or _empty_quest_cache_progress())
        return state


def _set_refresh_state(**updates: Any) -> None:
    with _refresh_lock:
        if "quest_cache" in updates:
            updates["quest_cache"] = dict(updates["quest_cache"])
        _refresh_state.update(updates)
        stage = str(_refresh_state.get("stage") or "")
        _refresh_state["stage_label"] = REFRESH_STAGE_LABELS.get(stage, "")


def _set_quest_cache_progress(**updates: Any) -> None:
    with _refresh_lock:
        progress = dict(_refresh_state.get("quest_cache") or _empty_quest_cache_progress())
        progress.update(updates)
        _refresh_state["quest_cache"] = progress


def _normalize_refresh_compute(value: Any) -> str:
    compute = str(value or "").strip().lower() or DEFAULT_REFRESH_COMPUTE
    if compute not in {"cpu", "gpu"}:
        raise HTTPException(status_code=400, detail="Dashboard refresh compute must be 'cpu' or 'gpu'.")
    return compute


def _default_refresh_compute() -> str:
    return _normalize_refresh_compute(os.environ.get("ADVISOR_REFRESH_COMPUTE", DEFAULT_REFRESH_COMPUTE))


def _refresh_lock_ttl_seconds() -> int:
    return int_env(
        "ADVISOR_REFRESH_LOCK_TTL_SECONDS",
        DEFAULT_REFRESH_LOCK_TTL_SECONDS,
        minimum=1,
    )


def _refresh_lock_path(cache_dir: Path) -> Path:
    return cache_dir / REFRESH_LOCK_FILENAME


def _acquire_refresh_lease(cache_dir: Path, *, run_id: str, compute: str, reason: str) -> None:
    lock_path = _refresh_lock_path(cache_dir)
    now = time.time()
    lease = {
        "schema_version": 1,
        "run_id": run_id,
        "compute": compute,
        "reason": reason,
        "owner": _refresh_owner(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expires_at_epoch": now + _refresh_lock_ttl_seconds(),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as error:
            existing = _read_refresh_lease(lock_path)
            if existing is None or _refresh_lease_expired(existing):
                run_label = str((existing or {}).get("run_id") or "unknown")
                print(f"[dashboard-refresh] removing stale refresh lock run={run_label}", flush=True)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as unlink_error:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Dashboard refresh lock exists and could not be removed: {unlink_error}",
                    ) from unlink_error
                continue
            raise HTTPException(
                status_code=409,
                detail=(
                    "Dashboard refresh is already running "
                    f"(run {existing.get('run_id', 'unknown')}, owner {existing.get('owner', 'unknown')})."
                ),
            ) from error
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(lease, ensure_ascii=False) + "\n")
        print(
            f"[dashboard-refresh] acquired refresh lock run={run_id} compute={compute} reason={reason}",
            flush=True,
        )
        return


def _release_refresh_lease(cache_dir: Path, run_id: str) -> None:
    lock_path = _refresh_lock_path(cache_dir)
    existing = _read_refresh_lease(lock_path)
    if existing is None:
        return
    if str(existing.get("run_id") or "") != run_id:
        print(
            f"[dashboard-refresh] refresh lock belongs to {existing.get('run_id', 'unknown')}; "
            f"not releasing run={run_id}",
            flush=True,
        )
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    print(f"[dashboard-refresh] released refresh lock run={run_id}", flush=True)


def _refresh_lease_heartbeat(cache_dir: Path, run_id: str) -> None:
    lock_path = _refresh_lock_path(cache_dir)
    existing = _read_refresh_lease(lock_path)
    if existing is None or str(existing.get("run_id") or "") != run_id:
        return
    existing["heartbeat_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing["expires_at_epoch"] = time.time() + _refresh_lock_ttl_seconds()
    tmp_path = lock_path.with_name(f".{REFRESH_LOCK_FILENAME}.{run_id}.heartbeat.tmp")
    tmp_path.write_text(json.dumps(existing, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, lock_path)


def _read_refresh_lease(lock_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _refresh_lease_expired(lease: dict[str, Any]) -> bool:
    try:
        expires_at = float(lease.get("expires_at_epoch"))
    except (TypeError, ValueError):
        return True
    return expires_at <= time.time()


def _refresh_owner() -> str:
    node = getattr(os, "uname", lambda: None)()
    host = getattr(node, "nodename", "") if node is not None else ""
    return f"{host or 'process'}:{os.getpid()}"


def _start_refresh_thread(cache_dir: Path, *, compute: str, reason: str) -> dict[str, Any]:
    compute = _normalize_refresh_compute(compute)
    with _refresh_lock:
        if _refresh_state.get("status") == "running":
            raise HTTPException(status_code=409, detail="Dashboard refresh is already running.")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        _acquire_refresh_lease(cache_dir, run_id=run_id, compute=compute, reason=reason)
        _refresh_state.update(
            {
                "status": "running",
                "run_id": run_id,
                "compute": compute,
                "reason": reason,
                "stage": "crawling",
                "stage_label": REFRESH_STAGE_LABELS["crawling"],
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "finished_at": "",
                "error": "",
                "result": None,
                "quest_cache": _empty_quest_cache_progress(),
            }
        )
    thread = Thread(target=_run_refresh_job, args=(run_id, cache_dir, compute), daemon=True)
    try:
        thread.start()
    except Exception:
        _release_refresh_lease(cache_dir, run_id)
        _set_refresh_state(
            status="idle",
            run_id="",
            compute="",
            reason="",
            stage="",
            started_at="",
            finished_at="",
            error="",
            result=None,
            quest_cache=_empty_quest_cache_progress(),
        )
        raise
    return _refresh_public_state()


def _run_refresh_job(run_id: str, cache_dir: Path, compute: str) -> None:
    try:
        projects_payload, index_payload, refreshed_dashboard, quest_analysis_payload = _build_refresh_payloads(
            run_id,
            cache_dir=cache_dir,
            compute=compute,
        )
        _set_refresh_state(stage="persisting")
        _refresh_lease_heartbeat(cache_dir, run_id)
        artifacts = persist_refresh_artifacts(
            cache_dir,
            run_id,
            projects_payload=projects_payload,
            index_payload=index_payload,
            dashboard_payload=refreshed_dashboard,
            quest_analysis_payload=quest_analysis_payload,
        )
        _set_refresh_state(stage="swapping")
        _refresh_lease_heartbeat(cache_dir, run_id)
        _replace_runtime_from_files(artifacts.projects_path, artifacts.index_path, artifacts.dashboard)
        _release_refresh_lease(cache_dir, run_id)
        _set_refresh_state(
            status="succeeded",
            stage="",
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            result={
                "run_id": run_id,
                "project_count": refreshed_dashboard["project_count"],
                "snapshot_digest": refreshed_dashboard["provenance"]["snapshot_digest"],
                "dashboard_generated_at": refreshed_dashboard["generated_at"],
                "quest_cache": dict(quest_analysis_payload.get("summary") or {}),
            },
        )
    except Exception as error:  # noqa: BLE001 - background job must report every failure as state
        print("[dashboard-refresh] failed", flush=True)
        traceback.print_exception(type(error), error, error.__traceback__)
        _release_refresh_lease(cache_dir, run_id)
        _set_refresh_state(
            status="failed",
            stage="",
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            error=_format_refresh_error(error),
            result=None,
        )
    finally:
        _release_refresh_lease(cache_dir, run_id)


def _build_refresh_payloads(
    run_id: str,
    *,
    cache_dir: Path,
    compute: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts.crawl_hf_spaces import API, crawl_projects

    org = os.environ.get("ADVISOR_HF_ORG", DEFAULT_HF_ORG).strip() or DEFAULT_HF_ORG
    _set_refresh_state(stage="crawling")
    _refresh_lease_heartbeat(cache_dir, run_id)
    project_rows = sorted(crawl_projects(org), key=lambda project: project["id"].lower())
    projects_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{API}/spaces?author={org}",
        "projects": project_rows,
    }

    _set_refresh_state(stage="embedding")
    _refresh_lease_heartbeat(cache_dir, run_id)
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
    _refresh_lease_heartbeat(cache_dir, run_id)
    quest_analysis = _analyze_dashboard_quests(
        [project.to_refresh_snapshot_dict() for project in projects],
        cache_dir=cache_dir,
        compute=compute,
        run_id=run_id,
    )
    _set_refresh_state(stage="atlas")
    _refresh_lease_heartbeat(cache_dir, run_id)
    refreshed_dashboard = build_dashboard_payload(
        refreshed_index,
        quest_matches=quest_analysis["matches_by_project"],
        quest_source=str(quest_analysis["source"]),
    )
    return projects_payload, index_payload, refreshed_dashboard, quest_analysis["quest_analysis_payload"]


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
    return int_env(
        "ADVISOR_REFRESH_EMBEDDING_TIMEOUT_SECONDS",
        DEFAULT_REFRESH_EMBEDDING_TIMEOUT_SECONDS,
        minimum=1,
    )


def _record_refresh_subprocess_line(output_tail: list[str], raw_line: str) -> None:
    line = raw_line.rstrip()
    if not line:
        return
    print(f"[dashboard-refresh embedding] {line}", flush=True)
    output_tail.append(line)
    del output_tail[:-REFRESH_SUBPROCESS_LOG_TAIL_LINES]


def _format_output_tail(output_tail: list[str]) -> str:
    return "\n".join(output_tail) if output_tail else "(no output)"


def _format_refresh_error(error: BaseException) -> str:
    parts = [f"{type(error).__name__}: {error}"]
    cause = error.__cause__
    if cause is not None:
        parts.append(f"caused by {type(cause).__name__}: {cause}")
    context = error.__context__
    if context is not None and context is not cause:
        parts.append(f"context {type(context).__name__}: {context}")
    return "; ".join(parts)


def _replace_runtime_from_files(projects_path: Path, index_path: Path, refreshed_dashboard: dict[str, Any]) -> None:
    global index, engine, _cpu_engine, dashboard_payload, dashboard_search_index
    new_index = ProjectIndex.from_files(projects_path, index_path)
    new_search_index = DashboardSearchIndex(new_index.projects, refreshed_dashboard)
    with _runtime_lock:
        index = new_index
        engine = AdvisorEngine(new_index, engine.planner)
        if _cpu_engine is not None:
            _cpu_engine = AdvisorEngine(new_index, _cpu_engine.planner)
        dashboard_payload = refreshed_dashboard
        dashboard_search_index = new_search_index


def _public_dashboard_payload(payload: dict[str, Any]) -> dict[str, Any]:
    public_payload = dict(payload)
    public_payload["points"] = [_public_dashboard_point(point) for point in payload.get("points") or []]
    return public_payload


def _public_dashboard_point(point: Any) -> dict[str, Any]:
    if not isinstance(point, dict):
        return {}
    public_point = dict(point)
    public_point["tags"] = list(normalize_project_tags(public_point.get("tags") or []))
    return public_point


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
        payload = _public_dashboard_payload(dashboard_payload)
    payload["refresh"] = _refresh_public_state()
    return payload


@app.get("/api/dashboard/search")
def dashboard_search(q: str = "", limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    query = normalize_query(q)
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required.")
    try:
        normalized_limit = normalize_search_limit(limit)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with _runtime_lock:
        search_index = dashboard_search_index
        current_dashboard = dashboard_payload
    payload = search_index.search(query, limit=normalized_limit)
    public_points = {
        str(point.get("id") or ""): _public_dashboard_point(point)
        for point in current_dashboard.get("points") or []
        if isinstance(point, dict)
    }
    for result in payload["results"]:
        result["point"] = public_points.get(str(result.get("project_id") or ""), {})
    provenance = current_dashboard.get("provenance", {})
    payload["provenance"] = {
        "snapshot_digest": str(provenance.get("snapshot_digest") or ""),
        "snapshot_generated_at": str(provenance.get("snapshot_generated_at") or ""),
    }
    return payload


@app.post("/api/dashboard/refresh")
def dashboard_refresh_start(payload: dict[str, Any] | None = None) -> JSONResponse:
    try:
        cache_dir = require_writable_cache_dir()
    except DashboardStorageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    compute = _refresh_compute_from_payload(payload)
    return JSONResponse(_start_refresh_thread(cache_dir, compute=compute, reason="manual"), status_code=202)


@app.get("/api/dashboard/refresh")
def dashboard_refresh_status() -> dict:
    return _refresh_public_state()


def _refresh_compute_from_payload(payload: dict[str, Any] | None) -> str:
    payload = payload or {}
    return _normalize_refresh_compute(payload.get("compute") or _default_refresh_compute())


def _start_scheduled_refresh_loop() -> None:
    global _scheduler_started
    if not _scheduled_refresh_enabled():
        return
    with _scheduler_lock:
        if _scheduler_started:
            return
        _scheduler_started = True
    interval = _scheduled_refresh_interval_seconds()
    initial_delay = _scheduled_refresh_initial_delay_seconds()
    compute = _scheduled_refresh_compute()
    print(
        "[dashboard-refresh scheduler] enabled "
        f"interval={interval}s initial_delay={initial_delay}s compute={compute}",
        flush=True,
    )
    Thread(
        target=_scheduled_refresh_loop,
        args=(interval, initial_delay),
        daemon=True,
        name="dashboard-refresh-scheduler",
    ).start()


def _scheduled_refresh_enabled() -> bool:
    disabled = os.environ.get("ADVISOR_DISABLE_SCHEDULED_REFRESH", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return False
    raw = os.environ.get("ADVISOR_SCHEDULED_REFRESH", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return cache_dir_from_env() is not None


def _scheduled_refresh_interval_seconds() -> int:
    raw = (
        os.environ.get("ADVISOR_REFRESH_INTERVAL_SECONDS", "").strip()
        or os.environ.get("ADVISOR_SCHEDULED_REFRESH_INTERVAL_SECONDS", "").strip()
    )
    if not raw:
        return DEFAULT_SCHEDULED_REFRESH_INTERVAL_SECONDS
    interval = int(raw)
    if interval <= 0:
        raise RuntimeError("ADVISOR_REFRESH_INTERVAL_SECONDS must be a positive integer.")
    return interval


def _scheduled_refresh_initial_delay_seconds() -> int:
    raw = os.environ.get("ADVISOR_REFRESH_INITIAL_DELAY_SECONDS", "").strip()
    if not raw:
        return DEFAULT_SCHEDULED_REFRESH_INITIAL_DELAY_SECONDS
    delay = int(raw)
    if delay < 0:
        raise RuntimeError("ADVISOR_REFRESH_INITIAL_DELAY_SECONDS must not be negative.")
    return delay


def _scheduled_refresh_compute() -> str:
    return _normalize_refresh_compute(
        os.environ.get("ADVISOR_SCHEDULED_REFRESH_COMPUTE", "").strip() or _default_refresh_compute()
    )


def _scheduled_refresh_loop(interval_seconds: int, initial_delay_seconds: int) -> None:
    if initial_delay_seconds:
        time.sleep(initial_delay_seconds)
    while True:
        _run_scheduled_refresh_once()
        time.sleep(interval_seconds)


def _run_scheduled_refresh_once() -> None:
    try:
        cache_dir = require_writable_cache_dir()
        state = _start_refresh_thread(
            cache_dir,
            compute=_scheduled_refresh_compute(),
            reason="scheduled",
        )
        print(
            f"[dashboard-refresh scheduler] started run={state.get('run_id', '')} "
            f"compute={state.get('compute', '')}",
            flush=True,
        )
    except HTTPException as error:
        if error.status_code == 409:
            print(f"[dashboard-refresh scheduler] skipped: {error.detail}", flush=True)
            return
        print(f"[dashboard-refresh scheduler] failed to start: {error.detail}", flush=True)
    except Exception as error:  # noqa: BLE001 - scheduler must keep running after transient failures
        print(f"[dashboard-refresh scheduler] failed to start: {_format_refresh_error(error)}", flush=True)


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


_start_scheduled_refresh_loop()


if __name__ == "__main__":
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
