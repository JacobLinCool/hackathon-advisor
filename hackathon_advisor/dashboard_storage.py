from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
import uuid

from hackathon_advisor.dashboard import validate_dashboard_payload


LATEST_FILENAME = "latest.json"
STORAGE_SCHEMA_VERSION = 1


class DashboardStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class DashboardArtifacts:
    projects_path: Path
    index_path: Path
    dashboard_path: Path
    manifest_path: Path
    dashboard: dict[str, Any]
    manifest: dict[str, Any]
    quest_analysis_path: Path | None = None


def cache_dir_from_env(env: dict[str, str] | None = None) -> Path | None:
    raw = (env or os.environ).get("ADVISOR_CACHE_DIR", "").strip()
    return Path(raw).expanduser() if raw else None


def require_writable_cache_dir(env: dict[str, str] | None = None) -> Path:
    cache_dir = cache_dir_from_env(env)
    if cache_dir is None:
        raise DashboardStorageError(
            "ADVISOR_CACHE_DIR must point to a writable dashboard cache directory. "
            "Use a mounted Storage Bucket in deployment or a local directory in development."
        )
    if not cache_dir.exists() or not cache_dir.is_dir():
        raise DashboardStorageError(f"ADVISOR_CACHE_DIR does not exist or is not a directory: {cache_dir}")
    probe = cache_dir / f".advisor-write-check-{uuid.uuid4().hex}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        raise DashboardStorageError(f"ADVISOR_CACHE_DIR is not writable: {cache_dir}") from error
    return cache_dir


def load_latest_artifacts(cache_dir: Path | None) -> DashboardArtifacts | None:
    if cache_dir is None:
        return None
    latest_path = cache_dir / LATEST_FILENAME
    if not latest_path.exists():
        return None
    latest = _read_json(latest_path)
    if latest.get("schema_version") != STORAGE_SCHEMA_VERSION:
        raise DashboardStorageError("unsupported latest dashboard storage schema")
    projects_path = _resolve_under(cache_dir, str(latest.get("projects") or ""))
    index_path = _resolve_under(cache_dir, str(latest.get("index") or ""))
    dashboard_path = _resolve_under(cache_dir, str(latest.get("dashboard") or ""))
    manifest_path = _resolve_under(cache_dir, str(latest.get("manifest") or ""))
    for path in (projects_path, index_path, dashboard_path, manifest_path):
        if not path.is_file():
            raise DashboardStorageError(f"cached dashboard artifact is missing: {path}")
    dashboard = _read_json(dashboard_path)
    validate_dashboard_payload(dashboard)
    manifest = _read_json(manifest_path)
    return DashboardArtifacts(
        projects_path=projects_path,
        index_path=index_path,
        dashboard_path=dashboard_path,
        manifest_path=manifest_path,
        dashboard=dashboard,
        manifest=manifest,
    )


def persist_refresh_artifacts(
    cache_dir: Path,
    run_id: str,
    *,
    projects_payload: dict[str, Any],
    index_payload: dict[str, Any],
    dashboard_payload: dict[str, Any],
    quest_analysis_payload: dict[str, Any] | None = None,
) -> DashboardArtifacts:
    validate_dashboard_payload(dashboard_payload)
    relative_run_dir = Path("runs") / run_id
    run_dir = cache_dir / relative_run_dir
    run_dir.mkdir(parents=True, exist_ok=False)

    projects_path = run_dir / "projects.json"
    index_path = run_dir / "project_index.json"
    dashboard_path = run_dir / "dashboard.json"
    quest_analysis_path = run_dir / "quest_analysis.json" if quest_analysis_payload is not None else None
    manifest_path = run_dir / "manifest.json"
    _write_json(projects_path, projects_payload)
    _write_json(index_path, index_payload)
    _write_json(dashboard_path, dashboard_payload)
    if quest_analysis_path is not None:
        _write_json(quest_analysis_path, quest_analysis_payload)
    artifact_paths = {
        "projects": _relative(cache_dir, projects_path),
        "index": _relative(cache_dir, index_path),
        "dashboard": _relative(cache_dir, dashboard_path),
    }
    if quest_analysis_path is not None:
        artifact_paths["quest_analysis"] = _relative(cache_dir, quest_analysis_path)
    manifest = {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_count": dashboard_payload["project_count"],
        "snapshot_digest": dashboard_payload["provenance"]["snapshot_digest"],
        "artifacts": artifact_paths,
    }
    _write_json(manifest_path, manifest)

    latest = {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": manifest["generated_at"],
        "projects": _relative(cache_dir, projects_path),
        "index": _relative(cache_dir, index_path),
        "dashboard": _relative(cache_dir, dashboard_path),
        "manifest": _relative(cache_dir, manifest_path),
    }
    if quest_analysis_path is not None:
        latest["quest_analysis"] = _relative(cache_dir, quest_analysis_path)
    latest_path = cache_dir / LATEST_FILENAME
    tmp_path = cache_dir / f".{LATEST_FILENAME}.{run_id}.tmp"
    _write_json(tmp_path, latest)
    os.replace(tmp_path, latest_path)
    return DashboardArtifacts(
        projects_path=projects_path,
        index_path=index_path,
        dashboard_path=dashboard_path,
        manifest_path=manifest_path,
        dashboard=dashboard_payload,
        manifest=manifest,
        quest_analysis_path=quest_analysis_path,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardStorageError(f"could not read dashboard artifact: {path}") from error
    if not isinstance(payload, dict):
        raise DashboardStorageError(f"dashboard artifact must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _resolve_under(root: Path, relative: str) -> Path:
    if not relative:
        raise DashboardStorageError("cached dashboard artifact path is empty")
    target = (root / relative).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise DashboardStorageError(f"cached dashboard artifact path escapes cache dir: {relative}")
    return target
