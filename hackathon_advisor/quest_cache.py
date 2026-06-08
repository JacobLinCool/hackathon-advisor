from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from hackathon_advisor.data import Project
from hackathon_advisor.quest_analysis import (
    MAX_QUEST_TOKENS,
    QuestAnalysisError,
    render_project_quest_prompt,
    resolve_quest_identity,
    validate_matches_by_project,
)
from hackathon_advisor.quest_taxonomy import (
    APP_PROMPT_CHAR_LIMIT,
    QUEST_PROFILES,
    QUEST_SYSTEM_PROMPT,
    README_PROMPT_CHAR_LIMIT,
)
from hackathon_advisor._text import utc_now


QUEST_CACHE_SCHEMA_VERSION = 1
QUEST_CACHE_ROOT = Path("quest-cache") / "v1"
QUEST_PROMPT_VERSION = "quest-prompt-v1"
QUEST_ANALYZER_SOURCE = "minicpm-json-quest-analyzer"
QUEST_GENERATION_CONFIG = {
    "enable_thinking": False,
    "temperature": 0.0,
    "do_sample": False,
    "max_new_tokens": MAX_QUEST_TOKENS,
}


@dataclass(frozen=True)
class QuestCacheIdentity:
    project_id: str
    prompt_hash: str
    taxonomy_hash: str
    analyzer_fingerprint: dict[str, Any]
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "prompt_hash": self.prompt_hash,
            "taxonomy_hash": self.taxonomy_hash,
            "analyzer_fingerprint": self.analyzer_fingerprint,
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class QuestCacheEntry:
    identity: QuestCacheIdentity
    matches: list[dict[str, Any]]
    source: str
    path: Path
    generated_at: str


@dataclass(frozen=True)
class QuestCacheLookup:
    identity: QuestCacheIdentity
    entry: QuestCacheEntry | None
    reason: str


def quest_analyzer_fingerprint_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    model_id, adapter_id, adapter_revision = resolve_quest_identity(env)
    return {
        "source": QUEST_ANALYZER_SOURCE,
        "model_id": model_id,
        "adapter_id": adapter_id,
        "adapter_revision": adapter_revision,
        "adapter_digest": _local_artifact_digest(adapter_id),
        "prompt_version": QUEST_PROMPT_VERSION,
        "generation": dict(QUEST_GENERATION_CONFIG),
    }


def quest_taxonomy_hash() -> str:
    payload = {
        "system_prompt": QUEST_SYSTEM_PROMPT,
        "quest_profiles": list(QUEST_PROFILES),
        "readme_prompt_char_limit": README_PROMPT_CHAR_LIMIT,
        "app_prompt_char_limit": APP_PROMPT_CHAR_LIMIT,
        "prompt_version": QUEST_PROMPT_VERSION,
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_quest_cache_identity(
    project: Project,
    analyzer_fingerprint: Mapping[str, Any],
) -> QuestCacheIdentity:
    prompt_hash = sha256(render_project_quest_prompt(project).encode("utf-8")).hexdigest()
    taxonomy_hash = quest_taxonomy_hash()
    canonical_fingerprint = json.loads(_canonical_json(analyzer_fingerprint))
    key_payload = {
        "schema_version": QUEST_CACHE_SCHEMA_VERSION,
        "project_id": project.id,
        "prompt_hash": prompt_hash,
        "taxonomy_hash": taxonomy_hash,
        "analyzer_fingerprint": canonical_fingerprint,
    }
    cache_key = sha256(_canonical_json(key_payload).encode("utf-8")).hexdigest()
    return QuestCacheIdentity(
        project_id=project.id,
        prompt_hash=prompt_hash,
        taxonomy_hash=taxonomy_hash,
        analyzer_fingerprint=canonical_fingerprint,
        cache_key=cache_key,
    )


def quest_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / QUEST_CACHE_ROOT / cache_key[:2] / f"{cache_key}.json"


def read_quest_cache_entry(
    cache_dir: Path,
    project: Project,
    analyzer_fingerprint: Mapping[str, Any],
) -> QuestCacheLookup:
    identity = build_quest_cache_identity(project, analyzer_fingerprint)
    path = quest_cache_path(cache_dir, identity.cache_key)
    if not path.is_file():
        return QuestCacheLookup(identity=identity, entry=None, reason="absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return QuestCacheLookup(identity=identity, entry=None, reason=f"invalid_json:{error}")
    if not isinstance(payload, dict):
        return QuestCacheLookup(identity=identity, entry=None, reason="invalid_payload")
    try:
        entry = _validate_cache_payload(payload, project, identity, path)
    except QuestAnalysisError as error:
        return QuestCacheLookup(identity=identity, entry=None, reason=f"invalid_schema:{error}")
    return QuestCacheLookup(identity=identity, entry=entry, reason="hit")


def write_quest_cache_entry(
    cache_dir: Path,
    project: Project,
    analyzer_fingerprint: Mapping[str, Any],
    matches: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> QuestCacheEntry:
    identity = build_quest_cache_identity(project, analyzer_fingerprint)
    validated = validate_matches_by_project({project.id: list(matches)}, [project], source=source)
    generated_at = utc_now()
    payload = {
        "schema_version": QUEST_CACHE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source": validated.source,
        **identity.to_dict(),
        "matches": validated.matches_by_project[project.id],
    }
    path = quest_cache_path(cache_dir, identity.cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    return QuestCacheEntry(
        identity=identity,
        matches=validated.matches_by_project[project.id],
        source=validated.source,
        path=path,
        generated_at=generated_at,
    )


def quest_cache_run_record(
    *,
    project: Project,
    identity: QuestCacheIdentity,
    matches: Sequence[Mapping[str, Any]],
    status: str,
    source: str,
    path: Path | None = None,
) -> dict[str, Any]:
    return {
        "project_id": project.id,
        "cache_key": identity.cache_key,
        "prompt_hash": identity.prompt_hash,
        "taxonomy_hash": identity.taxonomy_hash,
        "status": status,
        "source": source,
        "cache_path": path.as_posix() if path is not None else "",
        "matches": [dict(match) for match in matches],
    }


def build_quest_analysis_run_payload(
    *,
    run_id: str,
    analyzer_fingerprint: Mapping[str, Any],
    summary: Mapping[str, Any],
    project_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": QUEST_CACHE_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": utc_now(),
        "source": QUEST_ANALYZER_SOURCE,
        "analyzer_fingerprint": json.loads(_canonical_json(analyzer_fingerprint)),
        "taxonomy_hash": quest_taxonomy_hash(),
        "summary": dict(summary),
        "projects": [dict(record) for record in project_records],
    }


def _validate_cache_payload(
    payload: Mapping[str, Any],
    project: Project,
    identity: QuestCacheIdentity,
    path: Path,
) -> QuestCacheEntry:
    if payload.get("schema_version") != QUEST_CACHE_SCHEMA_VERSION:
        raise QuestAnalysisError("unsupported quest cache schema")
    for field, expected in identity.to_dict().items():
        if payload.get(field) != expected:
            raise QuestAnalysisError(f"cache {field} mismatch")
    source = str(payload.get("source") or QUEST_ANALYZER_SOURCE)
    validated = validate_matches_by_project({project.id: payload.get("matches") or []}, [project], source=source)
    generated_at = str(payload.get("generated_at") or "")
    return QuestCacheEntry(
        identity=identity,
        matches=validated.matches_by_project[project.id],
        source=validated.source,
        path=path,
        generated_at=generated_at,
    )


def _local_artifact_digest(raw_path: str) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return ""
    digest = sha256()
    if path.is_file():
        _hash_file_into(digest, path, path.name)
        return digest.hexdigest()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        _hash_file_into(digest, file_path, file_path.relative_to(path).as_posix())
    return digest.hexdigest()


def _hash_file_into(digest: Any, file_path: Path, relative_name: str) -> None:
    digest.update(relative_name.encode("utf-8"))
    digest.update(b"\0")
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
