#!/usr/bin/env python3
"""Publish redacted Codex session logs as a Hugging Face dataset.

The script is intentionally project-agnostic: point it at a project root and a
set of Codex session directories, and it will select sessions that mention the
project, minimize non-project platform metadata, redact public log text with
OpenAI Privacy Filter, then upload the resulting JSONL dataset.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Protocol

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REPO = "build-small-hackathon/hackathon-advisor-codex-traces"
DEFAULT_PRIVACY_FILTER_MODEL = "openai/privacy-filter"

TEXT_KEYS = {
    "arguments",
    "content",
    "images",
    "input",
    "local_images",
    "message",
    "output",
    "queries",
    "query",
    "summary",
    "text",
    "text_elements",
}

SECRET_PATTERNS = [
    re.compile(
        r"(?i)\b(HF_TOKEN|HUGGINGFACEHUB_API_TOKEN|OPENAI_API_KEY|GITHUB_TOKEN|GH_TOKEN|"
        r"ANTHROPIC_API_KEY|API_KEY|TOKEN|PASSWORD|SECRET)\b\s*[:=]\s*['\"]?[^'\"\s,;}]+"
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
]


@dataclass
class RedactionResult:
    text: str
    count: int = 0
    labels: dict[str, int] = field(default_factory=dict)


class TextRedactor(Protocol):
    def redact_many(self, texts: list[str]) -> list[RedactionResult]:
        ...


@dataclass
class SessionStats:
    session_id: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    selected_reason: str
    input_records: int = 0
    published_records: int = 0
    dropped_records: int = 0
    redactions: int = 0
    redaction_labels: dict[str, int] = field(default_factory=dict)
    truncated_fields: int = 0
    truncated_chars: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None


@dataclass(frozen=True)
class TextCaps:
    message: int
    tool_argument: int
    tool_output: int
    other: int


class PrivacyFilterRedactor:
    def __init__(
        self,
        model_id: str,
        *,
        min_score: float,
        batch_size: int,
        chunk_chars: int,
        device: str,
    ) -> None:
        self.model_id = model_id
        self.min_score = min_score
        self.batch_size = max(1, batch_size)
        self.chunk_chars = max(4096, chunk_chars)
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(_privacy_filter_dependency_help()) from error

        try:
            resolved_device = resolve_privacy_filter_device(device)
            self.device = str(resolved_device)
            logging.info("loading privacy filter %s on device %s", model_id, self.device)
            self.classifier = pipeline(
                task="token-classification",
                model=model_id,
                aggregation_strategy="simple",
                device=resolved_device,
            )
        except ValueError as error:
            if "openai_privacy_filter" in str(error):
                raise RuntimeError(_privacy_filter_dependency_help()) from error
            raise

    def redact_many(self, texts: list[str]) -> list[RedactionResult]:
        results: list[RedactionResult | None] = [None] * len(texts)
        pending_indices: list[int] = []
        pending_texts: list[str] = []

        def flush_pending() -> None:
            if not pending_texts:
                return
            for index, result in zip(pending_indices, self._redact_batch(pending_texts)):
                results[index] = result
            pending_indices.clear()
            pending_texts.clear()

        for index, text in enumerate(texts):
            if not text:
                results[index] = RedactionResult(text=text)
                continue
            if len(text) > self.chunk_chars:
                flush_pending()
                results[index] = self._redact_long_text(text)
                continue
            pending_indices.append(index)
            pending_texts.append(text)
            if len(pending_texts) >= self.batch_size:
                flush_pending()
        flush_pending()
        return [result if result is not None else RedactionResult(text=text) for result, text in zip(results, texts)]

    def _redact_long_text(self, text: str) -> RedactionResult:
        pieces: list[str] = []
        total = 0
        labels: dict[str, int] = {}
        chunk_total = (len(text) + self.chunk_chars - 1) // self.chunk_chars
        logging.info(
            "privacy-filter long text: %s chars split into %s chunks",
            len(text),
            chunk_total,
        )
        for chunk_index, start in enumerate(range(0, len(text), self.chunk_chars), start=1):
            if chunk_index == 1 or chunk_index == chunk_total or chunk_index % 10 == 0:
                logging.info(
                    "privacy-filter long text progress: chunk %s/%s (%s remaining)",
                    chunk_index,
                    chunk_total,
                    chunk_total - chunk_index,
                )
            result = self._redact_batch([text[start : start + self.chunk_chars]])[0]
            pieces.append(result.text)
            total += result.count
            _merge_counts(labels, result.labels)
        return RedactionResult(text="".join(pieces), count=total, labels=labels)

    def _redact_batch(self, texts: list[str]) -> list[RedactionResult]:
        outputs = self.classifier(texts, batch_size=self.batch_size)
        if len(texts) == 1 and outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        return [_apply_privacy_spans(text, spans, self.min_score) for text, spans in zip(texts, outputs)]


def resolve_privacy_filter_device(device: str) -> str | int:
    normalized = device.strip().lower()
    if normalized == "auto":
        try:
            import torch
        except ImportError:
            return -1
        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return -1
    if normalized in {"cpu", "-1"}:
        return -1
    if normalized == "cuda":
        return 0
    return device


def _privacy_filter_dependency_help() -> str:
    return (
        "openai/privacy-filter requires a Transformers release that recognizes "
        "model_type=openai_privacy_filter. Run this publisher in an isolated tool "
        "environment, for example:\n\n"
        "uv run --with 'transformers>=5.6,<6' --with 'torch>=2.8,<3' "
        "python scripts/publish_codex_trace_dataset.py --project-root . "
        f"--repo-id {DEFAULT_REPO}"
    )


def _apply_privacy_spans(text: str, spans: list[dict[str, Any]], min_score: float) -> RedactionResult:
    normalized: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    for span in spans:
        start = span.get("start")
        end = span.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            continue
        score = float(span.get("score") or 0.0)
        if score < min_score:
            continue
        raw_label = str(span.get("entity_group") or span.get("entity") or "private")
        label = _redaction_label(raw_label)
        normalized.append({"start": start, "end": end, "label": label, "score": score})

    if not normalized:
        return RedactionResult(text=text)

    normalized.sort(key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, Any]] = []
    for span in normalized:
        if merged and span["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
            if merged[-1]["label"] != span["label"]:
                merged[-1]["label"] = "PRIVATE"
            continue
        merged.append(dict(span))

    redacted = text
    for span in reversed(merged):
        label = span["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        redacted = redacted[: span["start"]] + f"[{label}]" + redacted[span["end"] :]
    return RedactionResult(text=redacted, count=len(merged), labels=label_counts)


def _redaction_label(raw_label: str) -> str:
    label = raw_label
    if len(label) > 2 and label[1] == "-" and label[0] in {"B", "I", "E", "S"}:
        label = label[2:]
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper() or "PRIVATE"


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_remote_url(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    remote = result.stdout.strip()
    return remote or None


def default_session_roots() -> list[Path]:
    home = Path.home()
    return [home / ".codex" / "sessions", home / ".codex" / "archived_sessions"]


def build_project_terms(project_root: Path, includes: list[str]) -> list[str]:
    terms: list[str] = []
    root = project_root.resolve()
    terms.append(str(root))
    terms.append(root.name)
    remote = git_remote_url(root)
    if remote:
        terms.append(remote)
        terms.append(remote.removesuffix(".git").rsplit("/", 1)[-1])
    for term in includes:
        cleaned = term.strip()
        if cleaned:
            terms.append(cleaned)
    deduped: list[str] = []
    for term in terms:
        if len(term) >= 4 and term not in deduped:
            deduped.append(term)
    return deduped


def discover_session_files(session_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in session_roots:
        expanded = root.expanduser()
        if expanded.is_file() and expanded.suffix == ".jsonl":
            files.append(expanded)
        elif expanded.is_dir():
            files.extend(path for path in expanded.rglob("*.jsonl") if path.is_file())
    return sorted(set(files))


def session_matches_project(path: Path, project_terms: list[str]) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                for term in project_terms:
                    if term in line:
                        return True, f"matched term: {term}"
    except UnicodeDecodeError:
        return False, "not utf-8"
    return False, "no project term"


def build_public_payload(
    record_type: str,
    payload: Any,
    project_root: Path,
    path_redaction_prefixes: list[str],
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    if record_type == "session_meta":
        keep = {
            "id",
            "timestamp",
            "cwd",
            "originator",
            "cli_version",
            "source",
            "thread_source",
            "model_provider",
            "memory_mode",
            "git",
        }
        return {
            key: normalize_value(payload[key], project_root, path_redaction_prefixes)
            for key in keep
            if key in payload
        }

    if record_type == "turn_context":
        keep = {
            "turn_id",
            "cwd",
            "workspace_roots",
            "current_date",
            "timezone",
            "model",
            "personality",
            "effort",
            "summary",
            "realtime_active",
        }
        public = {
            key: normalize_value(payload[key], project_root, path_redaction_prefixes)
            for key in keep
            if key in payload
        }
        mode = payload.get("collaboration_mode")
        if isinstance(mode, dict) and "mode" in mode:
            public["collaboration_mode"] = {
                "mode": normalize_value(mode["mode"], project_root, path_redaction_prefixes)
            }
        return public

    if record_type == "event_msg":
        event_type = payload.get("type")
        public: dict[str, Any] = {"type": event_type}
        for key in (
            "turn_id",
            "started_at",
            "model_context_window",
            "collaboration_mode_kind",
            "phase",
            "message",
            "images",
            "local_images",
            "text_elements",
        ):
            if key in payload:
                public[key] = normalize_value(payload[key], project_root, path_redaction_prefixes)
        return public

    if record_type != "response_item":
        return None

    item_type = payload.get("type")
    if item_type == "message":
        return None

    if item_type in {
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "web_search_call",
        "image_generation_call",
        "image_generation_call_output",
    }:
        public = {"type": item_type}
        for key in ("name", "arguments", "input", "output", "call_id", "status", "action"):
            if key in payload:
                public[key] = normalize_value(payload[key], project_root, path_redaction_prefixes)
        return public

    return None


def normalize_value(value: Any, project_root: Path, path_redaction_prefixes: list[str]) -> Any:
    if isinstance(value, str):
        return structural_redact(value, project_root, path_redaction_prefixes)
    if isinstance(value, list):
        return [normalize_value(item, project_root, path_redaction_prefixes) for item in value]
    if isinstance(value, dict):
        return {
            str(key): normalize_value(item, project_root, path_redaction_prefixes)
            for key, item in value.items()
        }
    return value


def structural_redact(text: str, project_root: Path, path_redaction_prefixes: list[str] | None = None) -> str:
    redacted = text.replace(str(project_root.resolve()), "$PROJECT_ROOT")
    prefixes = [str(Path.home()), *(path_redaction_prefixes or [])]
    for prefix in sorted({item for item in prefixes if item}, key=len, reverse=True):
        replacement = "$PROJECT_ROOT" if prefix == str(project_root.resolve()) else "~"
        redacted = redacted.replace(prefix, replacement)
    for pattern in SECRET_PATTERNS:
        if "HF_TOKEN" in pattern.pattern:
            redacted = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED_SECRET]", redacted)
        else:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def collect_text_targets(value: Any, targets: list[tuple[Any, str | int, str]], *, key: str | None = None) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if isinstance(child_value, str) and child_key in TEXT_KEYS:
                targets.append((value, child_key, child_value))
            else:
                collect_text_targets(child_value, targets, key=child_key)
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            if isinstance(child_value, str) and key in TEXT_KEYS:
                targets.append((value, index, child_value))
            else:
                collect_text_targets(child_value, targets, key=key)


def redact_record_batch(records: list[dict[str, Any]], redactor: TextRedactor) -> tuple[int, dict[str, int]]:
    targets: list[tuple[Any, str | int, str]] = []
    for record in records:
        collect_text_targets(record, targets)
    redactions = 0
    labels: dict[str, int] = {}
    for start in range(0, len(targets), 64):
        chunk = targets[start : start + 64]
        results = redactor.redact_many([item[2] for item in chunk])
        for (container, key, _), result in zip(chunk, results):
            container[key] = result.text
            redactions += result.count
            _merge_counts(labels, result.labels)
    return redactions, labels


def truncate_record_batch(records: list[dict[str, Any]], caps: TextCaps) -> tuple[int, int]:
    fields = 0
    chars = 0
    for record in records:
        record_fields, record_chars = truncate_record_text(record, caps)
        fields += record_fields
        chars += record_chars
    return fields, chars


def truncate_record_text(record: dict[str, Any], caps: TextCaps) -> tuple[int, int]:
    payload = record.get("payload")
    payload_type = payload.get("type") if isinstance(payload, dict) else None
    fields = 0
    chars = 0
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if isinstance(child, str) and key in TEXT_KEYS:
                    cap = cap_for_text_field(str(record.get("type")), str(payload_type), str(key), caps)
                    truncated, omitted = truncate_text(child, cap)
                    if omitted:
                        value[key] = truncated
                        fields += 1
                        chars += omitted
                else:
                    stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return fields, chars


def cap_for_text_field(record_type: str, payload_type: str, key: str, caps: TextCaps) -> int:
    if record_type == "event_msg" and key == "message":
        return caps.message
    if payload_type in {"function_call_output", "custom_tool_call_output"} and key == "output":
        return caps.tool_output
    if payload_type in {"function_call", "custom_tool_call"} and key in {"arguments", "input"}:
        return caps.tool_argument
    return caps.other


def truncate_text(text: str, cap: int) -> tuple[str, int]:
    if cap <= 0 or len(text) <= cap:
        return text, 0
    omitted = len(text) - cap
    marker = f"\n[truncated {omitted} chars before privacy filtering]"
    if cap <= len(marker):
        return marker[-cap:], omitted
    return text[: cap - len(marker)] + marker, omitted


def count_text_targets(records: list[dict[str, Any]]) -> int:
    targets: list[tuple[Any, str | int, str]] = []
    for record in records:
        collect_text_targets(record, targets)
    return len(targets)


def session_id_from_record(record: dict[str, Any], fallback: str) -> str:
    if record.get("type") == "session_meta":
        payload = record.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("id"), str):
            return payload["id"]
    return fallback


def iter_public_records(
    path: Path,
    project_root: Path,
    path_redaction_prefixes: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]], SessionStats]:
    fallback_session_id = path.stem.removeprefix("rollout-")
    records: list[dict[str, Any]] = []
    stats = SessionStats(
        session_id=fallback_session_id,
        source_path=display_path(path),
        source_sha256=sha256_file(path),
        source_size_bytes=path.stat().st_size,
        selected_reason="",
    )

    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            stats.input_records += 1
            raw = json.loads(line)
            timestamp = raw.get("timestamp")
            if isinstance(timestamp, str):
                stats.first_timestamp = stats.first_timestamp or timestamp
                stats.last_timestamp = timestamp
            record_type = raw.get("type")
            if record_type == "session_meta":
                stats.session_id = session_id_from_record(raw, fallback_session_id)
            payload = build_public_payload(
                str(record_type),
                raw.get("payload"),
                project_root,
                path_redaction_prefixes or [str(Path.home())],
            )
            if payload is None:
                stats.dropped_records += 1
                continue
            records.append(
                {
                    "schema_version": 1,
                    "session_id": stats.session_id,
                    "record_index": index,
                    "timestamp": timestamp,
                    "type": record_type,
                    "payload": payload,
                }
            )

    for record in records:
        record["session_id"] = stats.session_id
    stats.published_records = len(records)
    return stats.session_id, records, stats


def display_path(path: Path) -> str:
    text = str(path.expanduser())
    home = str(Path.home())
    if text.startswith(home):
        return "~" + text[len(home) :]
    return text


def dataset_card(manifest: dict[str, Any], repo_id: str) -> str:
    privacy = manifest["privacy_filter"]
    return "\n".join(
        [
            "---",
            "configs:",
            "- config_name: default",
            "  data_files:",
            "  - split: train",
            "    path: codex_sessions.jsonl",
            "license: apache-2.0",
            "task_categories:",
            "- text-generation",
            "language:",
            "- en",
            "- zh",
            "tags:",
            "- codex",
            "- agent-traces",
            "- privacy-filter",
            "- hackathon-advisor",
            "pretty_name: Hackathon Advisor Codex Session Traces",
            "---",
            "",
            "# Hackathon Advisor Codex Session Traces",
            "",
            "Real Codex session logs for the Hackathon Advisor project, selected from local Codex",
            "rollout JSONL files and redacted before publication. The event stream preserves user",
            "requests, assistant messages, tool calls, tool outputs, browser/search events, and",
            "minimal session provenance needed to audit how the project was built.",
            "",
            "## Privacy filtering",
            "",
            f"The publisher applied [`{privacy['model_id']}`](https://huggingface.co/{privacy['model_id']})",
            f" at revision `{privacy['revision']}` with minimum score `{privacy['min_score']}`.",
            "System/developer prompts, encrypted payloads, compaction replacement history, and full",
            "tool metadata are intentionally excluded. Local home paths are normalized and common",
            "secret-token shapes are structurally redacted before model filtering. Long text fields",
            "are capped before filtering; the manifest records omitted character counts.",
            "",
            "## Files",
            "",
            "- `codex_sessions.jsonl` — redacted session-event records.",
            "- `dataset_manifest.json` — selected source sessions, raw SHA-256 hashes, counts,",
            "  redaction counts, and publication provenance.",
            "",
            "## Schema",
            "",
            "Each row has:",
            "",
            "```json",
            '{"schema_version":1,"session_id":"...","record_index":0,"timestamp":"...","type":"response_item","payload":{}}',
            "```",
            "",
            "## Build summary",
            "",
            f"- Selected sessions: {manifest['selected_session_count']}",
            f"- Published records: {manifest['published_record_count']}",
            f"- Privacy-filter redactions: {manifest['redaction_count']}",
            f"- Truncated fields: {manifest['truncated_field_count']}",
            f"- Omitted characters from truncated fields: {manifest['truncated_char_count']}",
            "",
            f"Dataset repo: [`{repo_id}`](https://huggingface.co/datasets/{repo_id}).",
            "",
        ]
    )


def build_dataset(
    *,
    project_root: Path,
    session_roots: list[Path],
    include_terms: list[str],
    out_dir: Path,
    redactor: TextRedactor,
    privacy_model_id: str,
    privacy_model_revision: str,
    privacy_device: str,
    min_score: float,
    record_batch_size: int,
    progress_interval_batches: int = 10,
    text_caps: TextCaps = TextCaps(message=4000, tool_argument=2000, tool_output=120, other=1000),
    path_redaction_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    redaction_prefixes = [
        str(project_root),
        str(Path.home()),
        *(path_redaction_prefixes or []),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "codex_sessions.jsonl"

    terms = build_project_terms(project_root, include_terms)
    candidates = discover_session_files(session_roots)
    selected: list[tuple[Path, str]] = []
    for path in candidates:
        matched, reason = session_matches_project(path, terms)
        if matched:
            selected.append((path, reason))
            logging.info("selected session %s (%s)", display_path(path), reason)

    if not selected:
        raise RuntimeError("no Codex session JSONL files matched the project terms")

    logging.info(
        "session selection complete: %s/%s JSONL files selected",
        len(selected),
        len(candidates),
    )

    published_records = 0
    dropped_records = 0
    redaction_count = 0
    redaction_labels: dict[str, int] = {}
    truncated_fields = 0
    truncated_chars = 0
    session_manifests: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as output:
        for session_index, (path, reason) in enumerate(selected, start=1):
            _, records, stats = iter_public_records(path, project_root, redaction_prefixes)
            stats.selected_reason = structural_redact(reason, project_root, redaction_prefixes)
            total_batches = (len(records) + max(1, record_batch_size) - 1) // max(1, record_batch_size)
            session_text_targets = count_text_targets(records)
            logging.info(
                "filtering session %s/%s %s: %s input records, %s public records, "
                "%s text fields, %s dropped",
                session_index,
                len(selected),
                stats.session_id,
                stats.input_records,
                len(records),
                session_text_targets,
                stats.dropped_records,
            )
            batch_size = max(1, record_batch_size)
            progress_interval = max(1, progress_interval_batches)
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                batch_index = (start // batch_size) + 1
                batch_truncated_fields, batch_truncated_chars = truncate_record_batch(batch, text_caps)
                truncated_fields += batch_truncated_fields
                truncated_chars += batch_truncated_chars
                stats.truncated_fields += batch_truncated_fields
                stats.truncated_chars += batch_truncated_chars
                batch_redactions, batch_labels = redact_record_batch(batch, redactor)
                redaction_count += batch_redactions
                stats.redactions += batch_redactions
                _merge_counts(redaction_labels, batch_labels)
                _merge_counts(stats.redaction_labels, batch_labels)
                if batch_index == 1 or batch_index == total_batches or batch_index % progress_interval == 0:
                    processed_after_batch = min(start + len(batch), len(records))
                    remaining = max(0, len(records) - processed_after_batch)
                    logging.info(
                        "privacy-filter session %s/%s %s: batch %s/%s, "
                        "processed records %s/%s, remaining %s, redactions so far %s, "
                        "truncated fields so far %s",
                        session_index,
                        len(selected),
                        stats.session_id,
                        batch_index,
                        total_batches,
                        processed_after_batch,
                        len(records),
                        remaining,
                        stats.redactions,
                        stats.truncated_fields,
                    )
                for record in batch:
                    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    json.loads(line)
                    output.write(line + "\n")
            published_records += stats.published_records
            dropped_records += stats.dropped_records
            logging.info(
                "published %s: %s records, %s privacy redactions, %s truncated fields",
                stats.session_id,
                stats.published_records,
                stats.redactions,
                stats.truncated_fields,
            )
            session_manifests.append(stats.__dict__)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "root_name": project_root.name,
            "git_remote": git_remote_url(project_root),
        },
        "selection": {
            "session_roots": [display_path(path) for path in session_roots],
            "project_terms_sha256": hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest(),
        },
        "privacy_filter": {
            "model_id": privacy_model_id,
            "revision": privacy_model_revision,
            "device": privacy_device,
            "min_score": min_score,
        },
        "redaction_policy": {
            "structural_secret_patterns": len(SECRET_PATTERNS),
            "path_normalization": ["project_root", "home_directory"],
            "path_redaction_prefix_count": len({item for item in redaction_prefixes if item}),
            "dropped_record_types": ["compacted"],
            "dropped_response_items": ["message"],
            "dropped_payload_fields": ["base_instructions", "dynamic_tools", "encrypted_content"],
            "text_caps": {
                "message": text_caps.message,
                "tool_argument": text_caps.tool_argument,
                "tool_output": text_caps.tool_output,
                "other": text_caps.other,
            },
        },
        "selected_session_count": len(session_manifests),
        "published_record_count": published_records,
        "dropped_record_count": dropped_records,
        "redaction_count": redaction_count,
        "redaction_labels": redaction_labels,
        "truncated_field_count": truncated_fields,
        "truncated_char_count": truncated_chars,
        "sessions": session_manifests,
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def upload_dataset(out_dir: Path, repo_id: str, manifest: dict[str, Any]) -> str:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    (out_dir / "README.md").write_text(dataset_card(manifest, repo_id), encoding="utf-8")
    commit = api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Publish redacted Codex session traces",
        allow_patterns=["README.md", "codex_sessions.jsonl", "dataset_manifest.json"],
        delete_patterns=["*.jsonl", "*.json", "README.md", "modal-input/**"],
    )
    return getattr(commit, "oid", None) or getattr(commit, "commit_id", None) or str(commit)


def model_revision(model_id: str) -> str:
    try:
        return HfApi().model_info(model_id).sha or "unknown"
    except Exception as error:  # pragma: no cover - network/auth failures are reported by caller logs.
        logging.warning("could not resolve %s revision: %s", model_id, error)
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--location",
        choices=("local", "modal"),
        default="local",
        help="Where to run the privacy filter (default: local).",
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--session-root", action="append", type=Path, dest="session_roots")
    parser.add_argument("--include", action="append", default=[], help="Additional project term used for selection.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / ".cache" / "codex-trace-dataset")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--privacy-filter-model", default=DEFAULT_PRIVACY_FILTER_MODEL)
    parser.add_argument("--privacy-filter-min-score", type=float, default=0.5)
    parser.add_argument("--privacy-filter-batch-size", type=int, default=32)
    parser.add_argument("--privacy-filter-chunk-chars", type=int, default=12_000)
    parser.add_argument("--privacy-filter-device", default="auto")
    parser.add_argument("--record-batch-size", type=int, default=256)
    parser.add_argument("--progress-interval-batches", type=int, default=10)
    parser.add_argument("--max-message-chars", type=int, default=4000)
    parser.add_argument("--max-tool-argument-chars", type=int, default=2000)
    parser.add_argument("--max-tool-output-chars", type=int, default=120)
    parser.add_argument("--max-other-text-chars", type=int, default=1000)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    if args.location == "modal":
        # Imported lazily so the local path never requires the `modal` package.
        from scripts.modal_publish_codex_trace_dataset import run_modal

        run_modal(args)
        return
    session_roots = args.session_roots or default_session_roots()
    revision = model_revision(args.privacy_filter_model)
    redactor = PrivacyFilterRedactor(
        args.privacy_filter_model,
        min_score=args.privacy_filter_min_score,
        batch_size=args.privacy_filter_batch_size,
        chunk_chars=args.privacy_filter_chunk_chars,
        device=args.privacy_filter_device,
    )
    manifest = build_dataset(
        project_root=args.project_root,
        session_roots=session_roots,
        include_terms=args.include,
        out_dir=args.out_dir,
        redactor=redactor,
        privacy_model_id=args.privacy_filter_model,
        privacy_model_revision=revision,
        privacy_device=redactor.device,
        min_score=args.privacy_filter_min_score,
        record_batch_size=args.record_batch_size,
        progress_interval_batches=args.progress_interval_batches,
        text_caps=TextCaps(
            message=args.max_message_chars,
            tool_argument=args.max_tool_argument_chars,
            tool_output=args.max_tool_output_chars,
            other=args.max_other_text_chars,
        ),
        path_redaction_prefixes=[str(args.project_root.resolve()), str(Path.home())],
    )
    if args.skip_upload:
        print(f"wrote dataset staging directory: {args.out_dir}")
    else:
        commit = upload_dataset(args.out_dir, args.repo_id, manifest)
        print(f"published dataset https://huggingface.co/datasets/{args.repo_id}")
        print(f"revision: {commit}")
    print(
        "summary: "
        f"{manifest['selected_session_count']} sessions, "
        f"{manifest['published_record_count']} records, "
        f"{manifest['redaction_count']} privacy redactions"
    )


if __name__ == "__main__":
    main()
