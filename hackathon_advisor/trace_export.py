from __future__ import annotations

import json
from typing import Any

from hackathon_advisor._text import utc_now


TRACE_SCHEMA_VERSION = 1


def build_trace_jsonl(session: dict[str, Any], metadata: dict[str, Any]) -> str:
    trace = session.get("trace") or []
    ideas = session.get("ideas") or []
    records = [
        {
            "type": "trace_manifest",
            "schema_version": TRACE_SCHEMA_VERSION,
            "generated_at": utc_now(),
            "app": "hackathon-advisor",
            "index": {
                "algorithm": metadata.get("index_algorithm", ""),
                "snapshot_generated_at": metadata.get("snapshot_generated_at", ""),
                "index_generated_at": metadata.get("index_generated_at", ""),
                "snapshot_digest": metadata.get("snapshot_digest", ""),
            },
            "idea_count": len(ideas),
            "turn_count": len(trace),
        }
    ]
    for index, event in enumerate(trace, start=1):
        records.append(
            {
                "type": "agent_turn",
                "schema_version": TRACE_SCHEMA_VERSION,
                "turn_index": index,
                "input": str(event.get("input") or ""),
                "tools": _tools(event),
                "verdict": str(event.get("verdict") or ""),
                "overall": event.get("overall"),
                "plan_steps": int(event.get("plan_steps") or 0),
                "artifact_title": str(event.get("artifact_title") or ""),
                "response": str(event.get("response") or ""),
                "tool_resolution": _tool_resolution(event),
            }
        )
    return "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n"


def trace_metadata(index: Any) -> dict[str, str]:
    metadata = {
        "snapshot_generated_at": index.generated_at,
        "index_generated_at": index.index_generated_at,
        "index_algorithm": index.index_algorithm,
        "snapshot_digest": index.snapshot_digest,
    }
    embedding = getattr(index, "embedding_metadata", None)
    if isinstance(embedding, dict):
        metadata.update(
            {
                "embedding_model_repo": str(embedding.get("model_repo") or ""),
                "embedding_model_file": str(embedding.get("model_file") or ""),
                "embedding_runtime": str(embedding.get("runtime") or ""),
                "embedding_build_source": str(embedding.get("build_source") or ""),
                "embedding_dimensions": str(embedding.get("dimensions") or ""),
                "embedding_builder": str(embedding.get("builder") or ""),
                "embedding_modal_app": str(embedding.get("modal_app") or ""),
            }
        )
    return metadata


def _tools(event: dict[str, Any]) -> list[dict[str, str]]:
    tools = event.get("tools") or []
    return [
        {
            "name": str(tool.get("name") or ""),
            "summary": str(tool.get("summary") or ""),
        }
        for tool in tools
        if isinstance(tool, dict)
    ]


def _tool_resolution(event: dict[str, Any]) -> dict[str, Any]:
    resolution = event.get("tool_resolution") or {}
    call = resolution.get("call") if isinstance(resolution, dict) else {}
    return {
        "status": str(resolution.get("status") or "") if isinstance(resolution, dict) else "",
        "call": {
            "name": str(call.get("name") or "") if isinstance(call, dict) else "",
            "arguments": call.get("arguments") if isinstance(call, dict) else {},
        },
        "errors": list(resolution.get("errors") or []) if isinstance(resolution, dict) else [],
    }
