from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


TRACE_SCHEMA_VERSION = 1


def build_trace_jsonl(session: dict[str, Any], metadata: dict[str, Any]) -> str:
    trace = session.get("trace") or []
    ideas = session.get("ideas") or []
    records = [
        {
            "type": "trace_manifest",
            "schema_version": TRACE_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "app": "hackathon-advisor",
            "index": {
                "algorithm": metadata["index_algorithm"],
                "snapshot_generated_at": metadata["snapshot_generated_at"],
                "index_generated_at": metadata["index_generated_at"],
                "snapshot_digest": metadata["snapshot_digest"],
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
            }
        )
    return "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n"


def trace_metadata(index: Any) -> dict[str, str]:
    return {
        "snapshot_generated_at": index.generated_at,
        "index_generated_at": index.index_generated_at,
        "index_algorithm": index.index_algorithm,
        "snapshot_digest": index.snapshot_digest,
    }


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
