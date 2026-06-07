from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any


LORA_DATASET_SCHEMA_VERSION = 1
BASE_MODEL = "openbmb/MiniCPM5-1B"
ADAPTER_TASK = "hackathon_advisor_tool_call_and_voice"

TOOL_CALL_SYSTEM_PROMPT = (
    "You are The Unwritten Almanac's originality and build-plan advisor. Choose exactly one validated tool call for "
    "the user's project-advice request. Return only the XML function call."
)

RESPONSE_SYSTEM_PROMPT = (
    "You are The Unwritten Almanac's originality and build-plan advisor. Write concise, evidence-grounded advice from "
    "the tool observations, cited pages, score, and selected goals."
)


def build_lora_dataset_jsonl(session: dict[str, Any], metadata: dict[str, Any]) -> str:
    trace = _list_of_dicts(session.get("trace"))
    ideas = _list_of_dicts(session.get("ideas"))
    targets = [str(target) for target in session.get("targets") or []]
    examples = _examples(trace, targets)
    records = [
        {
            "type": "lora_sft_manifest",
            "schema_version": LORA_DATASET_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "app": "hackathon-advisor",
            "base_model": BASE_MODEL,
            "adapter_task": ADAPTER_TASK,
            "format": "chat-jsonl",
            "record_kinds": ["tool_call", "advisor_response"],
            "source": "exact_session_trace",
            "idea_count": len(ideas),
            "turn_count": len(trace),
            "included_turn_count": len({example["turn_index"] for example in examples}),
            "example_count": len(examples),
            "index": _index_metadata(metadata),
        }
    ]
    records.extend(examples)
    return "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n"


def _examples(trace: list[dict[str, Any]], targets: list[str]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for turn_index, event in enumerate(trace, start=1):
        if not _is_successful_turn(event):
            continue
        input_text = _clean(event.get("input"))
        response = _clean(event.get("response"))
        if not input_text or not response:
            continue
        tool_call = _tool_call(event)
        if not tool_call["name"]:
            continue
        shared = {
            "type": "lora_sft_example",
            "schema_version": LORA_DATASET_SCHEMA_VERSION,
            "base_model": BASE_MODEL,
            "adapter_task": ADAPTER_TASK,
            "turn_index": turn_index,
            "targets": targets,
            "score": _score(event),
            "tool_call": tool_call,
            "tool_observations": _tool_observations(event),
        }
        examples.append(
            {
                **shared,
                "example_index": len(examples) + 1,
                "example_kind": "tool_call",
                "messages": [
                    {"role": "system", "content": TOOL_CALL_SYSTEM_PROMPT},
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": _tool_call_xml(tool_call)},
                ],
            }
        )
        examples.append(
            {
                **shared,
                "example_index": len(examples) + 1,
                "example_kind": "advisor_response",
                "messages": [
                    {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": _response_context(input_text, event, tool_call)},
                    {"role": "assistant", "content": response},
                ],
            }
        )
    return examples


def _is_successful_turn(event: dict[str, Any]) -> bool:
    resolution = event.get("tool_resolution") if isinstance(event.get("tool_resolution"), dict) else {}
    return str(resolution.get("status") or "") == "valid"


def _tool_call(event: dict[str, Any]) -> dict[str, Any]:
    resolution = event.get("tool_resolution") if isinstance(event.get("tool_resolution"), dict) else {}
    call = resolution.get("call") if isinstance(resolution.get("call"), dict) else {}
    arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    return {
        "name": _clean(call.get("name")),
        "arguments": arguments,
    }


def _tool_call_xml(tool_call: dict[str, Any]) -> str:
    arguments = json.dumps(tool_call["arguments"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f'<function name="{tool_call["name"]}">{arguments}</function>'


def _response_context(input_text: str, event: dict[str, Any], tool_call: dict[str, Any]) -> str:
    observations = _tool_observations(event)
    lines = [
        input_text,
        "",
        f"Tool call: {_tool_call_xml(tool_call)}",
        "Tool observations:",
    ]
    if observations:
        for observation in observations:
            lines.append(f"- {observation['name']}: {observation['summary']}")
    else:
        lines.append("- none")

    score = _score(event)
    verdict = score["verdict"] or "n/a"
    overall = score["overall"] if score["overall"] is not None else "n/a"
    lines.extend(
        [
            f"Verdict: {verdict}",
            f"Overall: {overall}",
            f"Plan steps: {score['plan_steps']}",
        ]
    )
    return "\n".join(lines)


def _tool_observations(event: dict[str, Any]) -> list[dict[str, str]]:
    observations = []
    for tool in _list_of_dicts(event.get("tools")):
        name = _clean(tool.get("name"))
        summary = _clean(tool.get("summary"))
        if name or summary:
            observations.append({"name": name, "summary": summary})
    return observations


def _score(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": _clean(event.get("verdict")),
        "overall": event.get("overall"),
        "plan_steps": int(event.get("plan_steps") or 0),
    }


def _index_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "algorithm": _clean(metadata.get("index_algorithm")),
        "snapshot_generated_at": _clean(metadata.get("snapshot_generated_at")),
        "index_generated_at": _clean(metadata.get("index_generated_at")),
        "snapshot_digest": _clean(metadata.get("snapshot_digest")),
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
