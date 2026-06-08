"""Build the quest-classification SFT dataset.

Two responsibilities:
  1. Turn a crawled corpus record into the README / app-file segments that both the
     teacher labeller and the trained model see (front-loading imports and asset ids
     so the decisive evidence survives the prompt budget).
  2. Emit the chat-JSONL SFT file (manifest row + example rows) consumed by
     scripts/train_minicpm_lora.py and scripts/modal_train_quest_lora.py.
"""
from __future__ import annotations

import json
from typing import Any

from hackathon_advisor.quest_taxonomy import (
    QUEST_SYSTEM_PROMPT,
    QUESTS,
    build_app_segment,
    build_readme_segment,
    normalize_match,
    render_quest_prompt,
)
from hackathon_advisor._text import utc_now


LORA_DATASET_SCHEMA_VERSION = 1
BASE_MODEL = "openbmb/MiniCPM5-1B"
ADAPTER_TASK = "hackathon_advisor_quest_classification"


def project_segments(record: dict[str, Any]) -> tuple[str, str]:
    return (
        build_readme_segment(record.get("readme_body", "")),
        build_app_segment(record.get("app_source", ""), record.get("app_signals", "")),
    )


def render_record_prompt(record: dict[str, Any], readme_segment: str, app_segment: str) -> str:
    return render_quest_prompt(
        title=record.get("title", ""),
        sdk=record.get("sdk", ""),
        declared_models=record.get("models", []),
        tags=record.get("tags", []),
        readme_segment=readme_segment,
        app_file_name=record.get("app_file", ""),
        app_file_segment=app_segment,
    )


def matches_to_completion(matches: list[dict[str, Any]]) -> str:
    """Render the gold completion exactly as the model must emit it (compact JSON)."""
    clean = [normalize_match(match) for match in matches]
    clean.sort(key=lambda match: match["confidence"], reverse=True)
    return json.dumps({"matches": clean}, ensure_ascii=False, separators=(",", ":"))


def build_example(prompt: str, matches: list[dict[str, Any]], *, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "lora_sft_example",
        "schema_version": LORA_DATASET_SCHEMA_VERSION,
        "base_model": BASE_MODEL,
        "adapter_task": ADAPTER_TASK,
        "example_kind": meta.get("kind", "project"),
        "project_id": meta.get("project_id", ""),
        "variant": meta.get("variant", "natural"),
        "match_count": len(matches),
        "quests": sorted({match["quest"] for match in matches}),
        "messages": [
            {"role": "system", "content": QUEST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": matches_to_completion(matches)},
        ],
    }


def build_dataset_jsonl(examples: list[dict[str, Any]], *, source_note: str = "") -> str:
    quest_counts: dict[str, int] = {quest: 0 for quest in QUESTS}
    variant_counts: dict[str, int] = {}
    empty = 0
    for example in examples:
        variant_counts[example["variant"]] = variant_counts.get(example["variant"], 0) + 1
        if example["match_count"] == 0:
            empty += 1
        for quest in example["quests"]:
            quest_counts[quest] = quest_counts.get(quest, 0) + 1
    manifest = {
        "type": "lora_sft_manifest",
        "schema_version": LORA_DATASET_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "app": "hackathon-advisor",
        "base_model": BASE_MODEL,
        "adapter_task": ADAPTER_TASK,
        "format": "chat-jsonl",
        "record_kinds": ["quest_classification"],
        "source": source_note or "build_small_hackathon_real_projects",
        "example_count": len(examples),
        "empty_match_examples": empty,
        "variant_counts": variant_counts,
        "quest_positive_counts": quest_counts,
        "quests": list(QUESTS),
    }
    records = [manifest, *examples]
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"


def parse_quest_dataset_jsonl(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError("quest dataset is empty")
    # Tolerate both layouts: a leading manifest row (local training file), or an
    # examples-only file (the Hub dataset, where the manifest lives in a sidecar so
    # the rows stay homogeneous for the dataset viewer). Synthesize a manifest when absent.
    if records[0].get("type") == "lora_sft_manifest":
        manifest, examples = records[0], records[1:]
    else:
        examples = records
        manifest = {
            "type": "lora_sft_manifest",
            "schema_version": LORA_DATASET_SCHEMA_VERSION,
            "base_model": BASE_MODEL,
            "adapter_task": ADAPTER_TASK,
            "format": "chat-jsonl",
            "example_count": len(examples),
        }
    for index, example in enumerate(examples, start=1):
        if example.get("type") != "lora_sft_example":
            raise ValueError(f"record {index} is not a lora_sft_example")
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"record {index} has no chat messages")
        assistant = messages[-1]
        if assistant.get("role") != "assistant" or not assistant.get("content"):
            raise ValueError(f"record {index} has no assistant completion")
        payload = json.loads(assistant["content"])
        if not isinstance(payload.get("matches"), list):
            raise ValueError(f"record {index} completion has no matches list")
        for match in payload["matches"]:
            normalize_match(match)
    return manifest, examples
