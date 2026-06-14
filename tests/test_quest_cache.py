from __future__ import annotations

import json

from hackathon_advisor.data import Project
from hackathon_advisor.quest_cache import (
    build_quest_cache_identity,
    quest_analyzer_fingerprint_from_env,
    read_quest_cache_entry,
    write_quest_cache_entry,
)


def _project(readme_body: str = "Uses MiniCPM locally.") -> Project:
    return Project(
        id="build-small-hackathon/cache-unit",
        title="Cache Unit",
        summary="A small MiniCPM app.",
        tags=("gradio",),
        models=("openbmb/MiniCPM5-1B",),
        datasets=(),
        likes=0,
        sdk="gradio",
        license="mit",
        created_at="2026-06-01T00:00:00+00:00",
        last_modified="2026-06-08T00:00:00+00:00",
        host="https://cache-unit.hf.space",
        url="https://huggingface.co/spaces/build-small-hackathon/cache-unit",
        app_file="app.py",
        app_file_source="from transformers import AutoModelForCausalLM",
        readme_body=readme_body,
    )


def test_quest_cache_key_changes_when_prompt_changes() -> None:
    fingerprint = quest_analyzer_fingerprint_from_env({"ADVISOR_QUEST_ADAPTER_ID": ""})

    first = build_quest_cache_identity(_project("Uses MiniCPM locally."), fingerprint)
    second = build_quest_cache_identity(_project("Exports a PDF report."), fingerprint)

    assert first.prompt_hash != second.prompt_hash
    assert first.cache_key != second.cache_key


def test_quest_cache_round_trip_validates_cached_matches(tmp_path) -> None:
    project = _project()
    fingerprint = quest_analyzer_fingerprint_from_env({"ADVISOR_QUEST_ADAPTER_ID": ""})
    matches = [
        {
            "quest": "OpenBMB",
            "confidence": 0.91,
            "evidence": "Uses MiniCPM locally",
            "source": "readme",
        }
    ]

    stored = write_quest_cache_entry(
        tmp_path,
        project,
        fingerprint,
        matches,
        source="metadata-first-minicpm-json-quest-analyzer",
    )
    lookup = read_quest_cache_entry(tmp_path, project, fingerprint)

    assert lookup.reason == "hit"
    assert lookup.entry is not None
    assert lookup.entry.path == stored.path
    assert lookup.entry.matches == stored.matches


def test_quest_cache_rejects_corrupt_record(tmp_path) -> None:
    project = _project()
    fingerprint = quest_analyzer_fingerprint_from_env({"ADVISOR_QUEST_ADAPTER_ID": ""})
    stored = write_quest_cache_entry(
        tmp_path,
        project,
        fingerprint,
        [
            {
                "quest": "OpenBMB",
                "confidence": 0.91,
                "evidence": "Uses MiniCPM locally",
                "source": "readme",
            }
        ],
        source="metadata-first-minicpm-json-quest-analyzer",
    )
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["matches"][0]["quest"] = "Unknown Quest"
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    lookup = read_quest_cache_entry(tmp_path, project, fingerprint)

    assert lookup.entry is None
    assert lookup.reason.startswith("invalid_schema:")
