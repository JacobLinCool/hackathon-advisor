#!/usr/bin/env python3
"""Publish the quest-classification SFT dataset to the Hub as a dataset repo.

The Hub layout is kept viewer-clean: `quest_sft.jsonl` holds only the homogeneous
example rows (the manifest lives in `dataset_manifest.json`, the per-project verified
teacher labels in `provenance/labeled.json`), and the dataset card pins the viewer to
the examples file with a `configs:` block. The local training file keeps its leading
manifest row; `parse_quest_dataset_jsonl` reads either layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "build-small-hackathon/hackathon-advisor-quest-dataset"
ADAPTER_REPO = "build-small-hackathon/hackathon-advisor-quest-minicpm5-lora"


def dataset_card(manifest: dict) -> str:
    qc = manifest.get("quest_positive_counts", {})
    vc = manifest.get("variant_counts", {})
    quest_rows = "\n".join(f"| {q} | {n} |" for q, n in sorted(qc.items(), key=lambda kv: -kv[1]))
    variant_rows = "\n".join(f"| {v} | {n} |" for v, n in sorted(vc.items(), key=lambda kv: -kv[1]))
    return "\n".join(
        [
            "---",
            "configs:",
            "- config_name: default",
            "  data_files:",
            "  - split: train",
            "    path: quest_sft.jsonl",
            "license: apache-2.0",
            "task_categories:",
            "- text-generation",
            "language:",
            "- en",
            "tags:",
            "- hackathon-advisor",
            "- quest-classification",
            "- lora-sft",
            "- minicpm5",
            "pretty_name: Hackathon Advisor Quest Classification SFT",
            "size_categories:",
            "- n<1K",
            "---",
            "",
            "# Hackathon Advisor — Quest Classification SFT Dataset",
            "",
            "Supervised fine-tuning data that teaches MiniCPM5-1B to classify a Build Small",
            "Hackathon project against 13 judging dimensions from a two-segment README + app-file",
            "prompt, emitting strict JSON with short, source-attributed evidence. Trains the LoRA at",
            f"[`{ADAPTER_REPO}`](https://huggingface.co/{ADAPTER_REPO}).",
            "",
            "## Files",
            "",
            "- `quest_sft.jsonl` — the dataset (one `lora_sft_example` per line; the viewer split).",
            "- `dataset_manifest.json` — build manifest and per-quest / per-variant counts.",
            "- `provenance/labeled.json` — the per-project verified teacher labels.",
            "",
            "## Row format (`quest_sft.jsonl`)",
            "",
            "Each line is a chat example with a `messages` list (system / user / assistant). The",
            "assistant turn is exactly one JSON object:",
            "",
            "```json",
            '{"matches":[{"quest":"...","confidence":0.0,"evidence":"...","source":"readme|app_file"}]}',
            "```",
            "",
            "No markdown, no prose, no renamed quests; an empty `matches` list when no dimension has",
            "clear evidence. The user turn splits the project into a `[README]` segment and an",
            "`[APP_FILE]` segment so the model judges product description and implementation",
            "evidence separately and attributes each match to its source.",
            "",
            "## Quest dimensions (13)",
            "",
            "Six merit badges (Off the Grid, Well-Tuned, Off-Brand, Llama Champion, Sharing is",
            "Caring, Field Notes), two tracks (Backyard AI, Thousand Token Wood), and five",
            "sponsor / special awards (OpenBMB, Nemotron, Modal, Tiny Titan, Best Agent).",
            "",
            f"## Examples: {manifest.get('example_count')} ({manifest.get('empty_match_examples')} with empty matches)",
            "",
            "| variant | count |",
            "| --- | --- |",
            variant_rows,
            "",
            "Positive examples per quest:",
            "",
            "| quest | examples |",
            "| --- | --- |",
            quest_rows,
            "",
            "## Provenance",
            "",
            "Built from the real public Spaces of the `build-small-hackathon` org: 125 crawled",
            "projects → deduped + length-filtered to 108 content-rich ones → labelled by a",
            "teacher-then-adversarial-verifier multi-agent workflow → plus targeted augmentations",
            "(app-only, readme-only / missing app file, README↔app contradictions, empty matches,",
            "noisy metadata). Examples are derived from public hackathon submissions for research",
            "and hackathon use; each project remains under its own Space license.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the quest SFT dataset.")
    parser.add_argument("--dataset", default="data/quest_sft.jsonl", type=Path)
    parser.add_argument("--labels", default="data/quest_labels/labeled.json", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    args = parser.parse_args()

    records = [line for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(records[0])
    example_lines = records[1:] if manifest.get("type") == "lora_sft_manifest" else records
    if manifest.get("type") != "lora_sft_manifest":
        manifest = {"type": "lora_sft_manifest", "example_count": len(example_lines)}

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        (staging / "quest_sft.jsonl").write_text("\n".join(example_lines) + "\n", encoding="utf-8")
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "README.md").write_text(dataset_card(manifest), encoding="utf-8")
        if args.labels.exists():
            (staging / "provenance").mkdir()
            (staging / "provenance" / "labeled.json").write_text(
                args.labels.read_text(encoding="utf-8"), encoding="utf-8"
            )
        commit = api.upload_folder(
            folder_path=str(staging),
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Restructure dataset for the Hub viewer (examples-only split + sidecar manifest)",
            delete_patterns=["labeled.json", "*.parquet"],
        )
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_id", None) or str(commit)
    print(f"published dataset https://huggingface.co/datasets/{args.repo_id}")
    print(f"examples: {len(example_lines)} | revision: {revision}")


if __name__ == "__main__":
    main()
