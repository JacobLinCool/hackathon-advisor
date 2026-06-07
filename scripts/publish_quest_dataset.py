#!/usr/bin/env python3
"""Publish the quest-classification SFT dataset to the Hub as a dataset repo.

Uploads data/quest_sft.jsonl (manifest + examples), the per-project verified teacher
labels, and a generated dataset card. Prints the dataset URL and commit revision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
            "license: apache-2.0",
            "task_categories:",
            "- text-classification",
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
            "## Format (`quest_sft.jsonl`)",
            "",
            "Chat-JSONL. The **first line** is a `lora_sft_manifest`; every following line is a",
            "`lora_sft_example` with a `messages` list (system / user / assistant). The assistant",
            "turn is exactly one JSON object:",
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
            "noisy metadata). `labeled.json` holds the per-project verified labels. Examples are",
            "derived from public hackathon submissions for research and hackathon use; each project",
            "remains under its own Space license.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the quest SFT dataset.")
    parser.add_argument("--dataset", default="data/quest_sft.jsonl", type=Path)
    parser.add_argument("--labels", default="data/quest_labels/labeled.json", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    args = parser.parse_args()

    manifest = json.loads(next(line for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()))
    card_path = ROOT / "data" / "quest_dataset_card.md"
    card_path.write_text(dataset_card(manifest), encoding="utf-8")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(args.dataset), path_in_repo="quest_sft.jsonl",
                    repo_id=args.repo_id, repo_type="dataset")
    if args.labels.exists():
        api.upload_file(path_or_fileobj=str(args.labels), path_in_repo="labeled.json",
                        repo_id=args.repo_id, repo_type="dataset")
    commit = api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                             repo_id=args.repo_id, repo_type="dataset",
                             commit_message="Publish Hackathon Advisor quest-classification SFT dataset")
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_id", None) or str(commit)
    print(f"published dataset https://huggingface.co/datasets/{args.repo_id}")
    print(f"revision: {revision}")


if __name__ == "__main__":
    main()
