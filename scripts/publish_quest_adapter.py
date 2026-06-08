#!/usr/bin/env python3
"""Publish the trained quest-classification LoRA adapter to the Hub.

Uploads artifacts/quest-lora/ to a model repo and prints the commit revision to pin
in hackathon_advisor.quest_analysis (ADVISOR_QUEST_ADAPTER_REVISION).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "build-small-hackathon/hackathon-advisor-quest-minicpm5-lora"


def model_card(recipe: dict, eval_report: dict) -> str:
    n = eval_report.get("n")
    exact = eval_report.get("quest_set_exact")
    f1 = eval_report.get("f1")
    return "\n".join(
        [
            "---",
            "base_model: openbmb/MiniCPM5-1B",
            "library_name: peft",
            "datasets:",
            "- build-small-hackathon/hackathon-advisor-quest-dataset",
            "tags:",
            "- lora",
            "- hackathon-advisor",
            "- quest-classification",
            "license: apache-2.0",
            "---",
            "",
            "# Hackathon Advisor — Quest Classification LoRA (MiniCPM5-1B)",
            "",
            "PEFT LoRA adapter that classifies a Build Small Hackathon project against 13 judging",
            "dimensions (6 merit badges + 2 tracks + 5 sponsor/special awards) from a two-segment",
            "README + app-file prompt, emitting strict JSON:",
            "",
            "```json",
            '{"matches":[{"quest":"...","confidence":0.0,"evidence":"...","source":"readme|app_file"}]}',
            "```",
            "",
            "Load it in the deployed Space by setting `ADVISOR_QUEST_ADAPTER_ID` to this repo.",
            "The backend revalidates every dashboard refresh and will not swap on schema failure.",
            "",
            "## Recipe",
            "",
            f"- Base model: `{recipe.get('base_model')}`",
            f"- Task: `{recipe.get('adapter_task')}`",
            f"- Method: {recipe.get('method')}",
            f"- Examples: {recipe.get('example_count')}",
            f"- Epochs: {recipe.get('epochs')}",
            f"- LoRA rank/alpha/dropout: {recipe.get('rank')}/{recipe.get('alpha')}/{recipe.get('dropout')}",
            f"- Max seq length: {recipe.get('max_seq_length')}",
            f"- GPU: {recipe.get('gpu')}",
            "",
            "## Dataset",
            "",
            "[`build-small-hackathon/hackathon-advisor-quest-dataset`]"
            "(https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-quest-dataset)"
            " — 156 chat-JSONL examples built from real `build-small-hackathon` Spaces: 108 teacher-",
            "labelled + adversarially-verified projects plus targeted augmentations (app-only,",
            "readme-only / missing app file, README↔app contradictions, empty matches, noisy",
            "metadata). All 13 quests covered.",
            "",
            f"## Full-dataset eval at training time: quest-set exact match {exact}/{n}, micro-F1 {f1}.",
            "",
            "Evaluated by reproducing the gold quest set for every example in the training dataset",
            "(the dataset is the spec — it is built from the real `build-small-hackathon` projects).",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the quest LoRA adapter.")
    parser.add_argument("--adapter-dir", default="artifacts/quest-lora", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    args = parser.parse_args()

    recipe_path = args.adapter_dir / "training-recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8")) if recipe_path.exists() else {}
    eval_path = args.adapter_dir / "self-eval.json"
    eval_report = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    (args.adapter_dir / "README.md").write_text(model_card(recipe, eval_report), encoding="utf-8")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(args.adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Publish Hackathon Advisor quest-classification MiniCPM5 LoRA",
        ignore_patterns=["self-eval.json"],
    )
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_id", None) or str(commit)
    print(f"published {args.repo_id}")
    print(f"revision: {revision}")


if __name__ == "__main__":
    main()
