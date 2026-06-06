from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from hackathon_advisor.lora_dataset import BASE_MODEL, build_lora_dataset_jsonl


TRAINING_RECIPE_SCHEMA_VERSION = 1
TRAINING_KIT_FILENAME = "hackathon-advisor-lora-training-kit.zip"
ADAPTER_REPO = "build-small-hackathon/hackathon-advisor-minicpm5-lora"


def parse_lora_dataset_jsonl(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError("LoRA dataset is empty")
    manifest = records[0]
    examples = records[1:]
    if manifest.get("type") != "lora_sft_manifest":
        raise ValueError("first LoRA dataset row must be a lora_sft_manifest")
    for index, example in enumerate(examples, start=1):
        if example.get("type") != "lora_sft_example":
            raise ValueError(f"record {index} is not a lora_sft_example")
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"record {index} has no chat messages")
        for message in messages:
            if not isinstance(message, dict) or not message.get("role") or not message.get("content"):
                raise ValueError(f"record {index} has an invalid chat message")
    return manifest, examples


def build_training_recipe(
    dataset_manifest: dict[str, Any],
    example_count: int,
    *,
    max_steps: int = 120,
) -> dict[str, Any]:
    return {
        "type": "lora_training_recipe",
        "schema_version": TRAINING_RECIPE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": dataset_manifest.get("base_model") or BASE_MODEL,
        "adapter_repo": ADAPTER_REPO,
        "adapter_task": dataset_manifest.get("adapter_task") or "hackathon_advisor_tool_call_and_voice",
        "dataset_format": dataset_manifest.get("format") or "chat-jsonl",
        "example_count": example_count,
        "method": "LoRA SFT",
        "runtime": "transformers + PEFT",
        "max_steps": max_steps,
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "learning_rate": 0.0002,
        "max_seq_length": 1024,
        "target_modules": "discovered torch.nn.Linear module suffixes at training runtime",
        "publish_status": "not-published",
    }


def build_training_model_card(recipe: dict[str, Any], dataset_manifest: dict[str, Any], ledger: dict[str, Any]) -> str:
    badges = ledger.get("badges") if isinstance(ledger.get("badges"), list) else []
    lines = [
        "# Hackathon Advisor MiniCPM5 LoRA",
        "",
        "This is the prepared model card for the Well-Tuned adapter candidate. The checked-in app can export the SFT "
        "dataset and this training kit; the adapter is not claimed as published until a real Hub repo and training run "
        "exist.",
        "",
        "## Recipe",
        "",
        f"- Base model: `{recipe['base_model']}`",
        f"- Adapter repo target: `{recipe['adapter_repo']}`",
        f"- Task: `{recipe['adapter_task']}`",
        f"- Method: {recipe['method']}",
        f"- Examples: {recipe['example_count']}",
        f"- Max steps: {recipe['max_steps']}",
        f"- LoRA rank: {recipe['rank']}",
        f"- LoRA alpha: {recipe['alpha']}",
        "",
        "## Dataset Provenance",
        "",
        f"- Source: {dataset_manifest.get('source', 'exact_session_trace')}",
        f"- Turn count: {dataset_manifest.get('turn_count', 0)}",
        f"- Index digest: `{(dataset_manifest.get('index') or {}).get('snapshot_digest', '')}`",
        "",
        "## Badge Ledger",
        "",
    ]
    for badge in badges:
        if not isinstance(badge, dict):
            continue
        lines.append(f"- {badge.get('name')}: {badge.get('status')} - {badge.get('evidence')}")
    return "\n".join(lines).rstrip() + "\n"


def build_train_command(recipe: dict[str, Any]) -> str:
    return (
        "pip install -e '.[train]'\n"
        "python scripts/train_minicpm_lora.py \\\n"
        "  --dataset lora-sft.jsonl \\\n"
        "  --output-dir ./minicpm5-hackathon-advisor-lora \\\n"
        f"  --base-model {recipe['base_model']} \\\n"
        f"  --max-steps {recipe['max_steps']}\n"
    )


def build_lora_training_kit_zip(session: dict[str, Any], metadata: dict[str, Any], ledger: dict[str, Any]) -> bytes:
    dataset_text = build_lora_dataset_jsonl(session, metadata)
    dataset_manifest, examples = parse_lora_dataset_jsonl(dataset_text)
    recipe = build_training_recipe(dataset_manifest, len(examples))
    model_card = build_training_model_card(recipe, dataset_manifest, ledger)
    command = build_train_command(recipe)
    files = {
        "lora-sft.jsonl": dataset_text,
        "training-recipe.json": json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "adapter-model-card.md": model_card,
        "train-command.txt": command,
        "README.md": _kit_readme(recipe),
    }
    manifest = {
        "type": "lora_training_kit_manifest",
        "schema_version": TRAINING_RECIPE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(files),
        "files": list(files),
        "example_count": len(examples),
        "adapter_repo": recipe["adapter_repo"],
        "publish_status": recipe["publish_status"],
    }

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        for filename, content in files.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


def write_lora_training_dry_run(dataset_path: Path, output_dir: Path, *, max_steps: int = 120) -> dict[str, Any]:
    dataset_text = dataset_path.read_text(encoding="utf-8")
    dataset_manifest, examples = parse_lora_dataset_jsonl(dataset_text)
    recipe = build_training_recipe(dataset_manifest, len(examples), max_steps=max_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training-recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "train-command.txt").write_text(build_train_command(recipe), encoding="utf-8")
    return recipe


def _kit_readme(recipe: dict[str, Any]) -> str:
    return (
        "# Hackathon Advisor LoRA Training Kit\n\n"
        "This kit prepares the Well-Tuned path without claiming the adapter has already been trained or published.\n\n"
        "Run `train-command.txt` in an environment with the `train` extra installed. The training script validates the "
        "dataset, loads the base model, discovers LoRA target modules from the loaded model, and saves the PEFT adapter.\n\n"
        f"Adapter repo target: `{recipe['adapter_repo']}`\n"
    )
