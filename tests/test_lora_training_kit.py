import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

from tests.helpers import load_test_index
from zipfile import ZipFile

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.demo_rehearsal import build_demo_rehearsal
from hackathon_advisor.lora_dataset import build_lora_dataset_jsonl
from hackathon_advisor.lora_training_kit import (
    build_lora_training_kit_zip,
    parse_lora_dataset_jsonl,
)
from hackathon_advisor.prize_ledger import prize_ledger
from hackathon_advisor.trace_export import trace_metadata


def test_lora_training_kit_contains_recipe_and_model_card() -> None:
    index = load_test_index()
    engine = AdvisorEngine(index)
    metadata = {
        **trace_metadata(index),
        "project_count": len(index.projects),
    }
    demo = build_demo_rehearsal(engine)
    content = build_lora_training_kit_zip(
        demo["session"],
        metadata,
        prize_ledger(engine.runtime_status()),
    )

    with ZipFile(BytesIO(content)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        recipe = json.loads(archive.read("training-recipe.json"))
        model_card = archive.read("adapter-model-card.md").decode("utf-8")
        command = archive.read("train-command.txt").decode("utf-8")

    assert names == {
        "manifest.json",
        "lora-sft.jsonl",
        "training-recipe.json",
        "adapter-model-card.md",
        "train-command.txt",
        "README.md",
    }
    assert manifest["type"] == "lora_training_kit_manifest"
    assert manifest["publish_status"] == "published"
    assert recipe["base_model"] == "openbmb/MiniCPM5-1B"
    assert recipe["adapter_repo"] == "build-small-hackathon/hackathon-advisor-minicpm5-lora"
    assert recipe["example_count"] == manifest["example_count"]
    assert "PEFT LoRA adapter is trained" in model_card
    assert "scripts/train_minicpm_lora.py" in command
    assert "--push-to-hub" in command
    assert "--hub-repo-id build-small-hackathon/hackathon-advisor-minicpm5-lora" in command


def test_parse_lora_dataset_jsonl_rejects_empty_payload() -> None:
    try:
        parse_lora_dataset_jsonl("")
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty dataset should be rejected")


def test_train_minicpm_lora_dry_run_writes_recipe(tmp_path: Path) -> None:
    index = load_test_index()
    engine = AdvisorEngine(index)
    metadata = {
        **trace_metadata(index),
        "project_count": len(index.projects),
    }
    dataset_path = tmp_path / "lora-sft.jsonl"
    output_dir = tmp_path / "dry-run"
    dataset_path.write_text(
        build_lora_dataset_jsonl(build_demo_rehearsal(engine)["session"], metadata),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_minicpm_lora.py",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            "7",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    recipe = json.loads((output_dir / "training-recipe.json").read_text(encoding="utf-8"))

    assert "dry-run ok" in result.stdout
    assert recipe["example_count"] > 0
    assert recipe["max_steps"] == 7
    assert (output_dir / "train-command.txt").is_file()
