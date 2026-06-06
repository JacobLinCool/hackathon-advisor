from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hackathon_advisor.lora_training_kit import (
    build_training_recipe,
    parse_lora_dataset_jsonl,
    write_lora_training_dry_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or dry-run the Hackathon Advisor MiniCPM5 LoRA adapter.")
    parser.add_argument("--dataset", required=True, type=Path, help="LoRA SFT JSONL exported by the app.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for adapter or dry-run artifacts.")
    parser.add_argument("--base-model", default="openbmb/MiniCPM5-1B", help="Base model id.")
    parser.add_argument("--max-steps", default=120, type=int, help="Maximum training steps.")
    parser.add_argument("--rank", default=16, type=int, help="LoRA rank.")
    parser.add_argument("--alpha", default=32, type=int, help="LoRA alpha.")
    parser.add_argument("--dropout", default=0.05, type=float, help="LoRA dropout.")
    parser.add_argument("--learning-rate", default=2e-4, type=float, help="Learning rate.")
    parser.add_argument("--max-seq-length", default=1024, type=int, help="Maximum tokenized sequence length.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and write recipe without training.")
    args = parser.parse_args()

    if args.dry_run:
        recipe = write_lora_training_dry_run(args.dataset, args.output_dir, max_steps=args.max_steps)
        print(f"dry-run ok: {recipe['example_count']} examples -> {args.output_dir}")
        return

    train_lora(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_steps=args.max_steps,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
    )


def train_lora(
    *,
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    max_steps: int,
    rank: int,
    alpha: int,
    dropout: float,
    learning_rate: float,
    max_seq_length: int,
) -> None:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as error:
        raise SystemExit("Install training dependencies first: pip install -e '.[train]'") from error

    dataset_text = dataset_path.read_text(encoding="utf-8")
    dataset_manifest, examples = parse_lora_dataset_jsonl(dataset_text)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    target_modules = _discover_lora_targets(model, torch)
    if not target_modules:
        raise RuntimeError("No torch.nn.Linear modules were found for LoRA target discovery.")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    train_dataset = _ChatDataset(examples, tokenizer, max_seq_length)
    recipe = build_training_recipe(dataset_manifest, len(examples), max_steps=max_steps)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=5,
        save_steps=max(20, max_steps),
        save_total_limit=1,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_causal_lm_collate(tokenizer),
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training-recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _discover_lora_targets(model: Any, torch_module: Any) -> list[str]:
    targets: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch_module.nn.Linear):
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix in {"lm_head", "embed_tokens"}:
            continue
        targets.add(suffix)
    return sorted(targets)


class _ChatDataset:
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        messages = self.examples[index]["messages"]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
        )
        input_ids = encoded["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"],
            "labels": list(input_ids),
        }


def _causal_lm_collate(tokenizer: Any):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return tokenizer.pad(batch, padding=True, return_tensors="pt")

    return collate


if __name__ == "__main__":
    main()
