#!/usr/bin/env python3
"""Train the MiniCPM5-1B quest-classification LoRA adapter on Modal.

The dataset (chat-JSONL produced by hackathon_advisor.quest_dataset) is sent to a
GPU container, fine-tuned with PEFT LoRA, self-evaluated on a held-out slice, and
the adapter is returned as a zip the local entrypoint unpacks under artifacts/.

Smoke test the GPU first:
    modal run scripts/modal_train_quest_lora.py::smoke
Train:
    modal run scripts/modal_train_quest_lora.py --dataset data/quest_sft.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import modal


APP_NAME = "hackathon-advisor-quest-lora"
BASE_MODEL = "openbmb/MiniCPM5-1B"
GPU = "A10G"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.55,<5",
        "peft>=0.13,<1",
        "accelerate>=1.0,<2",
        "huggingface-hub>=0.36,<1",
        "datasets>=3,<4",
        "sentencepiece>=0.2,<1",
    )
    .add_local_python_source("hackathon_advisor", copy=True)
)


@app.function(image=image, gpu=GPU, timeout=3600)
def smoke() -> dict:
    import torch

    return {
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }


@app.function(image=image, gpu=GPU, timeout=5400)
def train_remote(
    dataset_text: str,
    *,
    base_model: str = BASE_MODEL,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    learning_rate: float = 2e-4,
    epochs: float = 4.0,
    max_seq_length: int = 2560,
    eval_holdout: int = 10,
) -> dict:
    import io
    import json
    import os
    import random
    import zipfile

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from hackathon_advisor.quest_dataset import parse_quest_dataset_jsonl
    from hackathon_advisor.quest_taxonomy import normalize_match

    manifest, examples = parse_quest_dataset_jsonl(dataset_text)
    random.Random(42).shuffle(examples)  # representative holdout; keep edge cases mostly in train
    holdout = examples[-eval_holdout:] if eval_holdout and len(examples) > eval_holdout * 2 else []
    train_examples = examples[: len(examples) - len(holdout)] if holdout else examples
    print(f"examples: total={len(examples)} train={len(train_examples)} holdout={len(holdout)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    target_modules = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] not in {"lm_head", "embed_tokens"}
        }
    )
    if not target_modules:
        raise RuntimeError("no LoRA target modules discovered")
    print("LoRA targets:", target_modules, flush=True)

    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # required for gradient checkpointing over a frozen base

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def template(messages, *, add_generation_prompt):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

    def encode(example: dict) -> dict:
        # Build the sequence as the EXACT inference prompt (which includes the empty
        # <think></think> block emitted with enable_thinking=False) followed by the
        # strict-JSON completion and the <|im_end|> turn terminator. The prompt is
        # tokenized identically to inference so the model never sees a shifted context.
        messages = example["messages"]
        prompt_text = template(messages[:-1], add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text)["input_ids"]
        completion_ids = tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"] + [im_end_id]
        input_ids = (prompt_ids + completion_ids)[:max_seq_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = [encode(r) for r in rows]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attn, labels = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n)
            attn.append(b["attention_mask"] + [0] * n)
            labels.append(b["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels),
        }

    args = TrainingArguments(
        output_dir="/tmp/quest-lora",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        bf16=True,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=DS(train_examples), data_collator=collate)
    trainer.train()

    out = Path("/tmp/quest-lora-adapter")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "training-recipe.json").write_text(
        json.dumps(
            {
                "type": "lora_training_recipe",
                "base_model": base_model,
                "adapter_task": manifest.get("adapter_task"),
                "method": "LoRA SFT (completion-only loss)",
                "example_count": len(train_examples),
                "epochs": epochs,
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "max_seq_length": max_seq_length,
                "target_modules": target_modules,
                "gpu": GPU,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- self-eval on the held-out slice: does the adapter emit valid, schema-clean JSON? ---
    # Guarded so a generation hiccup never discards the trained adapter.
    import gc

    loss_history = [h.get("loss") for h in trainer.state.log_history if "loss" in h]
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = True
    try:
        model.gradient_checkpointing_disable()
    except Exception:  # noqa: BLE001
        pass
    model.eval()
    evals = []
    try:
        for ex in holdout:
            messages = ex["messages"]
            prompt_text = template(messages[:-1], add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            inputs.pop("token_type_ids", None)  # MiniCPM tokenizer emits it; generate() rejects it
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=384, do_sample=False, eos_token_id=im_end_id)
            text = tokenizer.decode(gen[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            ok, detail = False, ""
            try:
                payload = json.loads(text)
                for m in payload["matches"]:
                    normalize_match(m)
                ok = True
            except Exception as error:  # noqa: BLE001
                detail = f"{type(error).__name__}: {error}"
            evals.append({"project_id": ex.get("project_id", ""), "valid_json": ok, "detail": detail, "output": text[:400]})
    except Exception as error:  # noqa: BLE001 - keep the adapter even if eval breaks
        print(f"self-eval aborted: {type(error).__name__}: {error}", flush=True)
    valid = sum(1 for e in evals if e["valid_json"])
    print(f"self-eval: {valid}/{len(evals)} produced schema-valid JSON", flush=True)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out).as_posix())
    return {
        "adapter_zip": buffer.getvalue(),
        "eval": {"valid": valid, "total": len(evals), "samples": evals},
        "train_examples": len(train_examples),
        "loss_history": loss_history,
    }


@app.local_entrypoint()
def main(dataset: str = "data/quest_sft.jsonl", out_dir: str = "artifacts/quest-lora", epochs: float = 4.0) -> None:
    import io
    import json
    import zipfile

    dataset_text = Path(dataset).read_text(encoding="utf-8")
    result = train_remote.remote(dataset_text, epochs=epochs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(result["adapter_zip"])) as zf:
        zf.extractall(out)
    (out / "self-eval.json").write_text(json.dumps(result["eval"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"adapter written to {out}")
    print(f"self-eval: {result['eval']['valid']}/{result['eval']['total']} schema-valid JSON")
    print(f"loss history: {result['loss_history']}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train the quest-classification LoRA on Modal.")
    parser.add_argument("--dataset", default="data/quest_sft.jsonl")
    parser.add_argument("--out-dir", default="artifacts/quest-lora")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.parse_args()
    print("Run via: modal run scripts/modal_train_quest_lora.py --dataset data/quest_sft.jsonl")


if __name__ == "__main__":
    _cli()
