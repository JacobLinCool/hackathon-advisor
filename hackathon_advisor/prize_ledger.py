from __future__ import annotations

from typing import Any


MODEL_STACK = [
    {
        "role": "LLM brain",
        "model": "openbmb/MiniCPM5-1B",
        "adapter_repo": "build-small-hackathon/hackathon-advisor-minicpm5-lora",
        "params_b": 1.08,
        "status": "deployed adapter target",
        "runtime": "ZeroGPU + transformers + PEFT",
    },
    {
        "role": "Retriever",
        "model": "offline TF-IDF snapshot",
        "params_b": 0.0,
        "status": "deployed",
        "runtime": "local sparse index",
    },
    {
        "role": "Planned embedder",
        "model": "google/embeddinggemma-300m",
        "params_b": 0.30,
        "status": "documented build path",
        "runtime": "sentence-transformers / llama.cpp",
    },
    {
        "role": "Voice bonus",
        "model": "nvidia/nemotron-speech-streaming-en-0.6b",
        "params_b": 0.60,
        "status": "deferred bonus",
        "runtime": "batch ASR",
    },
]


BADGE_LEDGER = [
    {
        "name": "Off the Grid",
        "status": "ready",
        "evidence": "Runtime uses a checked-in snapshot and local search; no proprietary inference API.",
    },
    {
        "name": "Off-Brand",
        "status": "ready",
        "evidence": "Custom gr.Server frontend renders the agent as The Unwritten Almanac.",
    },
    {
        "name": "Sharing is Caring",
        "status": "ready",
        "evidence": "JSONL trace export and checked-in sample trace are published with the Space.",
    },
    {
        "name": "Field Notes",
        "status": "ready",
        "evidence": "Field Notes markdown export is generated from exact session state.",
    },
    {
        "name": "Tiny Titan",
        "status": "eligible",
        "evidence": "Documented stack stays under 4B parameters; largest model is MiniCPM5-1B.",
    },
    {
        "name": "Well-Tuned",
        "status": "ready",
        "evidence": "MiniCPM5 LoRA adapter target is published to the Hub and loaded by the ZeroGPU Transformers runtime.",
    },
    {
        "name": "Llama Champion",
        "status": "planned",
        "evidence": "MiniCPM5 GGUF and EmbeddingGemma GGUF paths are documented; runtime does not depend on them yet.",
    },
]


TRAINING_ARTIFACTS = [
    {
        "name": "MiniCPM5 LoRA SFT dataset",
        "status": "export-ready",
        "endpoint": "lora_dataset",
        "format": "chat-jsonl",
        "base_model": "openbmb/MiniCPM5-1B",
    },
    {
        "name": "MiniCPM5 LoRA training kit",
        "status": "published-recipe",
        "endpoint": "/api/lora-training-kit.zip",
        "format": "zip",
        "base_model": "openbmb/MiniCPM5-1B",
        "adapter_repo": "build-small-hackathon/hackathon-advisor-minicpm5-lora",
    }
]


def prize_ledger(runtime: dict[str, Any]) -> dict[str, Any]:
    total_params = round(sum(float(item["params_b"]) for item in MODEL_STACK), 2)
    largest = max(MODEL_STACK, key=lambda item: float(item["params_b"]))
    return {
        "runtime": runtime,
        "model_stack": MODEL_STACK,
        "total_params_b": total_params,
        "largest_model": {
            "model": largest["model"],
            "params_b": largest["params_b"],
        },
        "tiny_titan_limit_b": 4.0,
        "tiny_titan_eligible": total_params <= 4.0 and float(largest["params_b"]) <= 4.0,
        "badges": BADGE_LEDGER,
        "training_artifacts": TRAINING_ARTIFACTS,
    }
