from __future__ import annotations

from typing import Any


MODEL_STACK = [
    {
        "role": "LLM brain",
        "model": "openbmb/MiniCPM5-1B",
        "params_b": 1.08,
        "status": "optional runtime adapter",
        "runtime": "transformers / GGUF-ready",
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
        "status": "training-kit-ready",
        "evidence": "LoRA SFT dataset and training kit export are generated from exact session traces; adapter publication remains a separate build milestone.",
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
        "status": "export-ready",
        "endpoint": "/api/lora-training-kit.zip",
        "format": "zip",
        "base_model": "openbmb/MiniCPM5-1B",
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
