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
        "role": "Embedding retriever",
        "model": "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
        "params_b": 0.30,
        "status": "deployed",
        "runtime": "Modal-built llama.cpp GGUF index + runtime llama.cpp query embeddings",
    },
    {
        "role": "Voice input",
        "model": "nvidia/nemotron-speech-streaming-en-0.6b",
        "params_b": 0.60,
        "status": "deployed",
        "runtime": "ZeroGPU + NVIDIA NeMo ASR",
    },
]


BADGE_LEDGER = [
    {
        "name": "Off the Grid",
        "status": "ready",
        "evidence": "Runtime uses checked-in project vectors and local llama.cpp query embeddings; no proprietary inference API.",
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
        "status": "ready",
        "evidence": "Retrieval uses an EmbeddingGemma GGUF index built by llama.cpp on Modal and query embeddings computed through llama.cpp at runtime.",
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


def prize_ledger(
    runtime: dict[str, Any],
    index_metadata: dict[str, Any] | None = None,
    voice_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_params = round(sum(float(item["params_b"]) for item in MODEL_STACK), 2)
    largest = max(MODEL_STACK, key=lambda item: float(item["params_b"]))
    return {
        "runtime": runtime,
        "retrieval_index": index_metadata or {},
        "voice": voice_metadata or {},
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
