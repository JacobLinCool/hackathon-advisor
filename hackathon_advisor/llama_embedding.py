from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import os
from typing import Any

from hackathon_advisor.data import (
    DEFAULT_EMBEDDING_MODEL_FILE,
    DEFAULT_EMBEDDING_MODEL_REPO,
)


TRUE_VALUES = {"1", "true", "yes", "on"}
DEFAULT_N_CTX = 512


class LlamaCppEmbedder:
    def __init__(
        self,
        *,
        model_repo: str = DEFAULT_EMBEDDING_MODEL_REPO,
        model_file: str = DEFAULT_EMBEDDING_MODEL_FILE,
        model_path: str = "",
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_repo = model_repo.strip() or DEFAULT_EMBEDDING_MODEL_REPO
        self.model_file = model_file.strip() or DEFAULT_EMBEDDING_MODEL_FILE
        self.model_path = model_path.strip()
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._model = None

    def __call__(self, text: str) -> Sequence[float]:
        return self.embed(text)

    def embed(self, text: str) -> Sequence[float]:
        model = self._ensure_model()
        return model.embed(text, normalize=True)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from huggingface_hub import hf_hub_download
        from llama_cpp import LLAMA_POOLING_TYPE_MEAN, Llama

        model_path = self.model_path
        if not model_path:
            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                repo_type="model",
            )
        if not Path(model_path).is_file():
            raise RuntimeError(f"llama.cpp embedding model was not found: {model_path}")
        self._model = Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=LLAMA_POOLING_TYPE_MEAN,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )
        return self._model


def create_llama_cpp_embedder(metadata: dict[str, Any]) -> LlamaCppEmbedder:
    return LlamaCppEmbedder(
        model_repo=os.environ.get(
            "ADVISOR_EMBEDDING_MODEL_REPO",
            str(metadata.get("model_repo") or DEFAULT_EMBEDDING_MODEL_REPO),
        ),
        model_file=os.environ.get(
            "ADVISOR_EMBEDDING_MODEL_FILE",
            str(metadata.get("model_file") or DEFAULT_EMBEDDING_MODEL_FILE),
        ),
        model_path=os.environ.get("ADVISOR_EMBEDDING_MODEL_PATH", ""),
        n_ctx=_int_env("ADVISOR_EMBEDDING_N_CTX", DEFAULT_N_CTX),
        n_threads=_optional_int_env("ADVISOR_EMBEDDING_THREADS"),
        n_gpu_layers=_int_env("ADVISOR_EMBEDDING_GPU_LAYERS", 0),
        verbose=os.environ.get("ADVISOR_EMBEDDING_VERBOSE", "").strip().lower() in TRUE_VALUES,
    )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer.")
    return value


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value
