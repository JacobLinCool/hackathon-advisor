from __future__ import annotations

from collections.abc import Sequence
import atexit
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from typing import Any

from hackathon_advisor.data import (
    DEFAULT_EMBEDDING_MODEL_FILE,
    DEFAULT_EMBEDDING_MODEL_REPO,
)


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
DEFAULT_N_CTX = 2048


class LlamaCppEmbedder:
    def __init__(
        self,
        *,
        model_repo: str = DEFAULT_EMBEDDING_MODEL_REPO,
        model_file: str = DEFAULT_EMBEDDING_MODEL_FILE,
        model_path: str = "",
        n_ctx: int = DEFAULT_N_CTX,
        n_batch: int | None = None,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_repo = model_repo.strip() or DEFAULT_EMBEDDING_MODEL_REPO
        self.model_file = model_file.strip() or DEFAULT_EMBEDDING_MODEL_FILE
        self.model_path = model_path.strip()
        self.n_ctx = n_ctx
        self.n_batch = n_batch or n_ctx
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
            n_batch=self.n_batch,
            n_ubatch=self.n_batch,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )
        return self._model


class SubprocessLlamaCppEmbedder:
    def __init__(
        self,
        *,
        model_repo: str = DEFAULT_EMBEDDING_MODEL_REPO,
        model_file: str = DEFAULT_EMBEDDING_MODEL_FILE,
        model_path: str = "",
        n_ctx: int = DEFAULT_N_CTX,
        n_batch: int | None = None,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_repo = model_repo.strip() or DEFAULT_EMBEDDING_MODEL_REPO
        self.model_file = model_file.strip() or DEFAULT_EMBEDDING_MODEL_FILE
        self.model_path = model_path.strip()
        self.n_ctx = n_ctx
        self.n_batch = n_batch or n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        atexit.register(self.close)

    def __call__(self, text: str) -> Sequence[float]:
        return self.embed(text)

    def embed(self, text: str) -> Sequence[float]:
        with self._lock:
            process = self._ensure_process()
            self._request_id += 1
            request_id = self._request_id
            request = json.dumps({"id": request_id, "text": text}, ensure_ascii=False)
            try:
                assert process.stdin is not None
                assert process.stdout is not None
                process.stdin.write(f"{request}\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError) as error:
                self.close()
                raise RuntimeError("llama.cpp embedding worker stopped before returning a vector.") from error
            if not line:
                returncode = process.poll()
                self.close()
                detail = f" with exit code {returncode}" if returncode is not None else ""
                raise RuntimeError(f"llama.cpp embedding worker exited{detail}.")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError("llama.cpp embedding worker returned invalid JSON.") from error
            if response.get("id") != request_id:
                raise RuntimeError("llama.cpp embedding worker returned an out-of-order response.")
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            vector = response.get("vector")
            if not isinstance(vector, list):
                raise RuntimeError("llama.cpp embedding worker did not return a vector.")
            return vector

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", "hackathon_advisor.llama_embedding", "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if self.verbose else subprocess.DEVNULL,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        config = json.dumps(
            {
                "model_repo": self.model_repo,
                "model_file": self.model_file,
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_batch": self.n_batch,
                "n_threads": self.n_threads,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": self.verbose,
            },
            ensure_ascii=False,
        )
        assert self._process.stdin is not None
        self._process.stdin.write(f"{config}\n")
        self._process.stdin.flush()
        return self._process


def create_llama_cpp_embedder(metadata: dict[str, Any]) -> LlamaCppEmbedder | SubprocessLlamaCppEmbedder:
    embedder_cls = SubprocessLlamaCppEmbedder if _use_subprocess_embedder() else LlamaCppEmbedder
    return embedder_cls(
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
        n_batch=_optional_int_env("ADVISOR_EMBEDDING_BATCH"),
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


def _use_subprocess_embedder() -> bool:
    raw = os.environ.get("ADVISOR_EMBEDDING_SUBPROCESS", "").strip().lower()
    if raw in TRUE_VALUES:
        return True
    if raw in FALSE_VALUES:
        return False
    backend = os.environ.get("ADVISOR_MODEL_BACKEND", "").strip().lower()
    return platform.system() == "Darwin" and backend in {"minicpm", "minicpm-transformers"}


def _worker_loop() -> None:
    config_line = sys.stdin.readline()
    if not config_line:
        return
    embedder = LlamaCppEmbedder(**json.loads(config_line))
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        request_id = request.get("id")
        try:
            vector = list(embedder.embed(str(request.get("text") or "")))
            response = {"id": request_id, "vector": vector}
        except Exception as error:
            response = {"id": request_id, "error": str(error)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        _worker_loop()
    else:
        raise SystemExit("usage: python -m hackathon_advisor.llama_embedding --worker")
