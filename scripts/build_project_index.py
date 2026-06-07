#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hackathon_advisor.data import (
    DEFAULT_EMBEDDING_MODEL_FILE,
    DEFAULT_EMBEDDING_MODEL_REPO,
    Project,
    build_index_payload,
)
from hackathon_advisor.llama_embedding import DEFAULT_N_CTX, LlamaCppEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the offline project retrieval index with llama.cpp embeddings."
    )
    parser.add_argument("--projects", default="data/projects.json")
    parser.add_argument("--out", default="data/project_index.json")
    parser.add_argument("--model-repo", default=DEFAULT_EMBEDDING_MODEL_REPO)
    parser.add_argument("--model-file", default=DEFAULT_EMBEDDING_MODEL_FILE)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    parser.add_argument("--n-threads", type=int, default=0)
    args = parser.parse_args()

    payload = build_payload(
        Path(args.projects),
        model_repo=args.model_repo,
        model_file=args.model_file,
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads or None,
        build_source="local",
        builder="scripts/build_project_index.py",
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "wrote "
        f"{payload['document_count']} docs, {payload['embedding']['dimensions']} dims "
        f"to {output}"
    )


def build_payload(
    project_path: Path,
    *,
    model_repo: str,
    model_file: str,
    model_path: str = "",
    n_ctx: int = DEFAULT_N_CTX,
    n_threads: int | None = None,
    build_source: str,
    builder: str,
    modal_app: str = "",
) -> dict:
    data = json.loads(project_path.read_text(encoding="utf-8"))
    projects = [Project.from_dict(item) for item in data["projects"]]
    embedder = LlamaCppEmbedder(
        model_repo=model_repo,
        model_file=model_file,
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )
    embeddings = [embedder.embed(project.searchable_text) for project in projects]
    metadata = {
        "model_repo": model_repo,
        "model_file": model_file,
        "build_source": build_source,
        "builder": builder,
        "llama_cpp_python_version": importlib.metadata.version("llama-cpp-python"),
        "n_ctx": n_ctx,
    }
    if modal_app:
        metadata["modal_app"] = modal_app
    return build_index_payload(
        projects=projects,
        snapshot_generated_at=str(data.get("generated_at") or ""),
        source=str(data.get("source") or ""),
        embeddings=embeddings,
        embedding_metadata=metadata,
    )


if __name__ == "__main__":
    main()
