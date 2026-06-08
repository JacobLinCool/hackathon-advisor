#!/usr/bin/env python3
from __future__ import annotations

import argparse
from hashlib import sha256
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
    parser.add_argument(
        "--location",
        choices=("local", "modal"),
        default="local",
        help="Where to run the embedding build (default: local).",
    )
    parser.add_argument("--projects", default="data/projects.json")
    parser.add_argument("--out", default="data/project_index.json")
    parser.add_argument("--model-repo", default=DEFAULT_EMBEDDING_MODEL_REPO)
    parser.add_argument("--model-file", default=DEFAULT_EMBEDDING_MODEL_FILE)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--build-source", default="local")
    parser.add_argument("--builder", default="scripts/build_project_index.py")
    parser.add_argument("--reuse-index", default="")
    args = parser.parse_args()

    if args.location == "modal":
        if args.reuse_index:
            parser.error("--reuse-index is not supported with --location modal")
        # Imported lazily so the local path never requires the `modal` package.
        from scripts.modal_build_project_index import run_remote_build

        payload = run_remote_build(
            Path(args.projects),
            model_repo=args.model_repo,
            model_file=args.model_file,
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads or None,
        )
    else:
        payload = build_payload(
            Path(args.projects),
            model_repo=args.model_repo,
            model_file=args.model_file,
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads or None,
            build_source=args.build_source,
            builder=args.builder,
            reuse_index_path=Path(args.reuse_index) if args.reuse_index else None,
        )
    write_payload(Path(args.out), payload)


def write_payload(output: Path, payload: dict) -> None:
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
    reuse_index_path: Path | None = None,
) -> dict:
    data = json.loads(project_path.read_text(encoding="utf-8"))
    projects = [Project.from_dict(item) for item in data["projects"]]
    print(f"loaded {len(projects)} projects from {project_path}", flush=True)
    reusable_vectors = load_reusable_vectors(
        reuse_index_path,
        model_repo=model_repo,
        model_file=model_file,
        n_ctx=n_ctx,
    )
    if reusable_vectors:
        print(f"loaded {len(reusable_vectors)} reusable vectors from {reuse_index_path}", flush=True)
    print(
        "embedding projects with "
        f"{model_repo}/{model_file}; first vector may download and load the GGUF model",
        flush=True,
    )
    embeddings = []
    embedder = None
    reused_count = 0
    embedded_count = 0
    for index, project in enumerate(projects, start=1):
        digest = sha256(project.searchable_text.encode("utf-8")).hexdigest()
        reusable_vector = reusable_vectors.get((project.id, digest))
        if reusable_vector is not None:
            embeddings.append(reusable_vector)
            reused_count += 1
        else:
            if embedder is None:
                embedder = LlamaCppEmbedder(
                    model_repo=model_repo,
                    model_file=model_file,
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    verbose=False,
                )
            embeddings.append(embedder.embed(project.searchable_text))
            embedded_count += 1
        if index == 1 or index % 10 == 0 or index == len(projects):
            print(
                f"indexed {index}/{len(projects)} projects "
                f"(reused={reused_count}, embedded={embedded_count})",
                flush=True,
            )
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


def load_reusable_vectors(
    reuse_index_path: Path | None,
    *,
    model_repo: str,
    model_file: str,
    n_ctx: int,
) -> dict[tuple[str, str], list[float]]:
    if reuse_index_path is None:
        return {}
    payload = json.loads(reuse_index_path.read_text(encoding="utf-8"))
    embedding = payload.get("embedding")
    if not isinstance(embedding, dict):
        print(f"skipping reusable vectors from {reuse_index_path}: missing embedding metadata", flush=True)
        return {}
    expected = {
        "model_repo": model_repo,
        "model_file": model_file,
        "n_ctx": n_ctx,
    }
    try:
        actual_n_ctx = int(embedding.get("n_ctx") or 0)
    except (TypeError, ValueError):
        actual_n_ctx = 0
    actual = {
        "model_repo": str(embedding.get("model_repo") or ""),
        "model_file": str(embedding.get("model_file") or ""),
        "n_ctx": actual_n_ctx,
    }
    if actual != expected:
        print(
            f"skipping reusable vectors from {reuse_index_path}: "
            f"embedding config changed from {actual} to {expected}",
            flush=True,
        )
        return {}
    documents = payload.get("documents")
    if not isinstance(documents, list):
        return {}
    reusable: dict[tuple[str, str], list[float]] = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        project_id = str(document.get("project_id") or "")
        text_digest = str(document.get("text_digest") or "")
        vector = document.get("vector")
        if project_id and text_digest and isinstance(vector, list) and vector:
            reusable[(project_id, text_digest)] = [float(value) for value in vector]
    return reusable


if __name__ == "__main__":
    main()
