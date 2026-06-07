#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "hackathon-advisor-llama-index"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub>=0.36,<1",
        "llama-cpp-python>=0.3.26,<1",
    )
    .add_local_python_source("hackathon_advisor", copy=True)
    .add_local_python_source("scripts", copy=True)
)


@app.function(image=image, cpu=4.0, memory=4096, timeout=1800)
def build_project_index_remote(
    project_snapshot: dict[str, Any],
    model_repo: str,
    model_file: str,
) -> dict[str, Any]:
    from pathlib import Path
    import tempfile

    from scripts.build_project_index import build_payload

    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "projects.json"
        project_path.write_text(
            json.dumps(project_snapshot, ensure_ascii=False),
            encoding="utf-8",
        )
        return build_payload(
            project_path,
            model_repo=model_repo,
            model_file=model_file,
            build_source="modal remote function",
            builder="scripts/modal_build_project_index.py",
            modal_app=APP_NAME,
        )


@app.local_entrypoint()
def main(
    projects: str = "data/projects.json",
    out: str = "data/project_index.json",
    model_repo: str = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF",
    model_file: str = "embeddinggemma-300M-qat-Q4_0.gguf",
) -> None:
    project_snapshot = json.loads(Path(projects).read_text(encoding="utf-8"))
    payload = build_project_index_remote.remote(project_snapshot, model_repo, model_file)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "wrote "
        f"{payload['document_count']} docs, {payload['embedding']['dimensions']} dims "
        f"to {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the llama.cpp embedding index on Modal.")
    parser.add_argument("--projects", default="data/projects.json")
    parser.add_argument("--out", default="data/project_index.json")
    parser.add_argument("--model-repo", default="ggml-org/embeddinggemma-300M-qat-q4_0-GGUF")
    parser.add_argument("--model-file", default="embeddinggemma-300M-qat-Q4_0.gguf")
    args = parser.parse_args()
    with app.run():
        payload = build_project_index_remote.remote(
            json.loads(Path(args.projects).read_text(encoding="utf-8")),
            args.model_repo,
            args.model_file,
        )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "wrote "
        f"{payload['document_count']} docs, {payload['embedding']['dimensions']} dims "
        f"to {output}"
    )
