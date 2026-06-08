#!/usr/bin/env python3
"""Modal wiring for the project index build.

The user-facing entrypoint is `scripts/build_project_index.py --location modal`,
which calls `run_remote_build` below. The shared embedding logic lives in
`scripts.build_project_index.build_payload`; this module only owns the Modal
app/image/remote-function definitions. `modal run scripts/modal_build_project_index.py`
also works for callers who prefer the Modal CLI directly.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hackathon_advisor.data import DEFAULT_EMBEDDING_MODEL_FILE, DEFAULT_EMBEDDING_MODEL_REPO
from hackathon_advisor.llama_embedding import DEFAULT_N_CTX

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
    model_path: str = "",
    n_ctx: int = DEFAULT_N_CTX,
    n_threads: int | None = None,
) -> dict[str, Any]:
    import tempfile
    from pathlib import Path

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
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            build_source="modal remote function",
            builder="scripts/modal_build_project_index.py",
            modal_app=APP_NAME,
        )


def run_remote_build(
    projects_path: Path,
    *,
    model_repo: str = DEFAULT_EMBEDDING_MODEL_REPO,
    model_file: str = DEFAULT_EMBEDDING_MODEL_FILE,
    model_path: str = "",
    n_ctx: int = DEFAULT_N_CTX,
    n_threads: int | None = None,
) -> dict[str, Any]:
    """Build the index on Modal and return the payload.

    Used by `scripts/build_project_index.py --location modal`, which runs as a plain
    Python process, so this opens its own ephemeral Modal app context.
    """
    project_snapshot = json.loads(projects_path.read_text(encoding="utf-8"))
    with app.run():
        return build_project_index_remote.remote(
            project_snapshot,
            model_repo,
            model_file,
            model_path,
            n_ctx,
            n_threads,
        )


@app.local_entrypoint()
def main(
    projects: str = "data/projects.json",
    out: str = "data/project_index.json",
    model_repo: str = DEFAULT_EMBEDDING_MODEL_REPO,
    model_file: str = DEFAULT_EMBEDDING_MODEL_FILE,
) -> None:
    # Runs under `modal run`, which already manages the app context.
    from scripts.build_project_index import write_payload

    payload = build_project_index_remote.remote(
        json.loads(Path(projects).read_text(encoding="utf-8")),
        model_repo,
        model_file,
    )
    write_payload(Path(out), payload)
