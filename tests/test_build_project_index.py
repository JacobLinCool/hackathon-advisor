from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from hackathon_advisor.data import Project
from scripts import build_project_index


def test_build_project_index_reuses_matching_digest_vectors(monkeypatch, tmp_path: Path) -> None:
    project_row = {
        "id": "build-small-hackathon/reused-project",
        "title": "Reused Project",
        "summary": "compact local model demo",
        "tags": ["gradio"],
        "models": [],
        "datasets": [],
        "likes": 0,
        "sdk": "gradio",
        "license": "",
        "created_at": "",
        "last_modified": "",
        "host": "",
        "url": "https://example.test",
    }
    project = Project.from_dict(project_row)
    digest = sha256(project.searchable_text.encode("utf-8")).hexdigest()
    project_path = tmp_path / "projects.json"
    reuse_path = tmp_path / "reuse.json"
    project_path.write_text(
        json.dumps({"generated_at": "2026-06-08T00:00:00+00:00", "source": "test", "projects": [project_row]}),
        encoding="utf-8",
    )
    reuse_path.write_text(
        json.dumps(
            {
                "embedding": {
                    "model_repo": "test/repo",
                    "model_file": "model.gguf",
                    "n_ctx": build_project_index.DEFAULT_N_CTX,
                },
                "documents": [{"project_id": project.id, "text_digest": digest, "vector": [1.0, 0.0, 0.0]}],
            }
        ),
        encoding="utf-8",
    )

    def fail_embedder(**_kwargs):
        raise AssertionError("matching digest vectors should not initialize llama.cpp")

    monkeypatch.setattr(build_project_index, "LlamaCppEmbedder", fail_embedder)

    payload = build_project_index.build_payload(
        project_path,
        model_repo="test/repo",
        model_file="model.gguf",
        build_source="test",
        builder="test",
        reuse_index_path=reuse_path,
    )

    assert payload["document_count"] == 1
    assert payload["documents"][0]["project_id"] == project.id
    assert payload["documents"][0]["text_digest"] == digest
    assert payload["documents"][0]["vector"] == [1.0, 0.0, 0.0]


def test_build_project_index_rejects_vectors_when_embedding_config_changes(monkeypatch, tmp_path: Path) -> None:
    project_row = {
        "id": "build-small-hackathon/rebuilt-project",
        "title": "Rebuilt Project",
        "summary": "compact local model demo",
        "tags": ["gradio"],
        "models": [],
        "datasets": [],
        "likes": 0,
        "sdk": "gradio",
        "license": "",
        "created_at": "",
        "last_modified": "",
        "host": "",
        "url": "https://example.test",
    }
    project = Project.from_dict(project_row)
    digest = sha256(project.searchable_text.encode("utf-8")).hexdigest()
    project_path = tmp_path / "projects.json"
    reuse_path = tmp_path / "reuse.json"
    project_path.write_text(
        json.dumps({"generated_at": "2026-06-08T00:00:00+00:00", "source": "test", "projects": [project_row]}),
        encoding="utf-8",
    )
    reuse_path.write_text(
        json.dumps(
            {
                "embedding": {
                    "model_repo": "test/repo",
                    "model_file": "model.gguf",
                    "n_ctx": 768,
                },
                "documents": [{"project_id": project.id, "text_digest": digest, "vector": [1.0, 0.0, 0.0]}],
            }
        ),
        encoding="utf-8",
    )

    class FakeEmbedder:
        def __init__(self, **kwargs) -> None:
            assert kwargs["n_ctx"] == 2048

        def embed(self, _text: str) -> list[float]:
            return [0.0, 1.0, 0.0]

    monkeypatch.setattr(build_project_index, "LlamaCppEmbedder", FakeEmbedder)

    payload = build_project_index.build_payload(
        project_path,
        model_repo="test/repo",
        model_file="model.gguf",
        n_ctx=2048,
        build_source="test",
        builder="test",
        reuse_index_path=reuse_path,
    )

    assert payload["document_count"] == 1
    assert payload["documents"][0]["project_id"] == project.id
    assert payload["documents"][0]["text_digest"] == digest
    assert payload["documents"][0]["vector"] == [0.0, 1.0, 0.0]
