from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import crawl_hf_spaces


def test_readme_frontmatter_extracts_app_file() -> None:
    frontmatter = crawl_hf_spaces.readme_frontmatter(
        """---
title: Tiny Demo
app_file: "src/app.py" # main entrypoint
tags:
  - gradio
---
# Tiny Demo
"""
    )

    assert frontmatter["app_file"] == "src/app.py"


def test_validate_app_file_rejects_untrusted_paths() -> None:
    with pytest.raises(RuntimeError, match="invalid app_file path"):
        crawl_hf_spaces.validate_app_file("../app.py", space_id="build-small-hackathon/demo")


def test_project_from_space_downloads_frontmatter_app_file(monkeypatch) -> None:
    downloads = {
        ("build-small-hackathon/demo", "README.md"): "---\napp_file: app.py\n---\n# Demo\nREADME body evidence.\n",
        ("build-small-hackathon/demo", "app.py"): "import gradio as gr\ngr.Textbox(label='Idea')\n",
    }

    def fake_download(repo_id: str, filename: str) -> str:
        return downloads[(repo_id, filename)]

    monkeypatch.setattr(crawl_hf_spaces, "download_repo_text", fake_download)
    space = SimpleNamespace(
        id="build-small-hackathon/demo",
        card_data={"title": "Demo", "short_description": "Advisor demo", "sdk": "gradio"},
        siblings=[
            SimpleNamespace(rfilename="README.md"),
            SimpleNamespace(rfilename="app.py"),
        ],
        tags=["gradio"],
        models=[],
        datasets=[],
        likes=3,
        created_at=None,
        last_modified=None,
        host="https://example.test",
        private=False,
    )

    project = crawl_hf_spaces.project_from_space(space)

    assert project["app_file"] == "app.py"
    assert project["readme_body"] == "# Demo\nREADME body evidence."
    assert project["app_file_source"] == "import gradio as gr\ngr.Textbox(label='Idea')\n"
    assert "gr.Textbox" in project["app_file_embedding_text"]
    assert "Idea" in project["app_file_embedding_text"]


def test_project_from_space_tolerates_stale_frontmatter_app_file(monkeypatch) -> None:
    downloads = {
        ("build-small-hackathon/demo", "README.md"): "---\napp_file: app.py\n---\n",
    }

    def fake_download(repo_id: str, filename: str) -> str:
        if (repo_id, filename) in downloads:
            return downloads[(repo_id, filename)]
        raise crawl_hf_spaces.EntryNotFoundError("missing file")

    monkeypatch.setattr(crawl_hf_spaces, "download_repo_text", fake_download)
    space = SimpleNamespace(
        id="build-small-hackathon/demo",
        card_data={"title": "Demo", "short_description": "Advisor demo", "sdk": "gradio"},
        siblings=[
            SimpleNamespace(rfilename="README.md"),
            SimpleNamespace(rfilename="app.py"),
        ],
        tags=["gradio"],
        models=[],
        datasets=[],
        likes=3,
        created_at=None,
        last_modified=None,
        host="https://example.test",
        private=False,
    )

    project = crawl_hf_spaces.project_from_space(space)

    assert project["app_file"] == "app.py"
    assert project["readme_body"] == ""
    assert project["app_file_source"] == ""
    assert project["app_file_embedding_text"] == ""


def test_download_repo_text_uses_bounded_raw_request(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, int]]] = []

    class Response:
        status_code = 200
        encoding = ""
        text = "hello"

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, *, timeout: tuple[int, int]):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(crawl_hf_spaces.requests, "get", fake_get)

    text = crawl_hf_spaces.download_repo_text("build-small-hackathon/demo space", "src/app file.py")

    assert text == "hello"
    assert calls == [
        (
            "https://huggingface.co/spaces/build-small-hackathon/demo%20space/resolve/main/src/app%20file.py",
            crawl_hf_spaces.DOWNLOAD_TIMEOUT_SECONDS,
        )
    ]


def test_crawl_workers_rejects_non_integer_env(monkeypatch) -> None:
    monkeypatch.setenv("ADVISOR_CRAWL_WORKERS", "many")

    with pytest.raises(RuntimeError, match="ADVISOR_CRAWL_WORKERS"):
        crawl_hf_spaces.crawl_workers()
