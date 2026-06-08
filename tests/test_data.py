from pathlib import Path

from tests.helpers import load_test_index
import json

from hackathon_advisor.data import (
    Project,
    ProjectIndex,
    public_project_summary,
    public_project_title,
)


def test_project_index_searches_snapshot() -> None:
    index = load_test_index()

    hits = index.search("lullaby children audio", limit=3)

    assert hits
    assert hits[0].project.id.startswith("build-small-hackathon/")
    assert hits[0].page_number >= 1
    assert index.index_algorithm == "llama-cpp-embedding-v1"


def test_project_index_whitespace() -> None:
    index = load_test_index()

    items = index.find_whitespace(limit=3)

    assert len(items) == 3
    assert all(item.label for item in items)


def test_public_project_cards_hide_generic_submission_copy() -> None:
    assert public_project_title("My Build Small Hackathon") == "Untitled project"
    assert public_project_summary("This is my submission for the build-small-hackathon") == ""
    assert public_project_summary("Todo") == ""
    assert public_project_summary("Local-first personal knowledge agent") == "Local-first personal knowledge agent"

    project = Project(
        id="build-small-hackathon/my-build-small-hackathon",
        title="My Build Small Hackathon",
        summary="This is my submission for the build-small-hackathon",
        tags=(),
        models=(),
        datasets=(),
        likes=0,
        sdk="gradio",
        license="",
        created_at="",
        last_modified="",
        host="",
        url="https://example.test",
    )

    public = project.to_public_dict()

    assert public["title"] == "Untitled project"
    assert public["summary"] == ""


def test_searchable_text_includes_main_app_file_signals() -> None:
    project = Project(
        id="build-small-hackathon/idea-canvas",
        title="Idea Canvas",
        summary="",
        tags=("gradio",),
        models=(),
        datasets=(),
        likes=0,
        sdk="gradio",
        license="",
        created_at="",
        last_modified="",
        host="",
        url="https://example.test",
        app_file="app.py",
        app_file_embedding_text="score_idea\ngr.Textbox\nProject idea",
    )

    searchable = project.searchable_text

    assert "main app file: app.py" in searchable
    assert "score_idea" in searchable
    assert "Project idea" in searchable


def test_public_project_tags_exclude_hosting_metadata() -> None:
    project = Project.from_dict(
        {
            "id": "build-small-hackathon/idea-canvas",
            "title": "Idea Canvas",
            "summary": "",
            "tags": ["gradio", "region:us", "local-first", "region:eu", "gradio"],
            "models": [],
            "datasets": [],
            "url": "https://example.test",
        }
    )

    assert project.tags == ("gradio", "region:us", "local-first", "region:eu", "gradio")
    assert project.to_public_dict()["tags"] == ["gradio", "local-first"]


def test_searchable_text_excludes_refresh_readme_body_for_stable_reuse() -> None:
    project = Project(
        id="build-small-hackathon/long-readme",
        title="Long README",
        summary="",
        tags=(),
        models=(),
        datasets=(),
        likes=0,
        sdk="gradio",
        license="",
        created_at="",
        last_modified="",
        host="",
        url="https://example.test",
        readme_body="a" * 2500 + "middle should not be embedded" + "b" * 2500,
    )

    searchable = project.searchable_text

    assert "readme:" not in searchable
    assert "middle should not be embedded" not in searchable


def test_project_index_rejects_mismatched_snapshot(tmp_path: Path) -> None:
    payload = json.loads(Path("data/project_index.json").read_text(encoding="utf-8"))
    payload["snapshot_generated_at"] = "2000-01-01T00:00:00+00:00"
    bad_index = tmp_path / "project_index.json"
    bad_index.write_text(json.dumps(payload), encoding="utf-8")

    try:
        ProjectIndex.from_files(Path("data/projects.json"), bad_index)
    except ValueError as error:
        assert "different snapshot timestamp" in str(error)
    else:
        raise AssertionError("mismatched index should be rejected")


def test_project_index_retains_validated_payload() -> None:
    payload = json.loads(Path("data/project_index.json").read_text(encoding="utf-8"))
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))

    assert index.index_payload["snapshot_digest"] == payload["snapshot_digest"]
    assert len(index.index_payload["documents"]) == len(index.projects)
