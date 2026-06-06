from pathlib import Path
import json

from hackathon_advisor.data import ProjectIndex


def test_project_index_searches_snapshot() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))

    hits = index.search("lullaby children audio", limit=3)

    assert hits
    assert hits[0].project.id.startswith("build-small-hackathon/")
    assert hits[0].page_number >= 1
    assert index.index_algorithm == "tfidf-sparse-v1"


def test_project_index_whitespace() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))

    items = index.find_whitespace(limit=3)

    assert len(items) == 3
    assert all(item.label for item in items)


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
