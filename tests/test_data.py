from pathlib import Path

from hackathon_advisor.data import ProjectIndex


def test_project_index_searches_snapshot() -> None:
    index = ProjectIndex.from_file(Path("data/projects.json"))

    hits = index.search("lullaby children audio", limit=3)

    assert hits
    assert hits[0].project.id.startswith("build-small-hackathon/")


def test_project_index_whitespace() -> None:
    index = ProjectIndex.from_file(Path("data/projects.json"))

    items = index.find_whitespace(limit=3)

    assert len(items) == 3
    assert all(item.label for item in items)
