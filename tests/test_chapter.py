from pathlib import Path

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.chapter import build_chapter_markdown
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.trace_export import trace_metadata


def test_chapter_markdown_contains_idea_pages_and_citations() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)
    state = engine.turn("A local-first archive cartographer for family photos", {}).state
    state = engine.turn("write bolder and find whitespace", state).state

    markdown = build_chapter_markdown(
        state,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )

    assert markdown.startswith("# The Unwritten Almanac Chapter")
    assert "## Page 1:" in markdown
    assert "## Page 2:" in markdown
    assert "Targets:" in markdown
    assert "Closest inked pages:" in markdown
    assert "Page 30:" in markdown


def test_empty_chapter_markdown_is_explicit() -> None:
    markdown = build_chapter_markdown(
        {},
        {
            "snapshot_generated_at": "2026-06-06T00:00:00+00:00",
            "project_count": 100,
        },
    )

    assert "No fate pages have been written yet." in markdown
