import re

from tests.helpers import load_test_index

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.chapter import build_chapter_markdown
from hackathon_advisor.trace_export import trace_metadata


def test_chapter_markdown_contains_idea_pages_and_citations() -> None:
    index = load_test_index()
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
    assert "Goals:" in markdown
    assert "Targets:" not in markdown
    assert "Closest cited pages:" in markdown
    assert re.search(r"Page \d+:", markdown)


def test_empty_chapter_markdown_is_explicit() -> None:
    markdown = build_chapter_markdown(
        {},
        {
            "snapshot_generated_at": "2026-06-06T00:00:00+00:00",
            "project_count": 100,
        },
    )

    assert "No idea pages have been written yet." in markdown
