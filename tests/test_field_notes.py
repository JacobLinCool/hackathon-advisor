from pathlib import Path

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.field_notes import build_field_notes_markdown
from hackathon_advisor.trace_export import trace_metadata


def test_field_notes_markdown_contains_session_decisions() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)
    state = {
        "profile": {"skills": "frontend prototyping"},
        "targets": ["Field Notes"],
    }
    first = engine.turn("A local-first archive cartographer for family photos", state)
    planned = engine.turn("make a build plan", first.state)

    markdown = build_field_notes_markdown(
        planned.state,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )

    assert "# Hackathon Advisor Field Notes" in markdown
    assert "frontend prototyping" in markdown
    assert "Targets: Field Notes" in markdown
    assert "A local-first archive cartographer for family photos" in markdown
    assert "## Build Plan" in markdown
    assert "Record the trace and write Field Notes" in markdown
    assert "Closest cited Spaces" in markdown
    assert "Planner call: `make_plan`" in markdown
