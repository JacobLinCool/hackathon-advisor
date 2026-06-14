from tests.helpers import load_test_index

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.field_notes import build_field_notes_markdown
from hackathon_advisor.trace_export import trace_metadata


def test_field_notes_markdown_contains_session_decisions() -> None:
    index = load_test_index()
    engine = AdvisorEngine(index)
    state = {
        "profile": {"skills": "frontend prototyping"},
        "goals": ["Field Notes"],
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
    assert "Goals: Build notes" in markdown
    assert "Targets: Field Notes" not in markdown
    assert "A local-first archive cartographer for family photos" in markdown
    assert "## Build Plan" in markdown
    assert "Write build notes from the exact decisions" in markdown
    assert "## Session Decisions" in markdown
    assert "### Decision 2" in markdown
    assert "Recorded action: `make_plan`" in markdown
    assert "Closest cited Spaces" in markdown
    assert "Page " in markdown
    assert "## Wood Map" in markdown
    assert "echo score" in markdown
    assert "## Turn Trace" not in markdown
    assert "Planner call" not in markdown
    assert "tool trace" not in markdown.lower()
