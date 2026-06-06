import pytest

from hackathon_advisor.model_runtime import (
    MiniCPMTransformersPlanner,
    RuleBasedPlanner,
    create_tool_planner,
    render_context,
    runtime_status,
)


def test_rule_planner_emits_valid_search_call() -> None:
    planner = RuleBasedPlanner()

    resolution = planner.plan("search similar lullaby audio projects", {})

    assert resolution.status == "valid"
    assert resolution.call.name == "search_projects"
    assert resolution.call.arguments["query"] == "search similar lullaby audio projects"


def test_rule_planner_uses_plan_when_idea_exists() -> None:
    planner = RuleBasedPlanner()

    resolution = planner.plan("make a build plan", {"ideas": [{"title": "A", "pitch": "B"}]})

    assert resolution.status == "valid"
    assert resolution.call.name == "make_plan"


def test_rule_planner_defaults_blank_to_list_projects() -> None:
    planner = RuleBasedPlanner()

    resolution = planner.plan("", {})

    assert resolution.status == "valid"
    assert resolution.call.name == "list_projects"


def test_render_context_includes_state() -> None:
    context = render_context(
        "make a plan",
        {
            "ideas": [{"title": "Archive Cartographer", "pitch": "Map family memories."}],
            "trace": [{"input": "first", "verdict": "ECHO x2", "overall": 5.1}],
        },
    )

    assert "Archive Cartographer" in context
    assert "ECHO x2" in context
    assert '<function name="tool_name">' in context


def test_create_tool_planner_defaults_to_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADVISOR_MODEL_BACKEND", raising=False)

    planner = create_tool_planner()

    assert isinstance(planner, RuleBasedPlanner)
    assert runtime_status(planner).to_dict()["loaded"] is True


def test_create_tool_planner_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADVISOR_MODEL_BACKEND", "bogus")

    with pytest.raises(RuntimeError, match="Unsupported"):
        create_tool_planner()


def test_minicpm_status_is_lazy() -> None:
    planner = MiniCPMTransformersPlanner("openbmb/MiniCPM5-1B")
    status = runtime_status(planner).to_dict()

    assert status["backend"] == "minicpm-transformers"
    assert status["loaded"] is False
