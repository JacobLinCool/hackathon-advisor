import pytest

from hackathon_advisor.model_runtime import (
    DEFAULT_ADAPTER_ID,
    MiniCPMTransformersPlanner,
    RuleBasedPlanner,
    create_tool_planner,
    render_context,
    runtime_status,
    system_prompt,
    _strip_unused_generation_inputs,
)
from hackathon_advisor.zerogpu import gpu_task, zero_gpu_duration_seconds, zero_gpu_enabled


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


def test_rule_planner_keeps_empty_board_commands_as_commands() -> None:
    planner = RuleBasedPlanner()

    plan = planner.plan("make a build plan", {})
    rank = planner.plan("compare ideas", {})

    assert plan.status == "valid"
    assert plan.call.name == "make_plan"
    assert rank.status == "valid"
    assert rank.call.name == "compare_ideas"


def test_rule_planner_defaults_blank_to_list_projects() -> None:
    planner = RuleBasedPlanner()

    resolution = planner.plan("", {})

    assert resolution.status == "valid"
    assert resolution.call.name == "list_projects"


def test_rule_planner_routes_project_reference_commands() -> None:
    planner = RuleBasedPlanner()

    listed = planner.plan("show current map", {})
    project = planner.plan("read project lolaby", {})
    project_url = planner.plan("open space https://huggingface.co/spaces/build-small-hackathon/lolaby", {})

    assert listed.status == "valid"
    assert listed.call.name == "list_projects"
    assert project.status == "valid"
    assert project.call.name == "get_project"
    assert project.call.arguments["id"] == "lolaby"
    assert project_url.status == "valid"
    assert project_url.call.name == "get_project"
    assert project_url.call.arguments["id"] == "build-small-hackathon/lolaby"


def test_rule_planner_keeps_project_words_inside_ideas() -> None:
    planner = RuleBasedPlanner()

    resolution = planner.plan("A dashboard that helps teams show projects to mentors", {})

    assert resolution.status == "valid"
    assert resolution.call.name == "save_idea"


def test_rule_planner_splits_explicit_idea_pitch() -> None:
    planner = RuleBasedPlanner()

    resolution = planner.plan(
        "idea: Hands-on science coach -- A lab-notebook companion for household experiments.",
        {},
    )

    assert resolution.status == "valid"
    assert resolution.call.name == "save_idea"
    assert resolution.call.arguments["title"] == "Hands-on science coach"
    assert resolution.call.arguments["pitch"] == "A lab-notebook companion for household experiments."


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
    assert "Available tools:" in context
    assert "search_projects" in context


def test_system_prompt_keeps_runtime_role_user_facing() -> None:
    prompt = system_prompt()

    assert "The Unwritten Almanac" in prompt
    assert "Mothback" not in prompt
    assert "Build Small" not in prompt


def test_create_tool_planner_defaults_to_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADVISOR_MODEL_BACKEND", raising=False)

    planner = create_tool_planner()

    assert isinstance(planner, RuleBasedPlanner)
    assert runtime_status(planner).to_dict()["loaded"] is True
    assert runtime_status(planner).to_dict()["adapter_id"] == ""


def test_create_tool_planner_accepts_adapter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADVISOR_MODEL_BACKEND", "minicpm-transformers")
    monkeypatch.setenv("ADVISOR_MODEL_ID", "openbmb/MiniCPM5-1B")
    monkeypatch.setenv("ADVISOR_ADAPTER_ID", DEFAULT_ADAPTER_ID)

    planner = create_tool_planner()
    status = runtime_status(planner).to_dict()

    assert isinstance(planner, MiniCPMTransformersPlanner)
    assert status["backend"] == "minicpm-transformers"
    assert status["model_id"] == "openbmb/MiniCPM5-1B"
    assert status["adapter_id"] == DEFAULT_ADAPTER_ID
    assert status["loaded"] is False


def test_create_tool_planner_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADVISOR_MODEL_BACKEND", "bogus")

    with pytest.raises(RuntimeError, match="Unsupported"):
        create_tool_planner()


def test_minicpm_status_is_lazy() -> None:
    planner = MiniCPMTransformersPlanner("openbmb/MiniCPM5-1B", DEFAULT_ADAPTER_ID)
    status = runtime_status(planner).to_dict()

    assert status["backend"] == "minicpm-transformers"
    assert status["adapter_id"] == DEFAULT_ADAPTER_ID
    assert status["loaded"] is False


def test_zerogpu_disabled_leaves_function_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADVISOR_ZERO_GPU", raising=False)

    def marker() -> str:
        return "ok"

    assert zero_gpu_enabled() is False
    assert gpu_task(marker) is marker


def test_zerogpu_duration_validates_positive_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADVISOR_ZERO_GPU_DURATION", "7")
    assert zero_gpu_duration_seconds() == 7

    monkeypatch.setenv("ADVISOR_ZERO_GPU_DURATION", "0")
    with pytest.raises(RuntimeError, match="positive"):
        zero_gpu_duration_seconds()

    monkeypatch.setenv("ADVISOR_ZERO_GPU_DURATION", "121")
    with pytest.raises(RuntimeError, match="at most 120"):
        zero_gpu_duration_seconds()


def test_generation_inputs_drop_token_type_ids() -> None:
    inputs = {"input_ids": [1], "attention_mask": [1], "token_type_ids": [0]}

    _strip_unused_generation_inputs(inputs)

    assert inputs == {"input_ids": [1], "attention_mask": [1]}
