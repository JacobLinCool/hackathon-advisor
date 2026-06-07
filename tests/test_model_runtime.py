import pytest

from hackathon_advisor.model_runtime import (
    DEFAULT_ADAPTER_ID,
    MiniCPMTransformersPlanner,
    RuleBasedPlanner,
    create_tool_planner,
    render_context,
    runtime_status,
    system_prompt,
    _best_local_device,
    _disable_sampling_generation_defaults,
    _normalize_xml_tool_output,
    _resolve_torch_device,
    _strip_unused_generation_inputs,
)
from hackathon_advisor.zerogpu import gpu_task, zero_gpu_duration_seconds, zero_gpu_enabled


class FakeBackends:
    def __init__(self, mps: bool) -> None:
        self.mps = type("MPS", (), {"is_available": staticmethod(lambda: mps)})()


class FakeTorch:
    def __init__(self, cuda: bool = False, mps: bool = False) -> None:
        self.cuda = type("CUDA", (), {"is_available": staticmethod(lambda: cuda)})()
        self.backends = FakeBackends(mps)


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


def test_rule_planner_does_not_match_commands_inside_idea_words() -> None:
    planner = RuleBasedPlanner()

    planting = planner.plan(
        "A neighborhood seed swap archive that reminds gardeners when to plant shared seeds",
        {},
    )
    cooking_plan = planner.plan(
        "A countertop helper that turns pantry leftovers into a weekly cooking plan",
        {},
    )

    assert planting.status == "valid"
    assert planting.call.name == "save_idea"
    assert cooking_plan.status == "valid"
    assert cooking_plan.call.name == "save_idea"


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
    assert runtime_status(planner).to_dict()["adapter_revision"] == ""


def test_create_tool_planner_accepts_adapter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADVISOR_MODEL_BACKEND", "minicpm-transformers")
    monkeypatch.setenv("ADVISOR_MODEL_ID", "openbmb/MiniCPM5-1B")
    monkeypatch.setenv("ADVISOR_ADAPTER_ID", DEFAULT_ADAPTER_ID)
    monkeypatch.setenv("ADVISOR_ADAPTER_REVISION", "abc123")

    planner = create_tool_planner()
    status = runtime_status(planner).to_dict()

    assert isinstance(planner, MiniCPMTransformersPlanner)
    assert status["backend"] == "minicpm-transformers"
    assert status["model_id"] == "openbmb/MiniCPM5-1B"
    assert status["adapter_id"] == DEFAULT_ADAPTER_ID
    assert status["adapter_revision"] == "abc123"
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
    assert status["adapter_revision"] == ""
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


def test_generation_config_drops_sampling_defaults() -> None:
    class GenerationConfig:
        do_sample = True
        temperature = 0.7
        top_p = 0.95

    class Model:
        generation_config = GenerationConfig()

    _disable_sampling_generation_defaults(Model())

    assert Model.generation_config.do_sample is False
    assert Model.generation_config.temperature is None
    assert Model.generation_config.top_p is None


def test_model_xml_fragment_is_normalized() -> None:
    output = 'name="save_idea">{"title":"A","pitch":"B"}'

    assert _normalize_xml_tool_output(output) == '<function name="save_idea">{"title":"A","pitch":"B"}</function>'


def test_resolve_device_keeps_auto_and_explicit_cpu() -> None:
    assert _resolve_torch_device("auto", FakeTorch()) == "auto"
    assert _resolve_torch_device("cpu", FakeTorch(cuda=True, mps=True)) == "cpu"


def test_resolve_device_prefers_cuda_then_mps_then_cpu(monkeypatch) -> None:
    monkeypatch.delenv("ADVISOR_ZERO_GPU", raising=False)

    assert _best_local_device(FakeTorch(cuda=True, mps=True)) == "cuda"
    assert _best_local_device(FakeTorch(cuda=False, mps=True)) == "mps"
    assert _best_local_device(FakeTorch(cuda=False, mps=False)) == "cpu"
    # "local" resolves through the same ladder
    assert _resolve_torch_device("local", FakeTorch(cuda=False, mps=True)) == "mps"


def test_resolve_device_unavailable_request_degrades_gracefully(monkeypatch) -> None:
    monkeypatch.delenv("ADVISOR_ZERO_GPU", raising=False)

    # asking for cuda on an MPS-only box lands on mps, not a crash
    assert _resolve_torch_device("cuda", FakeTorch(cuda=False, mps=True)) == "mps"


def test_resolve_device_skips_cuda_under_zero_gpu(monkeypatch) -> None:
    # In a ZeroGPU main process there is no local CUDA, and probing it is avoided.
    monkeypatch.setenv("ADVISOR_ZERO_GPU", "1")

    assert _best_local_device(FakeTorch(cuda=True, mps=False)) == "cpu"


def test_runtime_status_reports_configured_device() -> None:
    planner = MiniCPMTransformersPlanner("openbmb/MiniCPM5-1B", device="local")

    assert runtime_status(planner).to_dict()["device"] == "local"
    assert runtime_status(RuleBasedPlanner()).to_dict()["device"] == ""
