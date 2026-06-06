from pathlib import Path

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.tool_contracts import ToolCall, ToolResolution


class StaticPlanner:
    backend = "test"
    model_id = "static"

    def __init__(self, call: ToolCall) -> None:
        self.call = call

    def plan(self, message: str, state: dict) -> ToolResolution:
        return ToolResolution(status="valid", call=self.call, errors=())


def test_agent_scores_and_persists_idea() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)

    result = engine.turn("A local-first archive cartographer for family photos", {})

    assert result.score is not None
    assert result.state["ideas"]
    assert result.state["ideas"][0]["score"] is not None
    assert result.state["trace"]
    assert result.state["last_tool_resolution"]["call"]["name"] == "save_idea"
    assert result.state["trace"][0]["tool_resolution"]["call"]["name"] == "save_idea"
    assert result.state["last_artifact"]["title"] == result.artifact["title"]
    assert result.artifact["wood_map"]["caption"]
    assert {dot["kind"] for dot in result.artifact["wood_map"]["dots"]} >= {"idea", "echo", "inked"}
    assert result.score.to_dict()["echoes"][0]["page_number"] >= 1
    assert "page " in result.response
    assert result.response


def test_agent_finds_whitespace() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)

    result = engine.turn("write bolder and find whitespace", {})

    assert result.whitespace
    assert result.score is not None
    assert result.artifact["verdict"] == "UNWRITTEN"
    assert result.state["ideas"][0]["title"] == result.whitespace[0].label


def test_agent_preserves_canonical_jargon_case() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)

    result = engine.turn("use neutron and mini cpm on zero gpu", {})

    assert "MiniCPM5" in result.artifact["title"]
    assert "ZeroGPU" in result.artifact["title"]


def test_plan_command_uses_current_idea() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)

    first = engine.turn("A local-first archive cartographer for family photos", {})
    planned = engine.turn("make a build plan", first.state)

    assert planned.plan
    assert planned.artifact["title"] == first.artifact["title"]
    assert planned.state["ideas"][0]["title"] == first.artifact["title"]


def test_plan_preserves_unwritten_whitespace_verdict() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)

    whitespace = engine.turn("write bolder and find whitespace", {})
    planned = engine.turn("make a build plan", whitespace.state)

    assert whitespace.artifact["verdict"] == "UNWRITTEN"
    assert planned.artifact["title"] == whitespace.artifact["title"]
    assert planned.artifact["verdict"] == "UNWRITTEN"


def test_planner_get_project_drives_project_response() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index, planner=StaticPlanner(ToolCall("get_project", {"id": "lolaby"})))

    result = engine.turn("read lolaby", {})

    assert result.projects
    assert result.projects[0].slug == "lolaby"
    assert result.tool_events[0].name == "get_project"


def test_planner_profile_and_targets_update_state() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    profile_engine = AdvisorEngine(
        index,
        planner=StaticPlanner(ToolCall("update_profile", {"field": "skills", "value": "frontend"})),
    )
    profile = profile_engine.turn("remember this", {})
    target_engine = AdvisorEngine(
        index,
        planner=StaticPlanner(ToolCall("set_target", {"side_quests": ["Off the Grid", "Field Notes"]})),
    )
    targeted = target_engine.turn("target prizes", profile.state)

    assert targeted.state["profile"]["skills"] == "frontend"
    assert targeted.state["targets"] == ["Off the Grid", "Field Notes"]


def test_session_targets_apply_to_new_and_current_ideas() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)
    state = {"targets": ["Field Notes"]}

    first = engine.turn("A local-first archive cartographer for family photos", state)
    first_idea = first.state["ideas"][0]
    planned = engine.turn("make a build plan", first.state)

    assert first_idea["targets"] == ["Field Notes"]
    assert all("LoRA" not in step for step in planned.plan)


def test_well_tuned_target_adds_training_step_to_plan() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)
    state = {"targets": ["Well-Tuned"]}

    first = engine.turn("A local-first archive cartographer for family photos", state)
    planned = engine.turn("make a build plan", first.state)

    assert first.state["ideas"][0]["targets"] == ["Well-Tuned"]
    assert any("LoRA" in step for step in planned.plan)


def test_planner_score_idea_scores_current_idea() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    first = AdvisorEngine(index).turn("A local-first archive cartographer for family photos", {})
    engine = AdvisorEngine(index, planner=StaticPlanner(ToolCall("score_idea", {})))

    scored = engine.turn("score it", first.state)

    assert scored.score is not None
    assert scored.artifact["title"] == first.artifact["title"]
