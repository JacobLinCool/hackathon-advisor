from pathlib import Path

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex


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
