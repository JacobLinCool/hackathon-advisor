from pathlib import Path

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex


def test_agent_scores_and_persists_idea() -> None:
    index = ProjectIndex.from_file(Path("data/projects.json"))
    engine = AdvisorEngine(index)

    result = engine.turn("A local-first archive cartographer for family photos", {})

    assert result.score is not None
    assert result.state["ideas"]
    assert result.state["ideas"][0]["score"] is not None
    assert result.response


def test_agent_finds_whitespace() -> None:
    index = ProjectIndex.from_file(Path("data/projects.json"))
    engine = AdvisorEngine(index)

    result = engine.turn("write bolder and find whitespace", {})

    assert result.whitespace
    assert result.score is not None
    assert result.artifact["verdict"]


def test_agent_preserves_canonical_jargon_case() -> None:
    index = ProjectIndex.from_file(Path("data/projects.json"))
    engine = AdvisorEngine(index)

    result = engine.turn("use neutron and mini cpm on zero gpu", {})

    assert "MiniCPM5" in result.artifact["title"]
    assert "ZeroGPU" in result.artifact["title"]
