from pathlib import Path

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.demo_rehearsal import DEMO_TARGETS, build_demo_rehearsal


def test_demo_rehearsal_builds_complete_session() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)

    payload = build_demo_rehearsal(engine)
    session = payload["session"]

    assert payload["turn_count"] == 2
    assert payload["score"]["overall"] > 0
    assert payload["artifact"]["title"] == "Archive Cartographer"
    assert session["ideas"][0]["title"] == "Archive Cartographer"
    assert payload["artifact"]["wood_map"]["dots"]
    assert payload["plan"]
    assert any("LoRA" in step for step in payload["plan"])
    assert session["targets"] == DEMO_TARGETS
    assert session["profile"]["constraints"] == "CPU Space runtime; no proprietary inference API"
    assert len(session["trace"]) == 2
    assert payload["export_ready"] == {
        "trace": True,
        "notes": True,
        "chapter": True,
        "lora_dataset": True,
        "submission_packet": True,
        "png": True,
    }
