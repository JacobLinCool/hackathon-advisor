from tests.helpers import load_test_index

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.demo_rehearsal import DEMO_GOALS, build_demo_rehearsal


def test_demo_rehearsal_builds_complete_session() -> None:
    index = load_test_index()
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
    assert session["goals"] == DEMO_GOALS
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
