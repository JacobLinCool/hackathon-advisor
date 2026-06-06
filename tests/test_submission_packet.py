from pathlib import Path

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.prize_ledger import prize_ledger
from hackathon_advisor.submission_packet import build_submission_packet_markdown
from hackathon_advisor.trace_export import trace_metadata


def test_submission_packet_contains_demo_and_prize_evidence() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)
    state = {"targets": ["Well-Tuned", "Field Notes"]}
    state = engine.turn("A local-first archive cartographer for family photos", state).state
    state = engine.turn("make a build plan", state).state

    markdown = build_submission_packet_markdown(
        state,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
        prize_ledger(engine.runtime_status()),
    )

    assert markdown.startswith("# Hackathon Advisor Submission Packet")
    assert "## Demo Script" in markdown
    assert "## Artifact Checklist" in markdown
    assert "## Prize Evidence" in markdown
    assert "## Model Budget" in markdown
    assert "## Social Post Draft" in markdown
    assert "Hackathon Advisor" in markdown
    assert "Well-Tuned | dataset-ready" in markdown
    assert "MiniCPM5 LoRA SFT JSONL | ready | lora_dataset" in markdown
    assert "Ready badges and planned badges are separated" in markdown
    assert "A local-first archive cartographer for family photos" in markdown


def test_empty_submission_packet_is_honest_about_missing_session_artifacts() -> None:
    markdown = build_submission_packet_markdown(
        {},
        {
            "snapshot_generated_at": "2026-06-06T00:00:00+00:00",
            "project_count": 100,
            "index_algorithm": "tfidf-sparse-v1",
            "index_generated_at": "2026-06-06T01:00:00+00:00",
            "snapshot_digest": "abc",
        },
        prize_ledger({"backend": "rules", "model_id": "deterministic-tool-router"}),
    )

    assert "Title: Unwritten Page" in markdown
    assert "Tool trace JSONL | needs session" in markdown
    assert "Submission packet markdown | ready" in markdown
    assert "No ideas recorded yet." in markdown
