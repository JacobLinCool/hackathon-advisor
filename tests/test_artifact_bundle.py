import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.artifact_bundle import build_demo_bundle_zip
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.demo_rehearsal import build_demo_rehearsal
from hackathon_advisor.prize_ledger import prize_ledger
from hackathon_advisor.trace_export import trace_metadata


def test_demo_bundle_contains_submission_evidence_files() -> None:
    index = ProjectIndex.from_files(Path("data/projects.json"), Path("data/project_index.json"))
    engine = AdvisorEngine(index)
    metadata = {
        **trace_metadata(index),
        "project_count": len(index.projects),
    }
    content = build_demo_bundle_zip(build_demo_rehearsal(engine), metadata, prize_ledger(engine.runtime_status()))

    with ZipFile(BytesIO(content)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        trace = archive.read("trace.jsonl").decode("utf-8")
        packet = archive.read("submission-packet.md").decode("utf-8")

    assert names == {
        "manifest.json",
        "demo-session.json",
        "prize-ledger.json",
        "trace.jsonl",
        "field-notes.md",
        "almanac-chapter.md",
        "lora-sft.jsonl",
        "lora-training-kit.zip",
        "submission-packet.md",
        "png-export-note.md",
    }
    assert manifest["type"] == "demo_bundle_manifest"
    assert manifest["turn_count"] == 2
    assert manifest["badge_status"]["Well-Tuned"] == "training-kit-ready"
    assert "agent_turn" in trace
    assert "## Prize Evidence" in packet
