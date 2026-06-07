import json
from pathlib import Path

from tests.helpers import load_test_index

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.trace_export import build_trace_jsonl, trace_metadata


def test_trace_jsonl_contains_manifest_and_turns() -> None:
    index = load_test_index()
    engine = AdvisorEngine(index)
    state = engine.turn("A local-first archive cartographer for family photos", {}).state
    state = engine.turn("make a build plan", state).state

    lines = [json.loads(line) for line in build_trace_jsonl(state, trace_metadata(index)).splitlines()]

    assert lines[0]["type"] == "trace_manifest"
    assert lines[0]["turn_count"] == 2
    assert lines[0]["index"]["algorithm"] == "llama-cpp-embedding-v1"
    assert lines[1]["type"] == "agent_turn"
    assert lines[1]["tools"]
    assert lines[1]["tool_resolution"]["call"]["name"] == "save_idea"
    assert lines[2]["plan_steps"] > 0


def test_checked_in_sample_trace_matches_schema() -> None:
    lines = [
        json.loads(line)
        for line in Path("data/sample_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sample_text = "\n".join(json.dumps(line) for line in lines)

    assert lines[0]["type"] == "trace_manifest"
    assert lines[0]["turn_count"] >= 3
    assert all(line["schema_version"] == 1 for line in lines)
    assert lines[1]["tool_resolution"]["status"] == "valid"
    assert "Mothback" not in sample_text
    assert "Add one prize" not in sample_text
    assert "set_target" not in sample_text
    assert "side_quests" not in sample_text
    assert "Record the trace" not in sample_text
    assert "Write build notes from the exact decisions" in lines[-1]["response"]
