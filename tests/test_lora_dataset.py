import json

from tests.helpers import load_test_index

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.lora_dataset import BASE_MODEL, build_lora_dataset_jsonl
from hackathon_advisor.trace_export import trace_metadata


def test_lora_dataset_exports_tool_call_and_response_examples() -> None:
    index = load_test_index()
    engine = AdvisorEngine(index)
    state = {"goals": ["Well-Tuned", "Field Notes"]}
    state = engine.turn("A local-first archive cartographer for family photos", state).state
    state = engine.turn("make a build plan", state).state

    lines = [json.loads(line) for line in build_lora_dataset_jsonl(state, trace_metadata(index)).splitlines()]
    manifest = lines[0]
    examples = lines[1:]

    assert manifest["type"] == "lora_sft_manifest"
    assert manifest["base_model"] == BASE_MODEL
    assert manifest["record_kinds"] == ["tool_call", "advisor_response"]
    assert manifest["example_count"] == len(examples)
    assert manifest["included_turn_count"] == 2
    assert manifest["index"]["algorithm"] == "llama-cpp-embedding-v1"
    assert {example["example_kind"] for example in examples} == {"tool_call", "advisor_response"}
    assert examples[0]["messages"][2]["content"].startswith('<function name="save_idea">')
    assert examples[0]["goals"] == ["Well-Tuned", "Field Notes"]
    assert examples[1]["messages"][1]["content"].startswith("A local-first archive")
    assert "Tool observations:" in examples[1]["messages"][1]["content"]
    assert examples[1]["messages"][2]["content"]
    system_messages = "\n".join(example["messages"][0]["content"] for example in examples)
    assert "Mothback" not in system_messages
    assert "Build Small" not in system_messages
    assert "prize " + "tar" + "gets" not in system_messages
    assert "selected goals" in system_messages


def test_empty_lora_dataset_only_exports_manifest() -> None:
    payload = build_lora_dataset_jsonl(
        {},
        {
            "index_algorithm": "llama-cpp-embedding-v1",
            "snapshot_generated_at": "2026-06-06T00:00:00+00:00",
            "index_generated_at": "2026-06-06T01:00:00+00:00",
            "snapshot_digest": "abc",
        },
    )
    lines = [json.loads(line) for line in payload.splitlines()]

    assert len(lines) == 1
    assert lines[0]["example_count"] == 0
    assert lines[0]["turn_count"] == 0
