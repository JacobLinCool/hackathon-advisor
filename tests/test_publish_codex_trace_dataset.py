import json
from pathlib import Path

from scripts.publish_codex_trace_dataset import RedactionResult, TextCaps, build_dataset


class FakePrivacyRedactor:
    def redact_many(self, texts: list[str]) -> list[RedactionResult]:
        results: list[RedactionResult] = []
        for text in texts:
            count = text.count("Alice Smith") + text.count("alice@example.com")
            redacted = text.replace("Alice Smith", "[PRIVATE_PERSON]")
            redacted = redacted.replace("alice@example.com", "[PRIVATE_EMAIL]")
            labels = {"PRIVATE": count} if count else {}
            results.append(RedactionResult(text=redacted, count=count, labels=labels))
        return results


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_codex_trace_dataset_selects_minimizes_and_redacts(tmp_path: Path) -> None:
    project_root = tmp_path / "hackathon-advisor"
    project_root.mkdir()
    session_root = tmp_path / "sessions"
    session_root.mkdir()
    session_file = session_root / "rollout-test.jsonl"
    home_secret_path = str(Path.home() / "Documents" / "private-note.txt")
    token = "hf_" + "a" * 24

    write_jsonl(
        session_file,
        [
            {
                "type": "session_meta",
                "timestamp": "2026-06-08T00:00:00Z",
                "payload": {
                    "id": "session-1",
                    "cwd": str(project_root),
                    "originator": "Codex Desktop",
                    "base_instructions": {"do_not_publish": True},
                    "dynamic_tools": ["internal"],
                    "git": {"repository_url": "https://github.com/example/hackathon-advisor.git"},
                },
            },
            {
                "type": "turn_context",
                "timestamp": "2026-06-08T00:00:01Z",
                "payload": {
                    "turn_id": "turn-1",
                    "cwd": str(project_root),
                    "workspace_roots": [str(project_root)],
                    "collaboration_mode": {"mode": "default", "settings": "internal"},
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-08T00:00:02Z",
                "payload": {
                    "type": "user_message",
                    "turn_id": "turn-1",
                    "message": (
                        f"Help Alice Smith at alice@example.com using {home_secret_path} "
                        f"and HF_TOKEN={token}"
                    ),
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-08T00:00:03Z",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "internal prompt"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-08T00:00:04Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Help Alice Smith at alice@example.com using {home_secret_path} "
                                f"and HF_TOKEN={token}"
                            ),
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-08T00:00:05Z",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "pytest", "workdir": str(project_root)}),
                    "call_id": "call-1",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-06-08T00:00:06Z",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "0123456789" * 12,
                },
            },
            {
                "type": "compacted",
                "timestamp": "2026-06-08T00:00:07Z",
                "payload": {"replacement_history": ["internal"]},
            },
        ],
    )

    out_dir = tmp_path / "dataset"
    manifest = build_dataset(
        project_root=project_root,
        session_roots=[session_root],
        include_terms=[],
        out_dir=out_dir,
        redactor=FakePrivacyRedactor(),
        privacy_model_id="openai/privacy-filter",
        privacy_model_revision="test",
        privacy_device="test",
        min_score=0.5,
        record_batch_size=2,
        text_caps=TextCaps(
            message=200,
            tool_argument=200,
            tool_output=80,
            other=200,
        ),
    )

    rows = [
        json.loads(line)
        for line in (out_dir / "codex_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    dataset_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)

    assert manifest["selected_session_count"] == 1
    assert manifest["published_record_count"] == 5
    assert manifest["dropped_record_count"] == 3
    assert manifest["redaction_count"] == 2
    assert manifest["truncated_field_count"] == 1
    assert manifest["truncated_char_count"] > 0
    assert len(manifest["sessions"][0]["source_sha256"]) == 64
    assert all(row["session_id"] == "session-1" for row in rows)
    assert "$PROJECT_ROOT" in dataset_text
    assert str(project_root) not in dataset_text
    assert str(Path.home()) not in dataset_text
    assert token not in dataset_text
    assert "base_instructions" not in dataset_text
    assert "dynamic_tools" not in dataset_text
    assert "internal prompt" not in dataset_text
    assert "replacement_history" not in dataset_text
    assert "role" not in dataset_text
    assert "alice@example.com" not in dataset_text
    assert "Alice Smith" not in dataset_text
    assert "[PRIVATE_EMAIL]" in dataset_text
    assert "[PRIVATE_PERSON]" in dataset_text
    assert "[truncated" in dataset_text


def test_build_dataset_redacts_caller_home_when_run_home_differs(tmp_path: Path, monkeypatch) -> None:
    # Simulates the Modal container, where Path.home() is /root rather than the user's
    # machine. The caller's real home must travel via path_redaction_prefixes to be redacted;
    # this guards the unified --location code path that passes [project, caller-home] on both lanes.
    project_root = tmp_path / "hackathon-advisor"
    project_root.mkdir()
    session_root = tmp_path / "sessions"
    session_root.mkdir()
    caller_home = "/home/realuser"
    secret_path = f"{caller_home}/Documents/private-note.txt"

    write_jsonl(
        session_root / "rollout-test.jsonl",
        [
            {
                "type": "session_meta",
                "timestamp": "2026-06-08T00:00:00Z",
                "payload": {
                    "id": "session-1",
                    "cwd": str(project_root),
                    "git": {"repository_url": "https://github.com/example/hackathon-advisor.git"},
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-06-08T00:00:01Z",
                "payload": {
                    "type": "user_message",
                    "turn_id": "turn-1",
                    "message": f"please open {secret_path} for the hackathon-advisor project",
                },
            },
        ],
    )

    # Container home differs from the caller's real home.
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/root")))

    out_dir = tmp_path / "dataset"
    manifest = build_dataset(
        project_root=project_root,
        session_roots=[session_root],
        include_terms=[],
        out_dir=out_dir,
        redactor=FakePrivacyRedactor(),
        privacy_model_id="openai/privacy-filter",
        privacy_model_revision="test",
        privacy_device="test",
        min_score=0.5,
        record_batch_size=2,
        text_caps=TextCaps(message=200, tool_argument=200, tool_output=80, other=200),
        path_redaction_prefixes=[caller_home, str(project_root)],
    )

    dataset_text = (out_dir / "codex_sessions.jsonl").read_text(encoding="utf-8")
    assert manifest["published_record_count"] >= 1
    assert caller_home not in dataset_text
    assert "~/Documents/private-note.txt" in dataset_text
