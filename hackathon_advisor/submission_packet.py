from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


SPACE_URL = "https://huggingface.co/spaces/build-small-hackathon/hackathon-advisor"
LIVE_URL = "https://build-small-hackathon-hackathon-advisor.hf.space"
GITHUB_URL = "https://github.com/JacobLinCool/hackathon-advisor"


def build_submission_packet_markdown(
    session: dict[str, Any],
    metadata: dict[str, Any],
    ledger: dict[str, Any],
) -> str:
    ideas = _list_of_dicts(session.get("ideas"))
    trace = _list_of_dicts(session.get("trace"))
    targets = [str(target) for target in session.get("targets") or []]
    last_plan = [str(step) for step in session.get("last_plan") or []]
    current = _current_idea(session, ideas)
    score = current.get("score") if isinstance(current.get("score"), dict) else {}
    verdict = _clean(score.get("verdict")) if score else "DRAFT"
    overall = _clean(score.get("overall")) if score else "0.0"
    title = _clean(current.get("title") or "Unwritten Page")

    lines = [
        "# Hackathon Advisor Submission Packet",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Links",
        "",
        f"- Live Space: {LIVE_URL}",
        f"- Hugging Face repository: {SPACE_URL}",
        f"- GitHub repository: {GITHUB_URL}",
        "",
        "## Snapshot",
        "",
        f"- Project snapshot: {_clean(metadata.get('snapshot_generated_at'))}",
        f"- Project count: {_clean(metadata.get('project_count', 'unknown'))}",
        f"- Index: {_clean(metadata.get('index_algorithm'))} at {_clean(metadata.get('index_generated_at'))}",
        f"- Digest: `{_clean(metadata.get('snapshot_digest'))}`",
        "",
        "## Current Demo Page",
        "",
        f"- Title: {title}",
        f"- Verdict: {verdict} {overall}/10",
        f"- Targets: {', '.join(targets) if targets else 'No specific targets'}",
        f"- Pitch: {_clean(current.get('pitch')) or 'No pitch recorded.'}",
        "",
    ]

    lines.extend(_demo_script(title, verdict, overall, current, trace, last_plan))
    lines.extend(_artifact_checklist(trace, ideas, session))
    lines.extend(_prize_evidence(ledger))
    lines.extend(_model_budget(ledger))
    lines.extend(_session_summary(ideas, trace, last_plan))
    lines.extend(_social_post(title, verdict, ledger))
    lines.extend(_open_gaps(ledger))

    return "\n".join(lines).rstrip() + "\n"


def _demo_script(
    title: str,
    verdict: str,
    overall: str,
    idea: dict[str, Any],
    trace: list[dict[str, Any]],
    last_plan: list[str],
) -> list[str]:
    echoes = _echoes(idea)
    first_echo = echoes[0] if echoes else {}
    project = first_echo.get("project") if isinstance(first_echo.get("project"), dict) else {}
    echo_title = _clean(project.get("title") or project.get("id") or "a cited Space")
    page = _clean(first_echo.get("page_number")) or "?"
    plan_step = _clean(last_plan[0]) if last_plan else "Press Plan to show the build path."
    turn_count = len(trace)
    return [
        "## Demo Script",
        "",
        "| Time | Beat | On-screen proof |",
        "| --- | --- | --- |",
        f"| 0:00 | Open The Unwritten Almanac and point to the local snapshot count. | Snapshot metadata and Prize Ledger are visible. |",
        f"| 0:10 | Type the project instinct for `{title}`. | Streaming response starts, then the seal reads {verdict} at {overall}/10. |",
        f"| 0:30 | Show the nearest echo. | Page {page}: {echo_title}. |",
        f"| 0:45 | Press Gap or Plan to move from diagnosis to build path. | {plan_step} |",
        f"| 1:05 | Export evidence artifacts. | JSONL, Notes, Chapter, LoRA, Packet, and PNG buttons are available after {turn_count} recorded turns. |",
        "| 1:20 | Close with prize honesty. | Ready badges and planned badges are separated in the Prize Ledger. |",
        "",
    ]


def _artifact_checklist(trace: list[dict[str, Any]], ideas: list[dict[str, Any]], session: dict[str, Any]) -> list[str]:
    has_trace = bool(trace)
    has_ideas = bool(ideas)
    has_artifact = isinstance(session.get("last_artifact"), dict) and bool(session["last_artifact"].get("title"))
    rows = [
        ("Tool trace JSONL", has_trace, "trace_artifact"),
        ("Field Notes markdown", has_trace, "field_notes"),
        ("Almanac chapter markdown", has_ideas, "chapter"),
        ("MiniCPM5 LoRA SFT JSONL", has_trace, "lora_dataset"),
        ("Submission packet markdown", True, "submission_packet"),
        ("Fate page PNG", has_artifact, "client-side canvas export"),
    ]
    lines = ["## Artifact Checklist", "", "| Artifact | Status | Source |", "| --- | --- | --- |"]
    for name, ready, source in rows:
        lines.append(f"| {name} | {'ready' if ready else 'needs session'} | {source} |")
    lines.append("")
    return lines


def _prize_evidence(ledger: dict[str, Any]) -> list[str]:
    lines = ["## Prize Evidence", "", "| Prize path | Status | Evidence |", "| --- | --- | --- |"]
    for badge in _list_of_dicts(ledger.get("badges")):
        lines.append(
            "| "
            + " | ".join(
                [
                    _clean(badge.get("name")),
                    _clean(badge.get("status")),
                    _clean(badge.get("evidence")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _model_budget(ledger: dict[str, Any]) -> list[str]:
    total = _clean(ledger.get("total_params_b"))
    limit = _clean(ledger.get("tiny_titan_limit_b"))
    eligible = "yes" if ledger.get("tiny_titan_eligible") else "no"
    lines = [
        "## Model Budget",
        "",
        f"- Total documented stack: {total}B params",
        f"- Tiny Titan limit: {limit}B params",
        f"- Tiny Titan eligible: {eligible}",
        "",
        "| Role | Model | Params | Status | Runtime |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in _list_of_dicts(ledger.get("model_stack")):
        lines.append(
            "| "
            + " | ".join(
                [
                    _clean(item.get("role")),
                    _clean(item.get("model")),
                    f"{_clean(item.get('params_b'))}B",
                    _clean(item.get("status")),
                    _clean(item.get("runtime")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _session_summary(ideas: list[dict[str, Any]], trace: list[dict[str, Any]], last_plan: list[str]) -> list[str]:
    lines = ["## Session Evidence", ""]
    if ideas:
        lines.extend(["### Idea Board", ""])
        for index, idea in enumerate(ideas[-4:], start=max(1, len(ideas) - 3)):
            score = idea.get("score") if isinstance(idea.get("score"), dict) else {}
            verdict = _clean(score.get("verdict")) if score else "DRAFT"
            overall = _clean(score.get("overall")) if score else "0.0"
            lines.append(f"- {index}. {_clean(idea.get('title'))}: {verdict} {overall}/10")
    else:
        lines.append("- No ideas recorded yet.")

    lines.extend(["", "### Latest Build Plan", ""])
    if last_plan:
        for index, step in enumerate(last_plan, start=1):
            lines.append(f"{index}. {_clean(step)}")
    else:
        lines.append("No build plan recorded yet.")

    lines.extend(["", "### Tool Trace", ""])
    if trace:
        for index, event in enumerate(trace[-5:], start=max(1, len(trace) - 4)):
            tools = " -> ".join(
                _clean(tool.get("name")) for tool in _list_of_dicts(event.get("tools")) if tool.get("name")
            )
            lines.append(f"- Turn {index}: {_clean(event.get('input'))} [{tools or 'reply'}]")
    else:
        lines.append("- No tool trace recorded yet.")
    lines.append("")
    return lines


def _social_post(title: str, verdict: str, ledger: dict[str, Any]) -> list[str]:
    ready = sum(1 for badge in _list_of_dicts(ledger.get("badges")) if badge.get("status") == "ready")
    total = len(_list_of_dicts(ledger.get("badges")))
    return [
        "## Social Post Draft",
        "",
        (
            "I built Hackathon Advisor, a small-model originality coach for Build Small. "
            f"The Unwritten Almanac checks `{title}` against a local snapshot of real Spaces, cites overlap, "
            f"and exports the evidence trail. Latest seal: {verdict}. Prize ledger: {ready}/{total} ready."
        ),
        "",
        f"Live: {LIVE_URL}",
        "",
    ]


def _open_gaps(ledger: dict[str, Any]) -> list[str]:
    pending = [
        badge
        for badge in _list_of_dicts(ledger.get("badges"))
        if badge.get("status") not in {"ready", "eligible"}
    ]
    lines = ["## Open Gaps", ""]
    if pending:
        for badge in pending:
            lines.append(f"- {_clean(badge.get('name'))}: {_clean(badge.get('status'))} - {_clean(badge.get('evidence'))}")
    else:
        lines.append("- No non-ready badge states reported by the current ledger.")
    lines.append("")
    return lines


def _current_idea(session: dict[str, Any], ideas: list[dict[str, Any]]) -> dict[str, Any]:
    current_id = _clean(session.get("current_idea_id"))
    if current_id:
        for idea in ideas:
            if _clean(idea.get("id")) == current_id:
                return idea
    if ideas:
        return ideas[-1]
    return {}


def _echoes(idea: dict[str, Any]) -> list[dict[str, Any]]:
    score = idea.get("score") if isinstance(idea.get("score"), dict) else {}
    return _list_of_dicts(score.get("echoes"))


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
