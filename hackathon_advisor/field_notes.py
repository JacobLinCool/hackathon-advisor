from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hackathon_advisor.tools import goal_label


def build_field_notes_markdown(session: dict[str, Any], metadata: dict[str, Any]) -> str:
    ideas = _list_of_dicts(session.get("ideas"))
    trace = _list_of_dicts(session.get("trace"))
    profile = session.get("profile") if isinstance(session.get("profile"), dict) else {}
    goals = _goal_labels(session.get("goals"))
    last_plan = [str(step) for step in session.get("last_plan") or []]
    last_artifact = session.get("last_artifact") if isinstance(session.get("last_artifact"), dict) else {}

    lines = [
        "# Hackathon Advisor Field Notes",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Snapshot",
        "",
        f"- Project snapshot: {_clean(metadata.get('snapshot_generated_at'))}",
        f"- Project count: {_clean(metadata.get('project_count', 'unknown'))}",
        f"- Index: {_clean(metadata.get('index_algorithm'))} at {_clean(metadata.get('index_generated_at'))}",
        f"- Digest: `{_clean(metadata.get('snapshot_digest'))}`",
        "",
        "## Builder Context",
        "",
    ]

    if profile:
        for field in ("skills", "time", "preferences", "constraints"):
            value = _clean(profile.get(field))
            if value:
                lines.append(f"- {field.title()}: {value}")
    else:
        lines.append("- No profile notes recorded.")
    lines.append(f"- Goals: {', '.join(goals) if goals else 'No specific goals'}")

    lines.extend(["", "## Idea Board", ""])
    if ideas:
        for index, idea in enumerate(ideas, start=1):
            lines.extend(_idea_section(index, idea))
    else:
        lines.append("No ideas were written.")

    lines.extend(["", "## Build Plan", ""])
    if last_plan:
        for index, step in enumerate(last_plan, start=1):
            lines.append(f"{index}. {_clean(step)}")
    else:
        lines.append("No build plan has been generated yet.")

    lines.extend(["", "## Turn Trace", ""])
    if trace:
        for index, event in enumerate(trace, start=1):
            lines.extend(_trace_section(index, event))
    else:
        lines.append("No tool trace was recorded.")

    wood_map = last_artifact.get("wood_map") if isinstance(last_artifact.get("wood_map"), dict) else {}
    if wood_map:
        lines.extend(["", "## Wood Map", "", _clean(wood_map.get("caption"))])
        for dot in _list_of_dicts(wood_map.get("dots")):
            if dot.get("kind") != "echo":
                continue
            title = _clean(dot.get("title"))
            score = _clean(dot.get("score"))
            url = _clean(dot.get("url"))
            page = _clean(dot.get("page_number")) or "?"
            if url:
                lines.append(f"- Page {page}: [{title}]({url}) - echo score {score}")
            else:
                lines.append(f"- Page {page}: {title} - echo score {score}")

    if last_artifact:
        lines.extend(
            [
                "",
                "## Share Caption",
                "",
                _clean(last_artifact.get("caption") or ""),
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _idea_section(index: int, idea: dict[str, Any]) -> list[str]:
    title = _clean(idea.get("title") or f"Idea {index}")
    pitch = _clean(idea.get("pitch"))
    goals = _goal_labels(idea.get("goals"))
    score = idea.get("score") if isinstance(idea.get("score"), dict) else {}
    lines = [
        f"### {index}. {title}",
        "",
        f"- Pitch: {pitch or 'No pitch recorded.'}",
        f"- Goals: {', '.join(goals) if goals else 'No specific goals'}",
    ]
    if score:
        lines.append(f"- Seal: {_clean(score.get('overall'))}/10 - {_clean(score.get('verdict'))}")
        echoes = _list_of_dicts(score.get("echoes"))
        if echoes:
            lines.append("- Closest cited Spaces:")
            for echo in echoes[:4]:
                project = echo.get("project") if isinstance(echo.get("project"), dict) else {}
                title = _clean(project.get("title") or project.get("id") or "Untitled Space")
                url = _clean(project.get("url") or project.get("host") or "")
                matched = ", ".join(str(term) for term in echo.get("matched_terms") or [])
                score_text = _clean(echo.get("score"))
                page = _clean(echo.get("page_number")) or "?"
                if url:
                    lines.append(
                        f"  - Page {page}: [{title}]({url}) - score {score_text}; "
                        f"matched {matched or 'no shared terms'}"
                    )
                else:
                    lines.append(f"  - Page {page}: {title} - score {score_text}; matched {matched or 'no shared terms'}")
    lines.append("")
    return lines


def _trace_section(index: int, event: dict[str, Any]) -> list[str]:
    tools = _list_of_dicts(event.get("tools"))
    tool_names = " -> ".join(_clean(tool.get("name")) for tool in tools if tool.get("name")) or "reply"
    lines = [
        f"### Turn {index}",
        "",
        f"- Input: {_clean(event.get('input'))}",
        f"- Tools: {tool_names}",
    ]
    verdict = _clean(event.get("verdict"))
    overall = event.get("overall")
    if verdict or overall is not None:
        lines.append(f"- Verdict: {verdict or 'n/a'} {overall if overall is not None else ''}".rstrip())
    response = _clean(event.get("response"))
    if response:
        lines.append(f"- Advisor note: {response}")
    resolution = event.get("tool_resolution") if isinstance(event.get("tool_resolution"), dict) else {}
    call = resolution.get("call") if isinstance(resolution.get("call"), dict) else {}
    if call:
        lines.append(f"- Planner call: `{_clean(call.get('name'))}`")
    lines.append("")
    return lines


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _goal_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [goal_label(str(goal)) for goal in value]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
