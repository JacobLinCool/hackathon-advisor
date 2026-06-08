from __future__ import annotations

from typing import Any

from hackathon_advisor.tools import goal_label
from hackathon_advisor._text import clean as _clean, list_of_dicts as _list_of_dicts, utc_now


def build_chapter_markdown(session: dict[str, Any], metadata: dict[str, Any]) -> str:
    ideas = _list_of_dicts(session.get("ideas"))
    goals = _goal_labels(session.get("goals"))
    artifact = session.get("last_artifact") if isinstance(session.get("last_artifact"), dict) else {}
    lines = [
        "# The Unwritten Almanac Chapter",
        "",
        f"Generated: {utc_now()}",
        f"Snapshot: {_clean(metadata.get('snapshot_generated_at'))} · {_clean(metadata.get('project_count'))} pages",
        f"Goals: {', '.join(goals) if goals else 'No specific goals'}",
        "",
    ]

    if not ideas:
        lines.extend(["No idea pages have been written yet.", ""])
        return "\n".join(lines)

    for index, idea in enumerate(ideas, start=1):
        lines.extend(_idea_page(index, idea))

    caption = _clean(artifact.get("caption")) if artifact else ""
    if caption:
        lines.extend(["## Share Caption", "", caption, ""])
    return "\n".join(lines).rstrip() + "\n"


def _idea_page(index: int, idea: dict[str, Any]) -> list[str]:
    title = _clean(idea.get("title") or f"Page {index}")
    pitch = _clean(idea.get("pitch"))
    goals = _goal_labels(idea.get("goals"))
    score = idea.get("score") if isinstance(idea.get("score"), dict) else {}
    verdict = _clean(score.get("verdict")) if score else "DRAFT"
    overall = _clean(score.get("overall")) if score else "0.0"
    lines = [
        f"## Page {index}: {title}",
        "",
        f"Verdict: {verdict} · {overall}/10",
        f"Goals: {', '.join(goals) if goals else 'No specific goals'}",
        "",
        pitch or "No pitch recorded.",
        "",
    ]
    echoes = _list_of_dicts(score.get("echoes")) if score else []
    if echoes:
        lines.extend(["Closest cited pages:", ""])
        for echo in echoes[:3]:
            project = echo.get("project") if isinstance(echo.get("project"), dict) else {}
            page = _clean(echo.get("page_number")) or "?"
            project_title = _clean(project.get("title") or project.get("id") or "Untitled")
            url = _clean(project.get("url") or project.get("host") or "")
            score_text = _clean(echo.get("score"))
            if url:
                lines.append(f"- Page {page}: [{project_title}]({url}) · echo {score_text}")
            else:
                lines.append(f"- Page {page}: {project_title} · echo {score_text}")
        lines.append("")
    return lines


def _goal_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [goal_label(str(goal)) for goal in value]
