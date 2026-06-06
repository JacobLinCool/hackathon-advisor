from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_chapter_markdown(session: dict[str, Any], metadata: dict[str, Any]) -> str:
    ideas = _list_of_dicts(session.get("ideas"))
    targets = [str(target) for target in session.get("targets") or []]
    artifact = session.get("last_artifact") if isinstance(session.get("last_artifact"), dict) else {}
    lines = [
        "# The Unwritten Almanac Chapter",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Snapshot: {_clean(metadata.get('snapshot_generated_at'))} · {_clean(metadata.get('project_count'))} pages",
        f"Targets: {', '.join(targets) if targets else 'No specific targets'}",
        "",
    ]

    if not ideas:
        lines.extend(["No fate pages have been written yet.", ""])
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
    targets = [str(target) for target in idea.get("targets") or []]
    score = idea.get("score") if isinstance(idea.get("score"), dict) else {}
    verdict = _clean(score.get("verdict")) if score else "DRAFT"
    overall = _clean(score.get("overall")) if score else "0.0"
    lines = [
        f"## Page {index}: {title}",
        "",
        f"Verdict: {verdict} · {overall}/10",
        f"Targets: {', '.join(targets) if targets else 'No specific targets'}",
        "",
        pitch or "No prophecy text recorded.",
        "",
    ]
    echoes = _list_of_dicts(score.get("echoes")) if score else []
    if echoes:
        lines.extend(["Closest inked pages:", ""])
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


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
