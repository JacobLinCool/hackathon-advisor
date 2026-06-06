from __future__ import annotations

from hashlib import sha256
from typing import Any

from hackathon_advisor.data import Project, ProjectIndex, SearchHit
from hackathon_advisor.scoring import ScoreCard
from hackathon_advisor.tools import Idea


def build_wood_map(index: ProjectIndex, idea: Idea, score: ScoreCard) -> dict[str, Any]:
    echoes = list(score.echoes)
    background = _background_projects(index, echoes)
    dots = [_project_dot(project, "inked") for project in background]
    dots.extend(_echo_dot(hit) for hit in echoes[:5])
    dots.append(_idea_dot(idea, score, echoes))
    return {
        "caption": _caption(score, echoes),
        "dots": _dedupe_dots(dots),
    }


def _background_projects(index: ProjectIndex, echoes: list[SearchHit]) -> list[Project]:
    echo_ids = {hit.project.id for hit in echoes}
    projects = [project for project in index.top_projects(limit=22) if project.id not in echo_ids]
    return projects[:16]


def _project_dot(project: Project, kind: str) -> dict[str, Any]:
    x, y = _point(project.id)
    return {
        "id": project.id,
        "kind": kind,
        "title": project.title,
        "url": project.url,
        "x": x,
        "y": y,
        "radius": 3,
    }


def _echo_dot(hit: SearchHit) -> dict[str, Any]:
    dot = _project_dot(hit.project, "echo")
    dot["score"] = round(hit.score, 3)
    dot["matched_terms"] = list(hit.matched_terms)
    dot["radius"] = max(5, min(9, round(4 + hit.score * 14)))
    return dot


def _idea_dot(idea: Idea, score: ScoreCard, echoes: list[SearchHit]) -> dict[str, Any]:
    if echoes and not score.verdict.startswith("UNWRITTEN"):
        lead_x, lead_y = _point(echoes[0].project.id)
        x = _clamp(lead_x + 7, 8, 92)
        y = _clamp(lead_y - 5, 8, 92)
    else:
        x, y = _point(f"idea:{idea.id}:{idea.title}")
    return {
        "id": idea.id,
        "kind": "idea",
        "title": idea.title,
        "x": x,
        "y": y,
        "radius": 8,
        "verdict": score.verdict,
        "overall": score.overall,
    }


def _caption(score: ScoreCard, echoes: list[SearchHit]) -> str:
    if score.verdict.startswith("UNWRITTEN"):
        return "Your page sits in a pale margin beyond the nearest inked clusters."
    names = ", ".join(hit.project.title for hit in echoes[:2]) or "nearby pages"
    return f"Your page is pressed close to {names}; the red dots are the strongest echoes."


def _point(key: str) -> tuple[int, int]:
    digest = sha256(key.encode("utf-8")).hexdigest()
    x = 8 + int(digest[:4], 16) % 84
    y = 8 + int(digest[4:8], 16) % 84
    return x, y


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _dedupe_dots(dots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for dot in dots:
        key = (str(dot.get("kind")), str(dot.get("id")))
        if key in seen:
            continue
        deduped.append(dot)
        seen.add(key)
    return deduped
