from __future__ import annotations

from hashlib import sha256
from typing import Any

from hackathon_advisor.data import Project, ProjectIndex, SearchHit
from hackathon_advisor.scoring import ScoreCard
from hackathon_advisor.tools import Idea


def build_wood_map(index: ProjectIndex, idea: Idea, score: ScoreCard) -> dict[str, Any]:
    echoes = list(score.echoes)
    background = _background_projects(index, echoes)
    echo_projects = [hit.project for hit in echoes[:5]]

    layout, idea_xy = _layout(index, idea, background + echo_projects)

    dots = [_project_dot(project, "inked", layout) for project in background]
    dots.extend(_echo_dot(hit, layout) for hit in echoes[:5])
    dots.append(_idea_dot(idea, score, idea_xy))
    return {
        "caption": _caption(score, echoes),
        "dots": _dedupe_dots(dots),
    }


def _background_projects(index: ProjectIndex, echoes: list[SearchHit]) -> list[Project]:
    echo_ids = {hit.project.id for hit in echoes}
    projects = [project for project in index.top_projects(limit=22) if project.id not in echo_ids]
    return projects[:16]


def _project_dot(project: Project, kind: str, layout: dict[str, tuple[int, int]]) -> dict[str, Any]:
    x, y = layout.get(project.id) or _point(project.id)
    return {
        "id": project.id,
        "kind": kind,
        "title": project.title,
        "url": project.url,
        "x": x,
        "y": y,
        "radius": 3,
    }


def _echo_dot(hit: SearchHit, layout: dict[str, tuple[int, int]]) -> dict[str, Any]:
    dot = _project_dot(hit.project, "echo", layout)
    dot["score"] = round(hit.score, 3)
    dot["matched_terms"] = list(hit.matched_terms)
    dot["page_number"] = hit.page_number
    dot["radius"] = max(5, min(9, round(4 + hit.score * 14)))
    return dot


def _idea_dot(idea: Idea, score: ScoreCard, idea_xy: tuple[int, int]) -> dict[str, Any]:
    x, y = idea_xy
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


def _layout(
    index: ProjectIndex,
    idea: Idea,
    projects: list[Project],
) -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
    """Place every dot by projecting the real embedding vectors into 2D with PCA, so projects
    that are semantically similar land near each other and the idea lands among its closest
    echoes. Falls back to a deterministic hash scatter only when the projection cannot run
    (missing vectors, too few points, or no embedder)."""
    ids = [project.id for project in projects]
    vectors = [index.vector_for(project.id) for project in projects]
    fallback = ({project_id: _point(project_id) for project_id in ids}, _point(f"idea:{idea.id}:{idea.title}"))
    if len(vectors) < 3 or any(vector is None for vector in vectors):
        return fallback
    try:
        idea_vector = index.embed_query(idea.pitch or idea.title)
        coords, idea_xy = _pca_project(vectors, idea_vector)
    except Exception:  # noqa: BLE001 - any projection failure degrades to the hash scatter
        return fallback
    return {project_id: coord for project_id, coord in zip(ids, coords)}, idea_xy


def _pca_project(
    vectors: list[tuple[float, ...]],
    idea_vector: tuple[float, ...],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(idea_vector, dtype=np.float64)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # Top-2 principal directions of the project cloud define the map; the idea is projected
    # onto that same basis so its position reflects true embedding similarity.
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    basis = components[:2]
    projected = centered @ basis.T
    idea_projected = (query - mean) @ basis.T
    stacked = np.vstack([projected, idea_projected])
    scaled = _scale_to_canvas(stacked)
    coords = [(int(round(x)), int(round(y))) for x, y in scaled[:-1]]
    idea_xy = (int(round(scaled[-1][0])), int(round(scaled[-1][1])))
    return coords, idea_xy


def _scale_to_canvas(points: Any, low: float = 10.0, high: float = 90.0) -> Any:
    import numpy as np

    scaled = np.empty_like(points)
    for axis in range(points.shape[1]):
        column = points[:, axis]
        lo = float(column.min())
        hi = float(column.max())
        span = hi - lo
        if span < 1e-9:
            scaled[:, axis] = (low + high) / 2.0
        else:
            scaled[:, axis] = low + (column - lo) / span * (high - low)
    return scaled


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
