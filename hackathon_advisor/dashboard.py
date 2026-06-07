from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
from typing import Any

from hackathon_advisor.data import Project, ProjectIndex, public_project_summary, public_project_title, tokenize
from hackathon_advisor.quest_taxonomy import QUESTS, normalize_match, quest_profiles


DASHBOARD_SCHEMA_VERSION = 1
TSNE_RANDOM_STATE = 42
TSNE_MIN_PROJECTS = 3
LINKS_PER_PROJECT = 2

STOPWORDS = {
    "agent",
    "app",
    "assistant",
    "build",
    "demo",
    "face",
    "for",
    "from",
    "gradio",
    "hackathon",
    "hugging",
    "local",
    "model",
    "project",
    "small",
    "space",
    "this",
    "with",
}


class DashboardError(ValueError):
    pass


def build_dashboard_payload(
    index: ProjectIndex,
    *,
    quest_matches: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    quest_source: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    projects = list(index.projects)
    if len(projects) < TSNE_MIN_PROJECTS:
        raise DashboardError(f"dashboard atlas requires at least {TSNE_MIN_PROJECTS} projects")

    matrix = _embedding_matrix(index)
    coordinates = _tsne_coordinates(matrix)
    raw_cluster_labels = _cluster_labels(matrix)
    cluster_id_by_raw, clusters = _cluster_payloads(projects, coordinates, raw_cluster_labels)
    normalized_quest_matches = _normalize_quest_matches(projects, quest_matches)
    points = _point_payloads(projects, coordinates, raw_cluster_labels, cluster_id_by_raw, normalized_quest_matches)
    links = _nearest_links(projects, matrix)
    quest_report = _quest_report(points, normalized_quest_matches, quest_source)
    payload = {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_count": len(projects),
        "provenance": {
            "snapshot_generated_at": index.generated_at,
            "snapshot_source": index.source,
            "index_generated_at": index.index_generated_at,
            "index_algorithm": index.index_algorithm,
            "snapshot_digest": index.snapshot_digest,
            "embedding": index.embedding_metadata,
        },
        "layout": {
            "algorithm": "tsne",
            "metric": "cosine",
            "init": "pca",
            "random_state": TSNE_RANDOM_STATE,
            "perplexity": _tsne_perplexity(len(projects)),
        },
        "points": points,
        "links": links,
        "clusters": clusters,
        "quest_report": quest_report,
    }
    validate_dashboard_payload(payload)
    return payload


def validate_dashboard_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != DASHBOARD_SCHEMA_VERSION:
        raise DashboardError("unsupported dashboard schema version")
    project_count = int(payload.get("project_count") or 0)
    if project_count < TSNE_MIN_PROJECTS:
        raise DashboardError("dashboard project count is too small")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != project_count:
        raise DashboardError("dashboard point count does not match project count")
    ids: set[str] = set()
    cluster_ids: set[str] = set()
    for point in points:
        if not isinstance(point, dict):
            raise DashboardError("dashboard points must be objects")
        project_id = str(point.get("id") or "")
        if not project_id or project_id in ids:
            raise DashboardError("dashboard points must have unique project ids")
        ids.add(project_id)
        x = float(point.get("x"))
        y = float(point.get("y"))
        if not 0.0 <= x <= 100.0 or not 0.0 <= y <= 100.0:
            raise DashboardError("dashboard point coordinates must be percentages")
        cluster_id = str(point.get("cluster_id") or "")
        if not cluster_id:
            raise DashboardError("dashboard point cluster id is missing")
        cluster_ids.add(cluster_id)

    clusters = payload.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise DashboardError("dashboard clusters are missing")
    declared_cluster_ids = {str(cluster.get("id") or "") for cluster in clusters if isinstance(cluster, dict)}
    if cluster_ids - declared_cluster_ids:
        raise DashboardError("dashboard points reference missing clusters")

    links = payload.get("links")
    if not isinstance(links, list):
        raise DashboardError("dashboard links must be a list")
    for link in links:
        if str(link.get("source") or "") not in ids or str(link.get("target") or "") not in ids:
            raise DashboardError("dashboard link references an unknown project")

    quest_report = payload.get("quest_report")
    if not isinstance(quest_report, dict):
        raise DashboardError("dashboard quest report is missing")
    if quest_report.get("status") not in {"analyzed", "not_analyzed"}:
        raise DashboardError("dashboard quest report status is invalid")


def _embedding_matrix(index: ProjectIndex) -> Any:
    import numpy as np

    return np.asarray(index.project_vectors(), dtype=np.float32)


def _tsne_coordinates(matrix: Any) -> list[tuple[float, float]]:
    from sklearn.manifold import TSNE

    coords = TSNE(
        n_components=2,
        perplexity=_tsne_perplexity(int(matrix.shape[0])),
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        metric="cosine",
        random_state=TSNE_RANDOM_STATE,
    ).fit_transform(matrix)
    return _scale_points(coords)


def _tsne_perplexity(count: int) -> int:
    return max(2, min(30, count // 4))


def _cluster_labels(matrix: Any) -> list[int]:
    from sklearn.cluster import KMeans

    count = int(matrix.shape[0])
    cluster_count = min(10, max(min(6, count), round(math.sqrt(count))))
    labels = KMeans(
        n_clusters=cluster_count,
        random_state=TSNE_RANDOM_STATE,
        n_init=20,
    ).fit_predict(matrix)
    return [int(label) for label in labels]


def _scale_points(points: Any, low: float = 3.0, high: float = 97.0) -> list[tuple[float, float]]:
    import numpy as np

    scaled = np.empty_like(points, dtype=np.float64)
    for axis in range(points.shape[1]):
        column = points[:, axis]
        minimum = float(column.min())
        maximum = float(column.max())
        span = maximum - minimum
        if span <= 1e-9:
            scaled[:, axis] = (low + high) / 2.0
        else:
            scaled[:, axis] = low + (column - minimum) / span * (high - low)
    return [(round(float(x), 4), round(float(y), 4)) for x, y in scaled]


def _cluster_payloads(
    projects: Sequence[Project],
    coordinates: Sequence[tuple[float, float]],
    raw_labels: Sequence[int],
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(raw_labels):
        grouped[int(label)].append(index)

    ordered_raw_labels = sorted(
        grouped,
        key=lambda label: (-len(grouped[label]), _cluster_center(coordinates, grouped[label])),
    )
    cluster_id_by_raw = {label: f"cluster-{position + 1}" for position, label in enumerate(ordered_raw_labels)}
    clusters: list[dict[str, Any]] = []
    for raw_label in ordered_raw_labels:
        indexes = grouped[raw_label]
        keywords = _cluster_keywords(projects[index] for index in indexes)
        label = " / ".join(word.title() for word in keywords[:2]) if keywords else "Project cluster"
        representatives = sorted(
            (projects[index] for index in indexes),
            key=lambda project: (project.likes, project.last_modified, project.title.lower()),
            reverse=True,
        )[:4]
        clusters.append(
            {
                "id": cluster_id_by_raw[raw_label],
                "label": label,
                "keywords": keywords,
                "project_count": len(indexes),
                "center": {
                    "x": round(sum(coordinates[index][0] for index in indexes) / len(indexes), 4),
                    "y": round(sum(coordinates[index][1] for index in indexes) / len(indexes), 4),
                },
                "representative_projects": [project.to_public_dict() for project in representatives],
            }
        )
    return cluster_id_by_raw, clusters


def _cluster_center(coordinates: Sequence[tuple[float, float]], indexes: Sequence[int]) -> tuple[float, float]:
    return (
        sum(coordinates[index][0] for index in indexes) / len(indexes),
        sum(coordinates[index][1] for index in indexes) / len(indexes),
    )


def _cluster_keywords(projects: Sequence[Project]) -> list[str]:
    counts: Counter[str] = Counter()
    for project in projects:
        text = " ".join(
            [
                project.title,
                project.slug.replace("-", " ").replace("_", " "),
                project.summary,
                " ".join(project.tags),
                " ".join(project.models),
            ]
        )
        counts.update(token for token in tokenize(text) if token not in STOPWORDS and not token.startswith("region"))
    return [token for token, _count in counts.most_common(5)]


def _normalize_quest_matches(
    projects: Sequence[Project],
    quest_matches: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    project_ids = {project.id for project in projects}
    normalized = {project.id: [] for project in projects}
    if quest_matches is None:
        return normalized
    if set(quest_matches) != project_ids:
        missing = sorted(project_ids - set(quest_matches))
        extra = sorted(set(quest_matches) - project_ids)
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} projects")
        if extra:
            detail.append(f"unknown {len(extra)} projects")
        raise DashboardError("quest analysis project coverage is invalid: " + ", ".join(detail))
    for project_id, matches in quest_matches.items():
        normalized[project_id] = [_normalize_quest_match(match) for match in matches]
    return normalized


def _normalize_quest_match(match: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return normalize_match(match)
    except ValueError as error:
        raise DashboardError(f"invalid quest match: {error}") from error


def _point_payloads(
    projects: Sequence[Project],
    coordinates: Sequence[tuple[float, float]],
    raw_labels: Sequence[int],
    cluster_id_by_raw: Mapping[int, str],
    quest_matches: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for project, (x, y), raw_label in zip(projects, coordinates, raw_labels, strict=True):
        matches = list(quest_matches.get(project.id) or [])
        points.append(
            {
                "id": project.id,
                "title": public_project_title(project.title),
                "summary": public_project_summary(project.summary),
                "url": project.url,
                "host": project.host,
                "likes": project.likes,
                "sdk": project.sdk,
                "models": list(project.models),
                "tags": list(project.tags),
                "last_modified": project.last_modified,
                "x": x,
                "y": y,
                "cluster_id": cluster_id_by_raw[int(raw_label)],
                "quest_matches": matches,
                "quest_ids": [str(match["quest"]) for match in matches],
            }
        )
    return points


def _nearest_links(projects: Sequence[Project], matrix: Any) -> list[dict[str, Any]]:
    import numpy as np

    similarity = matrix @ matrix.T
    pairs: dict[tuple[int, int], float] = {}
    for index in range(len(projects)):
        order = np.argsort(similarity[index])[::-1]
        neighbors = [int(candidate) for candidate in order if int(candidate) != index][:LINKS_PER_PROJECT]
        for neighbor in neighbors:
            left, right = sorted((index, neighbor))
            pairs[(left, right)] = max(float(similarity[left, right]), pairs.get((left, right), -1.0))
    return [
        {
            "source": projects[left].id,
            "target": projects[right].id,
            "score": round(max(0.0, min(1.0, score)), 4),
        }
        for (left, right), score in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))
    ]


def _quest_report(
    points: Sequence[Mapping[str, Any]],
    quest_matches: Mapping[str, Sequence[Mapping[str, Any]]],
    quest_source: str,
) -> dict[str, Any]:
    profiles = {profile["id"]: profile for profile in quest_profiles()}
    status = "analyzed" if quest_source else "not_analyzed"
    quests = []
    for quest in QUESTS:
        matched_points = [
            point
            for point in points
            if any(match["quest"] == quest for match in quest_matches.get(str(point["id"]), []))
        ]
        examples = sorted(
            matched_points,
            key=lambda point: (
                max(
                    (
                        float(match["confidence"])
                        for match in quest_matches.get(str(point["id"]), [])
                        if match["quest"] == quest
                    ),
                    default=0.0,
                ),
                int(point.get("likes") or 0),
            ),
            reverse=True,
        )[:4]
        profile = profiles.get(quest, {"label": quest, "description": ""})
        quests.append(
            {
                "id": quest,
                "label": profile["label"],
                "description": profile["description"],
                "project_count": len(matched_points),
                "examples": [
                    {
                        "id": point["id"],
                        "title": point["title"],
                        "url": point["url"],
                    }
                    for point in examples
                ],
            }
        )
    return {
        "status": status,
        "source": quest_source,
        "quests": quests,
    }
