"""Read-only query layer over one immutable dashboard snapshot.

The atlas chat feature (and any future consumer) asks questions like "what is
everyone building", "which projects completed the most quests", or "what does the
Voice cluster contain". Those queries belong in one place — not in tool prompts,
not in route handlers — so this module wraps a single dashboard snapshot
(``dashboard_payload`` + its ``DashboardSearchIndex``) behind typed query methods
that return plain JSON-ready dicts.

A repository instance never touches module globals, locks, or models: the caller
captures a consistent snapshot (app.py does so under ``_runtime_lock``, the same
pattern as ``/api/dashboard/search``) and constructs the repository outside the
lock. Quest data may be absent (``quest_report.status == "not_analyzed"``); every
method degrades to empty-but-well-formed results in that case.
"""

from __future__ import annotations

from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any

from hackathon_advisor._text import clean, list_of_dicts
from hackathon_advisor.dashboard_search import DashboardSearchIndex
from hackathon_advisor.data import (
    normalize_project_tags,
    public_project_summary,
    public_project_title,
)
from hackathon_advisor.quest_taxonomy import (
    build_app_segment,
    build_readme_segment,
    canonical_quest_id,
    quest_label,
    quest_profiles,
)

CLUSTER_LABEL_MATCH_THRESHOLD = 0.6
DEFAULT_LEADERBOARD_LIMIT = 8
DEFAULT_SEARCH_LIMIT = 8
DEFAULT_RECENT_LIMIT = 6
DEFAULT_EXAMPLE_LIMIT = 6
README_EXCERPT_CHARS = 1500
APP_EXCERPT_CHARS = 1900


class DashboardRepository:
    """Pure queries over one dashboard snapshot; safe to use without locks."""

    def __init__(
        self, dashboard_payload: Mapping[str, Any], search_index: DashboardSearchIndex
    ) -> None:
        self._payload = dashboard_payload
        self._search_index = search_index
        self._points = list_of_dicts(dashboard_payload.get("points"))
        self._clusters = list_of_dicts(dashboard_payload.get("clusters"))
        quest_report = dashboard_payload.get("quest_report")
        self._quest_report = quest_report if isinstance(quest_report, Mapping) else {}
        self._cluster_label_by_id = {
            str(cluster.get("id") or ""): clean(cluster.get("label")) for cluster in self._clusters
        }
        # Full Project objects (incl. readme_body / app_file_source, which the public
        # dashboard points strip) ride along inside the search index's documents.
        self._project_by_id = {
            document.project.id: document.project for document in search_index.documents
        }
        self._point_by_id = {str(point.get("id") or ""): point for point in self._points}

    def quests_analyzed(self) -> bool:
        return str(self._quest_report.get("status") or "") == "analyzed"

    def overview(self) -> dict[str, Any]:
        """Field-wide counts plus the brightest clusters, quests, and projects."""
        top_quests = [
            {"id": quest["id"], "label": quest["label"], "project_count": quest["project_count"]}
            for quest in self._quests_by_coverage()[:3]
            if quest["project_count"] > 0
        ]
        most_liked = sorted(
            self._points,
            key=lambda point: (int(point.get("likes") or 0), clean(point.get("title")).casefold()),
            reverse=True,
        )[:3]
        return {
            "project_count": int(self._payload.get("project_count") or len(self._points)),
            "cluster_count": len(self._clusters),
            "generated_at": str(self._payload.get("generated_at") or ""),
            "quest_status": str(self._quest_report.get("status") or "not_analyzed"),
            "top_clusters": [
                {
                    "label": clean(cluster.get("label")),
                    "project_count": int(cluster.get("project_count") or 0),
                }
                for cluster in self._clusters[:3]
            ],
            "top_quests": top_quests,
            "most_liked": [self._project_row(point) for point in most_liked],
        }

    def list_clusters(self) -> dict[str, Any]:
        return {
            "cluster_count": len(self._clusters),
            "clusters": [
                {
                    "label": clean(cluster.get("label")),
                    "project_count": int(cluster.get("project_count") or 0),
                    "keywords": [clean(keyword) for keyword in (cluster.get("keywords") or [])[:4]],
                }
                for cluster in self._clusters
            ],
        }

    def cluster_detail(self, label: str) -> dict[str, Any] | None:
        """Resolve a cluster by (fuzzy) label or id; cluster ids are unstable across refreshes."""
        cluster = self._resolve_cluster(label)
        if cluster is None:
            return None
        examples = list_of_dicts(cluster.get("representative_projects"))[:DEFAULT_EXAMPLE_LIMIT]
        return {
            "label": clean(cluster.get("label")),
            "project_count": int(cluster.get("project_count") or 0),
            "keywords": [clean(keyword) for keyword in (cluster.get("keywords") or [])[:5]],
            "examples": [self._project_row(example) for example in examples],
        }

    def list_quests(self) -> dict[str, Any]:
        return {
            "status": str(self._quest_report.get("status") or "not_analyzed"),
            "quests": self._quests_by_coverage(),
        }

    def quest_detail(self, quest: str) -> dict[str, Any] | None:
        try:
            quest_id = canonical_quest_id(quest)
        except ValueError:
            quest_id = self._find_quest_in_text(quest)
            if quest_id is None:
                return None
        report_entry = next(
            (
                entry
                for entry in list_of_dicts(self._quest_report.get("quests"))
                if str(entry.get("id") or "") == quest_id
            ),
            {},
        )
        profile = next(
            (profile for profile in quest_profiles() if profile["id"] == quest_id),
            {"id": quest_id, "label": quest_id, "description": ""},
        )
        matched = [point for point in self._points if quest_id in (point.get("quest_ids") or [])]
        examples = list_of_dicts(report_entry.get("examples"))[:DEFAULT_EXAMPLE_LIMIT] or [
            self._project_row(point) for point in matched[:DEFAULT_EXAMPLE_LIMIT]
        ]
        return {
            "id": quest_id,
            "label": profile["label"],
            "description": profile["description"],
            "status": str(self._quest_report.get("status") or "not_analyzed"),
            "project_count": int(report_entry.get("project_count") or len(matched)),
            "examples": [self._project_row(example) for example in examples],
        }

    def top_by_quests(self, limit: int = DEFAULT_LEADERBOARD_LIMIT) -> dict[str, Any]:
        """Per-project quest leaderboard (projects ARE the teams: no author field exists)."""
        rows = [
            {
                **self._project_row(point),
                "quest_count": len(point.get("quest_ids") or []),
                "quest_ids": [str(quest) for quest in point.get("quest_ids") or []],
            }
            for point in self._points
            if point.get("quest_ids")
        ]
        rows.sort(
            key=lambda row: (row["quest_count"], row["likes"], row["title"].casefold()),
            reverse=True,
        )
        return {
            "status": str(self._quest_report.get("status") or "not_analyzed"),
            "rows": rows[: max(1, int(limit))],
            "projects_with_quests": len(rows),
        }

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict[str, Any]:
        payload = self._search_index.search(clean(query), limit=max(1, int(limit)))
        return {
            "query": payload["query"],
            "total": int(payload["total"]),
            "results": [
                {
                    "id": str(result.get("project_id") or ""),
                    "title": clean(result.get("title")),
                    "summary": clean(result.get("summary")),
                    "url": str(result.get("url") or ""),
                    "score": float(result.get("score") or 0.0),
                }
                for result in payload["results"]
            ],
        }

    def recent_activity(self, limit: int = DEFAULT_RECENT_LIMIT) -> dict[str, Any]:
        ordered = sorted(
            self._points,
            key=lambda point: str(point.get("last_modified") or ""),
            reverse=True,
        )[: max(1, int(limit))]
        return {
            "projects": [
                {
                    **self._project_row(point),
                    "last_modified": str(point.get("last_modified") or ""),
                    "cluster_label": self._cluster_label_by_id.get(
                        str(point.get("cluster_id") or ""), ""
                    ),
                }
                for point in ordered
            ],
        }

    def _quests_by_coverage(self) -> list[dict[str, Any]]:
        entries = [
            {
                "id": str(entry.get("id") or ""),
                "label": clean(entry.get("label")) or str(entry.get("id") or ""),
                "description": clean(entry.get("description")),
                "project_count": int(entry.get("project_count") or 0),
            }
            for entry in list_of_dicts(self._quest_report.get("quests"))
        ]
        return sorted(
            entries, key=lambda entry: (-entry["project_count"], entry["label"].casefold())
        )

    def project_detail(self, name: str) -> dict[str, Any] | None:
        """One project's card plus its README and main-app-file excerpts.

        The excerpts reuse the quest classifier's prompt view (build_readme_segment /
        build_app_segment) — the same budgeted slices MiniCPM already reads well."""
        project = self._resolve_project(name)
        if project is None:
            return None
        point = self._point_by_id.get(project.id, {})
        app_excerpt = _clip_excerpt(
            build_app_segment(project.app_file_source, project.app_file_embedding_text),
            APP_EXCERPT_CHARS,
        )
        return {
            "id": project.id,
            "title": public_project_title(project.title),
            "summary": public_project_summary(project.summary),
            "url": project.url,
            "likes": project.likes,
            "sdk": project.sdk,
            "models": list(project.models)[:4],
            "tags": list(normalize_project_tags(project.tags))[:6],
            "last_modified": project.last_modified,
            "cluster_label": self._cluster_label_by_id.get(str(point.get("cluster_id") or ""), ""),
            "quests": [quest_label(str(quest)) for quest in point.get("quest_ids") or []],
            "readme_excerpt": _clip_excerpt(
                build_readme_segment(project.readme_body), README_EXCERPT_CHARS
            ),
            "app_file": project.app_file,
            "app_excerpt": app_excerpt,
        }

    def _resolve_project(self, name: str) -> Any | None:
        """Match a project by id, slug, or title — exact first, then embedded in a
        longer question ("tell me about Jawbreaker"), longest title winning."""
        wanted = clean(name).casefold()
        if not wanted:
            return None
        for project in self._project_by_id.values():
            slug = project.id.rsplit("/", 1)[-1]
            if wanted in (project.id.casefold(), slug.casefold()):
                return project
            if public_project_title(project.title).casefold() == wanted:
                return project
        best, best_length = None, 0
        for project in self._project_by_id.values():
            title = public_project_title(project.title).casefold()
            slug = project.id.rsplit("/", 1)[-1].casefold()
            for candidate in (title, slug):
                if len(candidate) > 3 and candidate in wanted and len(candidate) > best_length:
                    best, best_length = project, len(candidate)
        return best

    def _find_quest_in_text(self, text: str) -> str | None:
        """Spot a quest id or label embedded in a longer question."""
        wanted = clean(text).casefold()
        if not wanted:
            return None
        for profile in quest_profiles():
            if profile["id"].casefold() in wanted or profile["label"].casefold() in wanted:
                return profile["id"]
        return None

    def _resolve_cluster(self, label: str) -> Mapping[str, Any] | None:
        wanted = clean(label).casefold()
        if not wanted:
            return None
        for cluster in self._clusters:
            if str(cluster.get("id") or "").casefold() == wanted:
                return cluster
        for cluster in self._clusters:
            if clean(cluster.get("label")).casefold() == wanted:
                return cluster
        for cluster in self._clusters:
            cluster_label = clean(cluster.get("label")).casefold()
            if wanted in cluster_label or cluster_label in wanted:
                return cluster
        for cluster in self._clusters:
            keywords = {clean(keyword).casefold() for keyword in cluster.get("keywords") or []}
            if any(token in keywords for token in wanted.split()):
                return cluster
        best, best_score = None, 0.0
        for cluster in self._clusters:
            score = SequenceMatcher(None, wanted, clean(cluster.get("label")).casefold()).ratio()
            if score > best_score:
                best, best_score = cluster, score
        return best if best_score >= CLUSTER_LABEL_MATCH_THRESHOLD else None

    def _project_row(self, point: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": str(point.get("id") or ""),
            "title": clean(point.get("title")) or str(point.get("id") or ""),
            "url": str(point.get("url") or ""),
            "likes": int(point.get("likes") or 0),
        }


def _clip_excerpt(text: str, limit: int) -> str:
    # Newlines stay (app files read as code); only the length is bounded.
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + " ..."
