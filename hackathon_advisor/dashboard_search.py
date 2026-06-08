from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any

from hackathon_advisor.data import Project, public_project_summary, public_project_title


SEARCH_SCHEMA_VERSION = 1
SEARCH_ALGORITHM = "bm25-text-v1"
DEFAULT_SEARCH_LIMIT = 12
MAX_SEARCH_LIMIT = 30
BM25_K1 = 1.35
BM25_B = 0.72
MAX_SNIPPET_CHARS = 170

SEARCH_TOKEN_RE = re.compile(r"[\w][\w.+-]*", re.UNICODE)
TOKEN_SPLIT_RE = re.compile(r"[._+\-/]+")
HIGHLIGHT_BOUNDARY_RE = re.compile(r"\s+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class SearchField:
    source: str
    text: str
    weight: float


@dataclass(frozen=True)
class SearchDocument:
    project: Project
    fields: tuple[SearchField, ...]
    term_counts: Counter[str]
    length: float


@dataclass(frozen=True)
class DashboardSearchHit:
    project: Project
    score: float
    matched_terms: tuple[str, ...]
    snippets: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project.to_public_dict(),
            "project_id": self.project.id,
            "title": public_project_title(self.project.title),
            "summary": public_project_summary(self.project.summary),
            "url": self.project.url,
            "score": round(self.score, 4),
            "matched_terms": list(self.matched_terms),
            "snippets": [dict(snippet) for snippet in self.snippets],
        }


class DashboardSearchIndex:
    def __init__(self, projects: Sequence[Project], dashboard_payload: Mapping[str, Any]) -> None:
        point_by_id = _point_by_project_id(dashboard_payload)
        cluster_by_id = _cluster_by_id(dashboard_payload)
        quest_label_by_id = _quest_label_by_id(dashboard_payload)
        self.documents = tuple(
            _build_document(
                project,
                point_by_id,
                cluster_by_id,
                quest_label_by_id,
            )
            for project in projects
        )
        if not self.documents:
            raise ValueError("dashboard search index requires at least one project")
        self.document_count = len(self.documents)
        self.average_length = (
            sum(document.length for document in self.documents) / self.document_count
        )
        self.document_frequency = _document_frequency(self.documents)
        self.index_metadata = {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "algorithm": SEARCH_ALGORITHM,
            "document_count": self.document_count,
        }

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict[str, Any]:
        normalized_query = normalize_query(query)
        terms = tuple(dict.fromkeys(search_tokens(normalized_query)))
        if not terms:
            return {
                "schema_version": SEARCH_SCHEMA_VERSION,
                "algorithm": SEARCH_ALGORITHM,
                "query": normalized_query,
                "total": 0,
                "results": [],
            }

        scored: list[tuple[float, SearchDocument]] = []
        for document in self.documents:
            score = self._score_document(document, terms, normalized_query)
            if score > 0:
                scored.append((score, document))
        scored.sort(
            key=lambda item: (
                item[0],
                item[1].project.likes,
                item[1].project.title.casefold(),
            ),
            reverse=True,
        )
        raw_top_score = scored[0][0] if scored else 0.0
        results = [
            DashboardSearchHit(
                project=document.project,
                score=raw_score / raw_top_score if raw_top_score else 0.0,
                matched_terms=tuple(
                    term for term in terms if document.term_counts.get(term, 0) > 0
                )[:8],
                snippets=tuple(_snippets(document, terms)),
            ).to_dict()
            for raw_score, document in scored[:limit]
        ]
        return {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "algorithm": SEARCH_ALGORITHM,
            "query": normalized_query,
            "total": len(scored),
            "results": results,
        }

    def _score_document(
        self,
        document: SearchDocument,
        terms: Sequence[str],
        normalized_query: str,
    ) -> float:
        score = 0.0
        length = max(document.length, 1.0)
        average_length = max(self.average_length, 1.0)
        for term in terms:
            frequency = float(document.term_counts.get(term, 0.0))
            if frequency <= 0:
                continue
            idf = self._idf(term)
            denominator = frequency + BM25_K1 * (1.0 - BM25_B + BM25_B * length / average_length)
            score += idf * ((frequency * (BM25_K1 + 1.0)) / denominator)

        query_for_exact = normalized_query.casefold()
        if query_for_exact:
            title = public_project_title(document.project.title).casefold()
            slug = document.project.slug.replace("-", " ").replace("_", " ").casefold()
            if query_for_exact in title:
                score += 2.0
            if query_for_exact in slug:
                score += 1.4
        return score

    def _idf(self, term: str) -> float:
        frequency = self.document_frequency.get(term, 0)
        return math.log(1.0 + (self.document_count - frequency + 0.5) / (frequency + 0.5))


def normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())


def normalize_search_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("search limit must be an integer") from error
    if not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise ValueError(f"search limit must be between 1 and {MAX_SEARCH_LIMIT}")
    return limit


def search_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    for raw_token in SEARCH_TOKEN_RE.findall(normalized):
        for token in _token_variants(raw_token):
            if (len(token) <= 1 and not token.isdigit()) or token in STOPWORDS:
                continue
            tokens.append(token)
    return tokens


def _token_variants(raw_token: str) -> tuple[str, ...]:
    cleaned = raw_token.strip("._+-/")
    if not cleaned:
        return ()
    parts = tuple(part for part in TOKEN_SPLIT_RE.split(cleaned) if len(part) > 1)
    if parts and parts != (cleaned,):
        return (cleaned, *parts)
    return (cleaned,)


def _document_frequency(documents: Sequence[SearchDocument]) -> dict[str, int]:
    frequency: Counter[str] = Counter()
    for document in documents:
        frequency.update(document.term_counts.keys())
    return dict(frequency)


def _build_document(
    project: Project,
    point_by_id: Mapping[str, Mapping[str, Any]],
    cluster_by_id: Mapping[str, Mapping[str, Any]],
    quest_label_by_id: Mapping[str, str],
) -> SearchDocument:
    point = point_by_id.get(project.id, {})
    fields = _project_fields(project, point, cluster_by_id, quest_label_by_id)
    term_counts: Counter[str] = Counter()
    for field in fields:
        for token in search_tokens(field.text):
            term_counts[token] += field.weight
    return SearchDocument(
        project=project,
        fields=fields,
        term_counts=term_counts,
        length=sum(term_counts.values()),
    )


def _point_by_project_id(dashboard_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(point.get("id") or ""): point
        for point in dashboard_payload.get("points") or []
        if isinstance(point, Mapping)
    }


def _project_fields(
    project: Project,
    point: Mapping[str, Any],
    cluster_by_id: Mapping[str, Mapping[str, Any]],
    quest_labels: Mapping[str, str],
) -> tuple[SearchField, ...]:
    cluster = cluster_by_id.get(str(point.get("cluster_id") or ""), {})
    quest_texts = []
    for match in point.get("quest_matches") or []:
        if not isinstance(match, Mapping):
            continue
        quest = str(match.get("quest") or "")
        quest_texts.append(
            " ".join(
                [
                    quest_labels.get(quest, quest),
                    str(match.get("evidence") or ""),
                ]
            ).strip()
        )

    return tuple(
        field
        for field in [
            SearchField("Title", public_project_title(project.title), 4.0),
            SearchField(
                "Space",
                " ".join(
                    [
                        project.id,
                        project.slug,
                        project.slug.replace("-", " ").replace("_", " "),
                    ]
                ),
                3.2,
            ),
            SearchField("Summary", public_project_summary(project.summary), 2.4),
            SearchField(
                "Tags",
                " ".join([*project.tags, *project.models, *project.datasets, project.sdk]),
                2.0,
            ),
            SearchField(
                "Cluster",
                " ".join(
                    [
                        str(cluster.get("label") or ""),
                        " ".join(str(keyword) for keyword in cluster.get("keywords") or []),
                    ]
                ),
                1.4,
            ),
            SearchField("Quest evidence", " ".join(quest_texts), 1.6),
            SearchField(
                "App",
                " ".join(
                    [
                        project.app_file,
                        project.app_file_embedding_text,
                        project.app_file_source,
                    ]
                ),
                1.0,
            ),
            SearchField("README", project.readme_body, 0.9),
        ]
        if field.text.strip()
    )


def _cluster_by_id(dashboard_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(cluster.get("id") or ""): cluster
        for cluster in dashboard_payload.get("clusters") or []
        if isinstance(cluster, Mapping)
    }


def _quest_label_by_id(dashboard_payload: Mapping[str, Any]) -> dict[str, str]:
    quest_report = dashboard_payload.get("quest_report")
    if not isinstance(quest_report, Mapping):
        return {}
    return {
        str(quest.get("id") or ""): str(quest.get("label") or quest.get("id") or "")
        for quest in quest_report.get("quests") or []
        if isinstance(quest, Mapping)
    }


def _snippets(document: SearchDocument, terms: Sequence[str]) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    for field in document.fields:
        field_terms = set(search_tokens(field.text))
        if not field_terms.intersection(terms):
            continue
        if field.source in seen_sources:
            continue
        snippet = _field_snippet(field.text, terms)
        if not snippet:
            continue
        snippets.append({"source": field.source, "text": snippet})
        seen_sources.add(field.source)
        if len(snippets) >= 2:
            break
    return snippets


def _field_snippet(text: str, terms: Sequence[str]) -> str:
    cleaned = HIGHLIGHT_BOUNDARY_RE.sub(" ", str(text or "")).strip()
    if not cleaned:
        return ""
    folded = unicodedata.normalize("NFKC", cleaned).casefold()
    indexes = [folded.find(term) for term in terms if folded.find(term) >= 0]
    center = min(indexes) if indexes else 0
    start = max(0, center - MAX_SNIPPET_CHARS // 2)
    end = min(len(cleaned), start + MAX_SNIPPET_CHARS)
    start = max(0, end - MAX_SNIPPET_CHARS)
    snippet = cleaned[start:end].strip()
    if start > 0:
        snippet = f"... {snippet}"
    if end < len(cleaned):
        snippet = f"{snippet} ..."
    return snippet
