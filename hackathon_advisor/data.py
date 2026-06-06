from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.+_-]*", re.IGNORECASE)


@dataclass(frozen=True)
class Project:
    id: str
    title: str
    summary: str
    tags: tuple[str, ...]
    models: tuple[str, ...]
    datasets: tuple[str, ...]
    likes: int
    sdk: str
    license: str
    created_at: str
    last_modified: str
    host: str
    url: str

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or data["id"].rsplit("/", 1)[-1]),
            summary=str(data.get("summary") or ""),
            tags=tuple(data.get("tags") or ()),
            models=tuple(data.get("models") or ()),
            datasets=tuple(data.get("datasets") or ()),
            likes=int(data.get("likes") or 0),
            sdk=str(data.get("sdk") or ""),
            license=str(data.get("license") or ""),
            created_at=str(data.get("created_at") or ""),
            last_modified=str(data.get("last_modified") or ""),
            host=str(data.get("host") or ""),
            url=str(data.get("url") or f"https://huggingface.co/spaces/{data['id']}"),
        )

    @property
    def slug(self) -> str:
        return self.id.rsplit("/", 1)[-1]

    @property
    def searchable_text(self) -> str:
        return " ".join(
            [
                self.title,
                self.slug.replace("-", " ").replace("_", " "),
                self.summary,
                " ".join(self.tags),
                " ".join(self.models),
                " ".join(self.datasets),
            ]
        )

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "tags": list(self.tags),
            "models": list(self.models),
            "datasets": list(self.datasets),
            "likes": self.likes,
            "sdk": self.sdk,
            "license": self.license,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "host": self.host,
            "url": self.url,
        }


@dataclass(frozen=True)
class SearchHit:
    project: Project
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class WhitespaceItem:
    label: str
    pitch: str
    evidence: str
    score: float
    nearby_projects: tuple[Project, ...]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "pitch": self.pitch,
            "evidence": self.evidence,
            "score": round(self.score, 3),
            "nearby_projects": [project.to_public_dict() for project in self.nearby_projects],
        }


@dataclass(frozen=True)
class WhitespaceSeed:
    label: str
    query: str
    pitch: str


WHITESPACE_SEEDS: tuple[WhitespaceSeed, ...] = (
    WhitespaceSeed(
        "Tiny civic repair desk",
        "local government forms benefits tenant aid accessibility paperwork",
        "A small agent that turns intimidating public-service forms into one-page action plans.",
    ),
    WhitespaceSeed(
        "Hands-on science coach",
        "kitchen science experiment kids sensor notebook classroom",
        "A lab-notebook companion that designs safe experiments from household materials.",
    ),
    WhitespaceSeed(
        "Offline field translator",
        "offline translation field guide travel emergency low connectivity",
        "A local-first phrase and intent helper for stressful travel or field-work moments.",
    ),
    WhitespaceSeed(
        "Personal archive cartographer",
        "photos notes memories archive timeline family history scrapbook",
        "A tiny model that maps a private archive into stories without sending it to cloud APIs.",
    ),
    WhitespaceSeed(
        "Small-team incident scribe",
        "incident retrospective logs on call debugging timeline root cause",
        "A local incident historian that turns messy notes into a calm timeline and next actions.",
    ),
    WhitespaceSeed(
        "Accessibility rehearsal room",
        "accessibility captions alt text screen reader rehearsal inclusive design",
        "A practice space that lets makers rehearse their demo for captions, contrast, and clarity.",
    ),
    WhitespaceSeed(
        "Neighborhood seed library",
        "garden plants seed library neighborhood seasons climate local exchange",
        "An advisor for hyperlocal seed swaps, planting plans, and community garden knowledge.",
    ),
)


class ProjectIndex:
    def __init__(self, projects: list[Project], generated_at: str, source: str) -> None:
        if not projects:
            raise ValueError("project index requires at least one project")
        self.projects = projects
        self.generated_at = generated_at
        self.source = source
        self._documents = [Counter(tokenize(project.searchable_text)) for project in projects]
        self._df = Counter(term for doc in self._documents for term in doc)
        self._idf = {
            term: math.log((1 + len(self._documents)) / (1 + freq)) + 1.0
            for term, freq in self._df.items()
        }
        self._norms = [self._norm(doc) for doc in self._documents]

    @classmethod
    def from_file(cls, path: Path) -> "ProjectIndex":
        data = json.loads(path.read_text(encoding="utf-8"))
        projects = [Project.from_dict(item) for item in data["projects"]]
        return cls(
            projects=projects,
            generated_at=str(data.get("generated_at") or ""),
            source=str(data.get("source") or ""),
        )

    def top_projects(self, limit: int = 8) -> list[Project]:
        return sorted(
            self.projects,
            key=lambda project: (project.likes, project.last_modified, project.title.lower()),
            reverse=True,
        )[:limit]

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        query_doc = Counter(query_terms)
        query_norm = self._norm(query_doc)
        hits: list[SearchHit] = []
        for project, doc, doc_norm in zip(self.projects, self._documents, self._norms, strict=True):
            if doc_norm == 0.0 or query_norm == 0.0:
                continue
            raw = 0.0
            matched: list[str] = []
            for term, count in query_doc.items():
                if term not in doc:
                    continue
                raw += count * doc[term] * self._idf.get(term, 1.0) ** 2
                matched.append(term)
            if not matched:
                continue
            title_bonus = sum(0.08 for term in matched if term in tokenize(project.title))
            tag_bonus = sum(0.05 for term in matched if term in tokenize(" ".join(project.tags)))
            score = raw / (query_norm * doc_norm) + title_bonus + tag_bonus
            hits.append(SearchHit(project=project, score=score, matched_terms=tuple(sorted(matched))))
        hits.sort(key=lambda hit: (hit.score, hit.project.likes), reverse=True)
        return hits[:limit]

    def get(self, project_id: str) -> Project | None:
        for project in self.projects:
            if project.id == project_id or project.slug == project_id:
                return project
        return None

    def find_whitespace(self, limit: int = 5) -> list[WhitespaceItem]:
        items: list[WhitespaceItem] = []
        for seed in WHITESPACE_SEEDS:
            hits = self.search(seed.query, limit=3)
            saturation = sum(hit.score for hit in hits) / max(len(hits), 1)
            score = max(0.0, 1.0 - min(saturation, 0.95))
            if hits:
                evidence = f"Nearest echoes are weak: {', '.join(hit.project.title for hit in hits[:2])}."
            else:
                evidence = "No close project echoes in the current snapshot."
            items.append(
                WhitespaceItem(
                    label=seed.label,
                    pitch=seed.pitch,
                    evidence=evidence,
                    score=score,
                    nearby_projects=tuple(hit.project for hit in hits),
                )
            )
        items.sort(key=lambda item: item.score, reverse=True)
        return items[:limit]

    def _norm(self, doc: Counter[str]) -> float:
        return math.sqrt(sum((count * self._idf.get(term, 1.0)) ** 2 for term, count in doc.items()))


def tokenize(text: str) -> list[str]:
    return [token.lower().strip("._-+") for token in TOKEN_RE.findall(text) if len(token.strip("._-+")) > 1]
