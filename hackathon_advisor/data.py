from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.+_-]*", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
GENERIC_PUBLIC_TITLE_RE = re.compile(
    r"^(?:my\s+)?build\s+small\s+hackathon$",
    re.IGNORECASE,
)
GENERIC_PUBLIC_SUMMARY_RE = re.compile(
    r"(?:\bthis\s+(?:is\s+)?(?:space\s+is\s+for|my\s+submission)\b.*\b(?:build[-\s]*small|hackathon)\b)"
    r"|(?:\bhacka?ton\s+project\b)"
    r"|(?:^\s*todo\s*$)",
    re.IGNORECASE,
)

INDEX_SCHEMA_VERSION = 3
INDEX_ALGORITHM = "llama-cpp-embedding-v1"
DEFAULT_EMBEDDING_MODEL_REPO = "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
DEFAULT_EMBEDDING_MODEL_FILE = "embeddinggemma-300m-qat-Q8_0.gguf"
DEFAULT_EMBEDDING_RUNTIME = "llama.cpp via llama-cpp-python"
APP_FILE_EMBEDDING_CHAR_LIMIT = 8000
README_EMBEDDING_CHAR_LIMIT = 4000


EmbeddingFunction = Callable[[str], Sequence[float]]


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
    app_file: str = ""
    app_file_embedding_text: str = ""
    readme_body: str = ""
    app_file_source: str = ""

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
            app_file=str(data.get("app_file") or ""),
            app_file_embedding_text=str(data.get("app_file_embedding_text") or ""),
            readme_body=str(data.get("readme_body") or ""),
            app_file_source=str(data.get("app_file_source") or data.get("app_source") or ""),
        )

    @property
    def slug(self) -> str:
        return self.id.rsplit("/", 1)[-1]

    @property
    def searchable_text(self) -> str:
        return "\n".join(
            part
            for part in [
                f"title: {self.title}",
                f"slug: {self.slug.replace('-', ' ').replace('_', ' ')}",
                f"summary: {self.summary}",
                f"readme:\n{bounded_embedding_text(self.readme_body, README_EMBEDDING_CHAR_LIMIT)}"
                if self.readme_body
                else "",
                f"tags: {' '.join(self.tags)}",
                f"models: {' '.join(self.models)}",
                f"datasets: {' '.join(self.datasets)}",
                f"main app file: {self.app_file}" if self.app_file else "",
                f"main app file content:\n{self.app_file_embedding_text}"
                if self.app_file_embedding_text
                else "",
            ]
            if part.strip()
        )

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "title": public_project_title(self.title),
            "summary": public_project_summary(self.summary),
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
            "app_file": self.app_file,
        }

    def to_snapshot_dict(self) -> dict:
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
            "app_file": self.app_file,
            "app_file_embedding_text": self.app_file_embedding_text,
        }

    def to_refresh_snapshot_dict(self) -> dict:
        payload = self.to_snapshot_dict()
        payload.update(
            {
                "readme_body": self.readme_body,
                "app_file_source": self.app_file_source,
            }
        )
        return payload

@dataclass(frozen=True)
class SearchHit:
    project: Project
    score: float
    matched_terms: tuple[str, ...]
    page_number: int


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


def public_project_title(title: str) -> str:
    cleaned = " ".join(str(title).split())
    if not cleaned:
        return "Untitled project"
    if GENERIC_PUBLIC_TITLE_RE.search(cleaned):
        return "Untitled project"
    return cleaned


def public_project_summary(summary: str) -> str:
    cleaned = " ".join(str(summary).split())
    if not cleaned:
        return ""
    if GENERIC_PUBLIC_SUMMARY_RE.search(cleaned):
        return ""
    return cleaned


def extract_app_file_embedding_text(app_file: str, text: str) -> str:
    cleaned_file = str(app_file).strip()
    cleaned_text = str(text or "")
    if not cleaned_file or not cleaned_text.strip():
        return ""

    suffix = PurePosixPath(cleaned_file).suffix.lower()
    if suffix == ".py":
        body = python_app_signals(cleaned_text)
    else:
        body = cleaned_text
    return bounded_embedding_text(body, APP_FILE_EMBEDDING_CHAR_LIMIT)


def python_app_signals(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    signals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signals.append(node.name)
            for arg in node.args.args:
                signals.append(arg.arg)
        elif isinstance(node, ast.ClassDef):
            signals.append(node.name)
        elif isinstance(node, ast.Call):
            name = call_name(node.func)
            if name:
                signals.append(name)
            signals.extend(keyword.arg for keyword in node.keywords if keyword.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            signals.append(node.value)

    return ordered_normalized_text(signals)


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def ordered_normalized_text(values: Sequence[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = clean_embedding_signal(value)
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return "\n".join(ordered)


def clean_embedding_signal(value: str) -> str:
    cleaned = HTML_TAG_RE.sub(" ", str(value))
    cleaned = " ".join(cleaned.split())
    if looks_like_style_blob(cleaned):
        return ""
    return cleaned


def looks_like_style_blob(text: str) -> bool:
    if len(text) < 80:
        return False
    style_markers = (
        text.count("{")
        + text.count("}")
        + text.count(";")
        + text.count("!important")
        + text.count("rgba(")
        + text.count("linear-gradient")
    )
    return style_markers >= 8 and style_markers / len(text) > 0.015


def bounded_embedding_text(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    marker = " ... "
    edge = max(1, (limit - len(marker)) // 2)
    return f"{cleaned[:edge].rstrip()}{marker}{cleaned[-edge:].lstrip()}"


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
    def __init__(
        self,
        projects: list[Project],
        generated_at: str,
        source: str,
        index_payload: dict,
        query_embedder: EmbeddingFunction | None = None,
    ) -> None:
        if not projects:
            raise ValueError("project index requires at least one project")
        validate_index_payload(index_payload, projects, generated_at, source)
        self.projects = projects
        self.generated_at = generated_at
        self.source = source
        self.index_generated_at = str(index_payload["generated_at"])
        self.index_algorithm = str(index_payload["algorithm"])
        self.snapshot_digest = str(index_payload["snapshot_digest"])
        self.embedding_metadata = dict(index_payload["embedding"])
        self.embedding_dimensions = int(self.embedding_metadata["dimensions"])
        self._query_embedder = query_embedder
        self._vectors = [
            tuple(float(value) for value in document["vector"])
            for document in index_payload["documents"]
        ]
        self._vector_by_id = {
            project.id: vector for project, vector in zip(self.projects, self._vectors)
        }

    def vector_for(self, project_id: str) -> tuple[float, ...] | None:
        return self._vector_by_id.get(project_id)

    def project_vectors(self) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vectors)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return tuple(normalize_vector(self._embed_query(text)))

    @classmethod
    def from_file(cls, path: Path, query_embedder: EmbeddingFunction | None = None) -> "ProjectIndex":
        data = json.loads(path.read_text(encoding="utf-8"))
        projects = [Project.from_dict(item) for item in data["projects"]]
        raise ValueError("ProjectIndex.from_file requires a separate embedding index payload")

    @classmethod
    def from_files(
        cls,
        project_path: Path,
        index_path: Path,
        query_embedder: EmbeddingFunction | None = None,
    ) -> "ProjectIndex":
        data = json.loads(project_path.read_text(encoding="utf-8"))
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        projects = [Project.from_dict(item) for item in data["projects"]]
        return cls(
            projects=projects,
            generated_at=str(data.get("generated_at") or ""),
            source=str(data.get("source") or ""),
            index_payload=index_payload,
            query_embedder=query_embedder,
        )

    def set_query_embedder(self, embedder: EmbeddingFunction) -> None:
        self._query_embedder = embedder

    def top_projects(self, limit: int = 8) -> list[Project]:
        return sorted(
            self.projects,
            key=lambda project: (project.likes, project.last_modified, project.title.lower()),
            reverse=True,
        )[:limit]

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        query_vector = normalize_vector(self._embed_query(query))
        hits: list[SearchHit] = []
        for page_number, (project, vector) in enumerate(
            zip(self.projects, self._vectors, strict=True),
            start=1,
        ):
            score = max(0.0, min(1.0, (dot_product(query_vector, vector) + 1.0) / 2.0))
            hits.append(
                SearchHit(
                    project=project,
                    score=score,
                    matched_terms=matched_terms(query_terms, project),
                    page_number=page_number,
                )
            )
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
            score = max(0.0, min(1.0, 1.0 - max(0.0, saturation - 0.35) / 0.60))
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

    def starter_directions(self, limit: int = 5) -> list[WhitespaceItem]:
        return [
            WhitespaceItem(
                label=seed.label,
                pitch=seed.pitch,
                evidence="Press this direction to test it against the current project map.",
                score=0.0,
                nearby_projects=(),
            )
            for seed in WHITESPACE_SEEDS[:limit]
        ]

    def _embed_query(self, query: str) -> Sequence[float]:
        if self._query_embedder is None:
            from hackathon_advisor.llama_embedding import create_llama_cpp_embedder

            self._query_embedder = create_llama_cpp_embedder(self.embedding_metadata)
        return self._query_embedder(query)


def tokenize(text: str) -> list[str]:
    return [token.lower().strip("._-+") for token in TOKEN_RE.findall(text) if len(token.strip("._-+")) > 1]


def matched_terms(query_terms: set[str], project: Project) -> tuple[str, ...]:
    project_terms = set(tokenize(project.searchable_text))
    return tuple(sorted(query_terms & project_terms)[:8])


def build_index_payload(
    projects: list[Project],
    snapshot_generated_at: str,
    source: str,
    embeddings: Sequence[Sequence[float]],
    *,
    embedding_metadata: dict[str, Any] | None = None,
) -> dict:
    if len(embeddings) != len(projects):
        raise ValueError("embedding count must match project count")
    normalized = [normalize_vector(vector) for vector in embeddings]
    dimensions = len(normalized[0]) if normalized else 0
    if dimensions <= 0:
        raise ValueError("embedding vectors must not be empty")
    if any(len(vector) != dimensions for vector in normalized):
        raise ValueError("embedding vectors must have one shared dimension")

    metadata = {
        "model_repo": DEFAULT_EMBEDDING_MODEL_REPO,
        "model_file": DEFAULT_EMBEDDING_MODEL_FILE,
        "runtime": DEFAULT_EMBEDDING_RUNTIME,
        "dimensions": dimensions,
        "normalized": True,
        **(embedding_metadata or {}),
    }
    indexed_documents = []
    for project, vector in zip(projects, normalized, strict=True):
        indexed_documents.append(
            {
                "project_id": project.id,
                "text_digest": sha256(project.searchable_text.encode("utf-8")).hexdigest(),
                "norm": round(vector_norm(vector), 8),
                "vector": [round(value, 8) for value in vector],
            }
        )
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "algorithm": INDEX_ALGORITHM,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_generated_at": snapshot_generated_at,
        "snapshot_source": source,
        "snapshot_digest": project_snapshot_digest(projects, snapshot_generated_at, source),
        "document_count": len(projects),
        "embedding": metadata,
        "documents": indexed_documents,
    }


def validate_index_payload(
    payload: dict,
    projects: list[Project],
    snapshot_generated_at: str,
    snapshot_source: str,
) -> None:
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported project index schema version")
    if payload.get("algorithm") != INDEX_ALGORITHM:
        raise ValueError(f"unsupported project index algorithm: {payload.get('algorithm')}")
    if payload.get("snapshot_generated_at") != snapshot_generated_at:
        raise ValueError("project index was built from a different snapshot timestamp")
    if payload.get("snapshot_source") != snapshot_source:
        raise ValueError("project index was built from a different snapshot source")
    if payload.get("snapshot_digest") != project_snapshot_digest(
        projects,
        snapshot_generated_at,
        snapshot_source,
    ):
        raise ValueError("project index digest does not match projects snapshot")

    embedding = payload.get("embedding")
    if not isinstance(embedding, dict):
        raise ValueError("project index embedding metadata is missing")
    dimensions = int(embedding.get("dimensions") or 0)
    if dimensions <= 0:
        raise ValueError("project index embedding dimensions must be positive")
    if embedding.get("runtime") != DEFAULT_EMBEDDING_RUNTIME:
        raise ValueError("project index embedding runtime must be llama.cpp")

    documents = payload.get("documents")
    if not isinstance(documents, list) or len(documents) != len(projects):
        raise ValueError("project index document count does not match projects snapshot")
    project_ids = [project.id for project in projects]
    indexed_ids = [document.get("project_id") for document in documents]
    if indexed_ids != project_ids:
        raise ValueError("project index project order does not match projects snapshot")
    for project, document in zip(projects, documents, strict=True):
        if document.get("text_digest") != sha256(project.searchable_text.encode("utf-8")).hexdigest():
            raise ValueError("project index text digest does not match searchable project text")
        vector = document.get("vector")
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError("project index vector dimensions do not match embedding metadata")
        norm = vector_norm(float(value) for value in vector)
        if not 0.99 <= norm <= 1.01:
            raise ValueError("project index vectors must be normalized")


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    norm = vector_norm(values)
    if norm == 0.0:
        raise ValueError("embedding vector norm must be non-zero")
    return tuple(value / norm for value in values)


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have equal dimensions")
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def project_snapshot_digest(projects: list[Project], generated_at: str, source: str) -> str:
    payload = {
        "generated_at": generated_at,
        "source": source,
        "projects": [project.to_snapshot_dict() for project in projects],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()
