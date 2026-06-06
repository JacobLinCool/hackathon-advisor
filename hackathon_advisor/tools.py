from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid

from hackathon_advisor.data import Project, ProjectIndex, WhitespaceItem
from hackathon_advisor.scoring import ScoreCard, score_idea


TARGETS = [
    "Off the Grid",
    "Well-Tuned",
    "Off-Brand",
    "Llama Champion",
    "Sharing is Caring",
    "Field Notes",
]


def normalize_targets(raw_targets: Any, default: list[str] | None = None) -> list[str]:
    if raw_targets is None:
        return list(default or [])
    if not isinstance(raw_targets, list):
        return list(default or [])

    targets: list[str] = []
    seen: set[str] = set()
    for raw_target in raw_targets:
        target = str(raw_target)
        if target in TARGETS and target not in seen:
            targets.append(target)
            seen.add(target)
    return targets


def targets_from_state(state: dict[str, Any]) -> list[str]:
    if "targets" not in state:
        return TARGETS[:3]
    return normalize_targets(state.get("targets"), default=[])


@dataclass
class Idea:
    id: str
    title: str
    pitch: str
    targets: list[str] = field(default_factory=lambda: TARGETS[:3])
    score: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "pitch": self.pitch,
            "targets": self.targets,
            "score": self.score,
        }


@dataclass(frozen=True)
class ToolEvent:
    name: str
    summary: str

    def to_dict(self) -> dict:
        return {"name": self.name, "summary": self.summary}


class AdvisorTools:
    def __init__(self, index: ProjectIndex) -> None:
        self.index = index

    def list_projects(self, limit: int = 8) -> tuple[list[Project], ToolEvent]:
        projects = self.index.top_projects(limit=limit)
        return projects, ToolEvent("list_projects", f"Read {len(projects)} prominent Space cards.")

    def search_projects(self, query: str, limit: int = 5) -> tuple[list[Project], ToolEvent]:
        hits = self.index.search(query, limit=limit)
        projects = [hit.project for hit in hits]
        return projects, ToolEvent("search_projects", f"Found {len(projects)} nearby Space echoes.")

    def find_whitespace(self, limit: int = 5) -> tuple[list[WhitespaceItem], ToolEvent]:
        items = self.index.find_whitespace(limit=limit)
        return items, ToolEvent("find_whitespace", f"Ranked {len(items)} under-explored regions.")

    def save_idea(self, state: dict[str, Any], title: str, pitch: str) -> tuple[Idea, ToolEvent]:
        ideas = [Idea(**item) for item in state.get("ideas", [])]
        current_id = state.get("current_idea_id")
        targets = targets_from_state(state)
        idea = next((item for item in ideas if item.id == current_id), None)
        if idea is None:
            idea = Idea(id=uuid.uuid4().hex[:8], title=title, pitch=pitch, targets=targets)
            ideas.append(idea)
        else:
            idea.title = title
            idea.pitch = pitch
            idea.targets = targets
        state["ideas"] = [item.to_dict() for item in ideas]
        state["current_idea_id"] = idea.id
        return idea, ToolEvent("save_idea", f"Wrote idea page '{idea.title}'.")

    def score_idea(self, idea: Idea) -> tuple[ScoreCard, ToolEvent]:
        score = score_idea(self.index, idea.title, idea.pitch, idea.targets)
        idea.score = score.to_dict()
        return score, ToolEvent("score_idea", f"Pressed a five-quadrant seal: {score.overall}/10.")

    def make_plan(self, idea: Idea) -> tuple[list[str], ToolEvent]:
        plan = [
            "Lock a one-sentence promise and one demo input that proves originality.",
            "Refresh the Space snapshot, then tune the bleed threshold against the closest echoes.",
            "Build the smallest happy path: input, citations, score seal, and shareable artifact.",
            "Add one prize hook only after the core loop is smooth enough to demo without narration.",
            "Record the trace and write Field Notes from the exact build decisions.",
        ]
        if any("Well" in target for target in idea.targets):
            plan.insert(4, "Prepare a tiny LoRA dataset from successful advisor turns before training.")
        return plan, ToolEvent("make_plan", f"Drafted {len(plan)} build steps.")


def idea_from_text(text: str) -> tuple[str, str]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "Blank Page", "A project direction waiting for one concrete user and one concrete tension."
    title = cleaned
    for prefix in ("i want to build", "build", "make", "my idea is", "idea:"):
        if cleaned.lower().startswith(prefix):
            title = cleaned[len(prefix) :].strip(" :-")
            break
    title = title[:64].strip(" .") or "Unwritten Page"
    if len(title) < len(cleaned):
        title = f"{title[:58].strip()}..."
    return _display_title(title), cleaned


def _display_title(title: str) -> str:
    if not title:
        return "Unwritten Page"
    if any(char.isupper() or char.isdigit() for char in title):
        return title[0].upper() + title[1:]
    return title.capitalize()
