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

TARGET_PROFILE_BY_ID = {
    "Off the Grid": {
        "label": "Local-first",
        "description": "Favor ideas that work without proprietary inference APIs.",
    },
    "Well-Tuned": {
        "label": "Trainable",
        "description": "Shape good turns into a tiny fine-tune dataset.",
    },
    "Off-Brand": {
        "label": "Distinct voice",
        "description": "Leave room for an interface and tone people remember.",
    },
    "Llama Champion": {
        "label": "llama.cpp path",
        "description": "Prefer small-model choices that can run locally.",
    },
    "Sharing is Caring": {
        "label": "Shareable artifact",
        "description": "Make an output people can save, post, or compare.",
    },
    "Field Notes": {
        "label": "Build notes",
        "description": "Keep decisions easy to write up from the session trace.",
    },
}


def target_profiles() -> list[dict[str, str]]:
    return [
        {
            "id": target,
            "label": TARGET_PROFILE_BY_ID[target]["label"],
            "description": TARGET_PROFILE_BY_ID[target]["description"],
        }
        for target in TARGETS
    ]


def target_label(target: str) -> str:
    return TARGET_PROFILE_BY_ID.get(target, {}).get("label", target)


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
        if idea is None or _is_new_idea(idea, title, pitch):
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

    def make_plan(self, idea: Idea, profile: dict[str, Any] | None = None) -> tuple[list[str], ToolEvent]:
        plan = [
            "Lock a one-sentence promise and one demo input that proves originality.",
            "Compare against the nearest echoes, then sharpen the part only this idea can own.",
            "Build the smallest happy path: input, citations, score seal, and shareable artifact.",
            "Add one goal hook only after the core loop is smooth enough to demo without narration.",
            "Record the trace and write build notes from the exact decisions.",
        ]
        profile_steps = profile_plan_steps(profile)
        if profile_steps:
            plan[1:1] = profile_steps
        if any("Well" in target for target in idea.targets):
            plan.insert(
                max(0, len(plan) - 1),
                "Prepare a tiny LoRA dataset from successful advisor turns before training.",
            )
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
    pitch = cleaned
    explicit_pitch = False
    if " -- " in title:
        title, pitch = (part.strip() for part in title.split(" -- ", 1))
        explicit_pitch = True
    raw_title = title
    title = raw_title[:64].strip(" .") or "Unwritten Page"
    if len(raw_title) > 64 or (not explicit_pitch and len(title) < len(cleaned)):
        title = f"{title[:58].strip()}..."
    return _display_title(title), pitch


def _is_new_idea(current: Idea, title: str, pitch: str) -> bool:
    return current.title.strip().casefold() != title.strip().casefold() or current.pitch.strip() != pitch.strip()


def profile_plan_steps(profile: dict[str, Any] | None) -> list[str]:
    if not isinstance(profile, dict):
        return []
    steps: list[str] = []
    time = _short_profile_value(profile.get("time"))
    skills = _short_profile_value(profile.get("skills"))
    constraints = _short_profile_value(profile.get("constraints"))
    preferences = _short_profile_value(profile.get("preferences"))
    if time:
        steps.append(f"Scope the first prototype to {time}; cut anything that cannot fit that window.")
    if skills:
        steps.append(f"Use your {skills} strength for the first working surface before adding new tooling.")
    if constraints:
        steps.append(f"Test the constraint early: {constraints}. Do this before polishing the artifact.")
    if preferences:
        steps.append(f"Shape the demo around {preferences} so the result feels intentional, not generic.")
    return steps


def _short_profile_value(value: Any, limit: int = 84) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip(" ,.;:") + "..."


def _display_title(title: str) -> str:
    if not title:
        return "Unwritten Page"
    if any(char.isupper() or char.isdigit() for char in title):
        return title[0].upper() + title[1:]
    return title.capitalize()
