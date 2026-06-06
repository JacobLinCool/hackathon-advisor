from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any
import re

from hackathon_advisor.aliases import Correction, normalize_text
from hackathon_advisor.data import Project, ProjectIndex, WhitespaceItem
from hackathon_advisor.scoring import ScoreCard
from hackathon_advisor.tools import AdvisorTools, Idea, ToolEvent, idea_from_text


PLAN_RE = re.compile(r"\b(plan|build order|roadmap|next step|milestone)\b", re.IGNORECASE)
COMPARE_RE = re.compile(r"\b(compare|choose|rank)\b", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\b(whitespace|original|new|bolder|unwritten|gap)\b", re.IGNORECASE)
SEARCH_RE = re.compile(r"\b(search|similar|already|existing|overlap|echo)\b", re.IGNORECASE)


@dataclass
class TurnResult:
    normalized_text: str
    corrections: list[Correction]
    response: str
    state: dict[str, Any]
    tool_events: list[ToolEvent]
    projects: list[Project]
    whitespace: list[WhitespaceItem]
    score: ScoreCard | None
    plan: list[str]
    artifact: dict[str, Any]

    def stream_chunks(self) -> list[str]:
        words = self.response.split(" ")
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            current.append(word)
            if len(" ".join(current)) >= 28:
                chunks.append(" ".join(current) + " ")
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks


class AdvisorEngine:
    def __init__(self, index: ProjectIndex) -> None:
        self.index = index
        self.tools = AdvisorTools(index)

    def turn(self, message: str, state: dict[str, Any] | None = None) -> TurnResult:
        state = dict(state or {})
        state.setdefault("ideas", [])
        normalized, corrections = normalize_text(message)
        tool_events: list[ToolEvent] = []
        projects: list[Project] = []
        whitespace: list[WhitespaceItem] = []
        score: ScoreCard | None = None
        plan: list[str] = []

        if not normalized.strip():
            projects, event = self.tools.list_projects(limit=6)
            tool_events.append(event)
            response = self._opening_response(projects)
            return self._result(normalized, corrections, response, state, tool_events, projects, [], None, [], {})

        if COMPARE_RE.search(normalized) and state.get("ideas"):
            response = self._compare_response(state)
            tool_events.append(ToolEvent("compare_ideas", "Compared the current idea board."))
            return self._result(normalized, corrections, response, state, tool_events, [], [], None, [], {})

        if PLAN_RE.search(normalized) and state.get("ideas"):
            idea = self._current_idea(state)
            if idea is not None:
                score, event = self.tools.score_idea(idea)
                self._store_idea(state, idea)
                tool_events.append(event)
                plan, event = self.tools.make_plan(idea)
                tool_events.append(event)
                response = self._plan_response(idea, score, plan)
                artifact = self._artifact(idea, score)
                return self._result(
                    normalized,
                    corrections,
                    response,
                    state,
                    tool_events,
                    [],
                    [],
                    score,
                    plan,
                    artifact,
                )

        title, pitch = idea_from_text(normalized)
        idea, event = self.tools.save_idea(state, title, pitch)
        tool_events.append(event)

        if PLAN_RE.search(normalized):
            score, event = self.tools.score_idea(idea)
            self._store_idea(state, idea)
            tool_events.append(event)
            plan, event = self.tools.make_plan(idea)
            tool_events.append(event)
            response = self._plan_response(idea, score, plan)
            artifact = self._artifact(idea, score)
            return self._result(normalized, corrections, response, state, tool_events, [], [], score, plan, artifact)

        if WHITESPACE_RE.search(normalized):
            whitespace, event = self.tools.find_whitespace(limit=4)
            tool_events.append(event)
            if whitespace:
                idea.title = whitespace[0].label
                idea.pitch = whitespace[0].pitch
                state["ideas"] = [
                    idea.to_dict() if item.get("id") == idea.id else item for item in state.get("ideas", [])
                ]
            score, event = self.tools.score_idea(idea)
            if whitespace:
                score = self._align_score_with_whitespace(score, whitespace[0])
                idea.score = score.to_dict()
            self._store_idea(state, idea)
            tool_events.append(event)
            response = self._whitespace_response(idea, whitespace, score)
            artifact = self._artifact(idea, score)
            return self._result(
                normalized,
                corrections,
                response,
                state,
                tool_events,
                [],
                whitespace,
                score,
                [],
                artifact,
            )

        hits = self.index.search(normalized, limit=5)
        projects = [hit.project for hit in hits]
        tool_events.append(ToolEvent("search_projects", f"Checked {len(projects)} closest project echoes."))
        score, event = self.tools.score_idea(idea)
        self._store_idea(state, idea)
        tool_events.append(event)

        if SEARCH_RE.search(normalized) or projects:
            response = self._overlap_response(idea, projects, score)
        else:
            whitespace, event = self.tools.find_whitespace(limit=3)
            tool_events.append(event)
            response = self._whitespace_response(idea, whitespace, score)

        artifact = self._artifact(idea, score)
        return self._result(
            normalized,
            corrections,
            response,
            state,
            tool_events,
            projects,
            whitespace,
            score,
            plan,
            artifact,
        )

    def _result(
        self,
        normalized_text: str,
        corrections: list[Correction],
        response: str,
        state: dict[str, Any],
        tool_events: list[ToolEvent],
        projects: list[Project],
        whitespace: list[WhitespaceItem],
        score: ScoreCard | None,
        plan: list[str],
        artifact: dict[str, Any],
    ) -> TurnResult:
        self._record_trace(state, normalized_text, response, tool_events, score, plan, artifact)
        return TurnResult(
            normalized_text=normalized_text,
            corrections=corrections,
            response=response,
            state=state,
            tool_events=tool_events,
            projects=projects,
            whitespace=whitespace,
            score=score,
            plan=plan,
            artifact=artifact,
        )

    def _store_idea(self, state: dict[str, Any], idea: Idea) -> None:
        state["ideas"] = [
            idea.to_dict() if item.get("id") == idea.id else item for item in state.get("ideas", [])
        ]

    def _current_idea(self, state: dict[str, Any]) -> Idea | None:
        current_id = state.get("current_idea_id")
        for item in state.get("ideas", []):
            if item.get("id") == current_id:
                return Idea(**item)
        if state.get("ideas"):
            return Idea(**state["ideas"][-1])
        return None

    def _record_trace(
        self,
        state: dict[str, Any],
        normalized_text: str,
        response: str,
        tool_events: list[ToolEvent],
        score: ScoreCard | None,
        plan: list[str],
        artifact: dict[str, Any],
    ) -> None:
        trace = list(state.get("trace", []))
        trace.append(
            {
                "input": normalized_text[:240],
                "tools": [event.to_dict() for event in tool_events],
                "verdict": score.verdict if score else "",
                "overall": score.overall if score else None,
                "plan_steps": len(plan),
                "artifact_title": artifact.get("title", ""),
                "response": response[:360],
            }
        )
        state["trace"] = trace[-12:]
        if artifact:
            state["last_artifact"] = artifact

    def _align_score_with_whitespace(self, score: ScoreCard, item: WhitespaceItem) -> ScoreCard:
        if item.score < 0.70:
            return score
        return replace(
            score,
            originality=max(score.originality, 8),
            verdict="UNWRITTEN",
        )

    def _opening_response(self, projects: list[Project]) -> str:
        names = ", ".join(project.title for project in projects[:4])
        return (
            "Mothback opens the Almanac. The Wood is already inked with "
            f"{len(self.index.projects)} project pages; the brightest current echoes include {names}. "
            "Give me one project instinct and I will test whether it bleeds red or blooms gold."
        )

    def _overlap_response(self, idea: Idea, projects: list[Project], score: ScoreCard) -> str:
        if score.verdict.startswith("UNWRITTEN"):
            nearby = ", ".join(project.title for project in projects[:2]) or "no close pages"
            return (
                f"The page for {idea.title} does not bleed much. I found {nearby}, but the seal reads "
                f"{score.verdict} at {score.overall}/10. Push the AI necessity harder: make the model decide, rank, "
                "or personalize something a static app cannot."
            )
        citations = "; ".join(f"page {idx + 1}: {project.title}" for idx, project in enumerate(projects[:3]))
        return (
            f"The ink bleeds around {idea.title}. Closest echoes: {citations}. The seal reads "
            f"{score.verdict} at {score.overall}/10. Keep the audience, but change the mechanism or artifact so the "
            "demo proves a gap instead of joining a cluster."
        )

    def _whitespace_response(
        self,
        idea: Idea,
        whitespace: list[WhitespaceItem],
        score: ScoreCard,
    ) -> str:
        if not whitespace:
            return (
                f"The page for {idea.title} stays pale: I could not find a strong whitespace candidate in the "
                "snapshot. Narrow the user and the moment, then ask again."
            )
        lead = whitespace[0]
        return (
            f"Gold gathers on {lead.label}. {lead.pitch} {lead.evidence} The seal reads "
            f"{score.verdict} at {score.overall}/10. The next move is to make one concrete before/after scene and "
            "cite the two weakest nearby echoes in the margin."
        )

    def _plan_response(self, idea: Idea, score: ScoreCard, plan: list[str]) -> str:
        steps = " ".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan))
        return (
            f"Mothback presses the wax for {idea.title}: {score.overall}/10, {score.verdict}. "
            f"The build path is: {steps}"
        )

    def _compare_response(self, state: dict[str, Any]) -> str:
        ideas = state.get("ideas", [])
        if not ideas:
            return "There are no written pages on the board yet."
        scored = sorted(
            ideas,
            key=lambda item: ((item.get("score") or {}).get("overall") or 0, item.get("title") or ""),
            reverse=True,
        )
        names = ", ".join(item.get("title", "Untitled") for item in scored[:3])
        return f"The board tilts toward {names}. Keep the top page only if its artifact can be understood in ten seconds."

    def _artifact(self, idea: Idea, score: ScoreCard) -> dict[str, Any]:
        return {
            "title": idea.title,
            "verdict": score.verdict,
            "overall": score.overall,
            "caption": f"Mothback inked my Build Small fate page: {idea.title} - {score.verdict}.",
            "seal": score.to_dict(),
        }
