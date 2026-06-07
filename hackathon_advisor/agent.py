from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from hackathon_advisor.aliases import Correction, normalize_text
from hackathon_advisor.data import Project, ProjectIndex, WhitespaceItem
from hackathon_advisor.model_runtime import ToolPlanner, create_tool_planner, runtime_status
from hackathon_advisor.scoring import ScoreCard
from hackathon_advisor.tool_contracts import ToolCall
from hackathon_advisor.tools import (
    GOALS,
    AdvisorTools,
    Idea,
    ToolEvent,
    goal_label,
    goals_from_state,
    idea_from_text,
    normalize_goals,
)
from hackathon_advisor.wood_map import build_wood_map


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
    def __init__(self, index: ProjectIndex, planner: ToolPlanner | None = None) -> None:
        self.index = index
        self.tools = AdvisorTools(index)
        self.planner = planner or create_tool_planner()

    def runtime_status(self) -> dict[str, Any]:
        return runtime_status(self.planner).to_dict()

    def turn(self, message: str, state: dict[str, Any] | None = None) -> TurnResult:
        state = dict(state or {})
        state.setdefault("ideas", [])
        state.setdefault("profile", {})
        state.setdefault("goals", GOALS[:3])
        normalized, corrections = normalize_text(message)
        resolution = self.planner.plan(normalized, state)
        state["last_tool_resolution"] = resolution.to_dict()
        tool_events: list[ToolEvent] = []
        projects: list[Project] = []
        whitespace: list[WhitespaceItem] = []
        score: ScoreCard | None = None
        plan: list[str] = []
        call = resolution.call

        if call.name == "list_projects":
            projects, event = self.tools.list_projects(limit=6)
            tool_events.append(event)
            response = self._opening_response(projects)
            return self._result(normalized, corrections, response, state, tool_events, projects, [], None, [], {})

        if call.name == "compare_ideas":
            return self._compare_turn(normalized, corrections, state, tool_events)

        if call.name == "make_plan":
            return self._plan_turn(call, normalized, corrections, state, tool_events)

        if call.name == "find_whitespace":
            whitespace, event = self.tools.find_whitespace(limit=4)
            tool_events.append(event)
            if whitespace:
                whitespace = self._prioritize_unused_whitespace(whitespace, state)
                selected = whitespace[0]
                idea, event = self.tools.save_idea(state, selected.label, selected.pitch)
                tool_events.append(event)
                state["current_whitespace"] = selected.to_dict()
            else:
                title, pitch = idea_from_text(normalized)
                idea, event = self.tools.save_idea(state, title, pitch)
                tool_events.append(event)
            score, event = self.tools.score_idea(idea)
            if whitespace:
                score = self._align_score_with_whitespace(score, whitespace[0])
                idea.score = score.to_dict()
            self._store_idea(state, idea)
            tool_events.append(event)
            response = self._whitespace_response(idea, whitespace, score)
            artifact = self._artifact(idea, score)
            self._attach_artifact(state, idea, artifact)
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

        if call.name == "get_project":
            return self._project_turn(call, normalized, corrections, state, tool_events)

        if call.name == "score_idea":
            return self._score_turn(call, normalized, corrections, state, tool_events)

        if call.name == "update_profile":
            return self._profile_turn(call, normalized, corrections, state, tool_events)

        if call.name == "set_goals":
            return self._goal_turn(call, normalized, corrections, state, tool_events)

        return self._idea_research_turn(call, normalized, corrections, state, tool_events)

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
        stored = []
        replaced = False
        for item in state.get("ideas", []):
            if item.get("id") == idea.id:
                stored.append(idea.to_dict())
                replaced = True
            else:
                stored.append(item)
        if not replaced:
            stored.append(idea.to_dict())
        state["ideas"] = stored

    def _attach_artifact(self, state: dict[str, Any], idea: Idea, artifact: dict[str, Any]) -> None:
        idea.artifact = artifact
        self._store_idea(state, idea)
        state["last_artifact"] = artifact

    def _current_idea(self, state: dict[str, Any]) -> Idea | None:
        current_id = state.get("current_idea_id")
        for item in state.get("ideas", []):
            if item.get("id") == current_id:
                return self._with_session_goals(Idea(**item), state)
        if state.get("ideas"):
            return self._with_session_goals(Idea(**state["ideas"][-1]), state)
        return None

    def _with_session_goals(self, idea: Idea, state: dict[str, Any]) -> Idea:
        idea.goals = goals_from_state(state)
        return idea

    def _profile_context(self, state: dict[str, Any]) -> dict[str, Any]:
        profile = state.get("profile")
        return profile if isinstance(profile, dict) else {}

    def _idea_research_turn(
        self,
        call: ToolCall,
        normalized: str,
        corrections: list[Correction],
        state: dict[str, Any],
        tool_events: list[ToolEvent],
    ) -> TurnResult:
        if call.name == "search_projects":
            pitch = str(call.arguments.get("query") or normalized)
            title, _ = idea_from_text(pitch)
        else:
            title, pitch = idea_from_text(normalized)
            title = str(call.arguments.get("title") or title)
            pitch = str(call.arguments.get("pitch") or pitch)

        idea, event = self.tools.save_idea(state, title, pitch)
        tool_events.append(event)
        hits = self.index.search(pitch, limit=5)
        projects = [hit.project for hit in hits]
        tool_events.append(ToolEvent("search_projects", f"Checked {len(projects)} closest project echoes."))
        score, event = self.tools.score_idea(idea)
        self._store_idea(state, idea)
        tool_events.append(event)
        if projects:
            response = self._overlap_response(idea, projects, score)
            whitespace: list[WhitespaceItem] = []
        else:
            whitespace, event = self.tools.find_whitespace(limit=3)
            tool_events.append(event)
            response = self._whitespace_response(idea, whitespace, score)
        artifact = self._artifact(idea, score)
        self._attach_artifact(state, idea, artifact)
        return self._result(
            normalized,
            corrections,
            response,
            state,
            tool_events,
            projects,
            whitespace,
            score,
            [],
            artifact,
        )

    def _plan_turn(
        self,
        call: ToolCall,
        normalized: str,
        corrections: list[Correction],
        state: dict[str, Any],
        tool_events: list[ToolEvent],
    ) -> TurnResult:
        idea = self._idea_for_optional_id(call, state)
        if idea is None:
            tool_events.append(ToolEvent("make_plan", "No idea page was available to plan."))
            response = (
                "Write one project instinct first, or press Gap for a starting direction. "
                "Then I can draft a build path."
            )
            return self._result(normalized, corrections, response, state, tool_events, [], [], None, [], {})
        score, event = self.tools.score_idea(idea)
        score = self._align_score_from_state(score, idea, state)
        idea.score = score.to_dict()
        self._store_idea(state, idea)
        tool_events.append(event)
        plan, event = self.tools.make_plan(idea, self._profile_context(state))
        tool_events.append(event)
        response = self._plan_response(idea, score, plan)
        artifact = self._artifact(idea, score)
        self._attach_artifact(state, idea, artifact)
        return self._result(normalized, corrections, response, state, tool_events, [], [], score, plan, artifact)

    def _compare_turn(
        self,
        normalized: str,
        corrections: list[Correction],
        state: dict[str, Any],
        tool_events: list[ToolEvent],
    ) -> TurnResult:
        ranked = self._rank_ideas(state)
        if not ranked:
            tool_events.append(ToolEvent("compare_ideas", "No idea pages were available to rank."))
            response = (
                "No idea pages are on the board yet. Write one project instinct first, "
                "or press Gap to seed a direction."
            )
            return self._result(normalized, corrections, response, state, tool_events, [], [], None, [], {})

        for idea, idea_score in ranked:
            idea.artifact = self._artifact(idea, idea_score)
        ideas = [idea for idea, _score in ranked]
        state["ideas"] = [idea.to_dict() for idea in ideas]
        winner, score = ranked[0]
        state["current_idea_id"] = winner.id
        tool_events.append(ToolEvent("compare_ideas", f"Ranked {len(ranked)} idea pages by current seal score."))
        plan, event = self.tools.make_plan(winner, self._profile_context(state))
        tool_events.append(event)
        response = self._compare_response(ranked, plan)
        artifact = winner.artifact or self._artifact(winner, score)
        self._attach_artifact(state, winner, artifact)
        return self._result(normalized, corrections, response, state, tool_events, [], [], score, plan, artifact)

    def _project_turn(
        self,
        call: ToolCall,
        normalized: str,
        corrections: list[Correction],
        state: dict[str, Any],
        tool_events: list[ToolEvent],
    ) -> TurnResult:
        project = self.index.get(str(call.arguments.get("id") or ""))
        if project is None:
            response = "The requested page is not inked in the current snapshot."
            tool_events.append(ToolEvent("get_project", "No matching project card was found."))
            return self._result(normalized, corrections, response, state, tool_events, [], [], None, [], {})
        tool_events.append(ToolEvent("get_project", f"Read project card '{project.title}'."))
        response = (
            f"Page found: {project.title}. {project.summary or project.id} "
            f"Models: {', '.join(project.models) or 'not listed'}. This is a real Space citation, not a guess."
        )
        return self._result(normalized, corrections, response, state, tool_events, [project], [], None, [], {})

    def _score_turn(
        self,
        call: ToolCall,
        normalized: str,
        corrections: list[Correction],
        state: dict[str, Any],
        tool_events: list[ToolEvent],
    ) -> TurnResult:
        idea = self._idea_for_optional_id(call, state)
        if idea is None:
            title, pitch = idea_from_text(normalized)
            idea, event = self.tools.save_idea(state, title, pitch)
            tool_events.append(event)
        score, event = self.tools.score_idea(idea)
        score = self._align_score_from_state(score, idea, state)
        idea.score = score.to_dict()
        self._store_idea(state, idea)
        tool_events.append(event)
        response = f"The wax seal reads {score.overall}/10, {score.verdict}, for {idea.title}."
        artifact = self._artifact(idea, score)
        self._attach_artifact(state, idea, artifact)
        return self._result(normalized, corrections, response, state, tool_events, [], [], score, [], artifact)

    def _profile_turn(
        self,
        call: ToolCall,
        normalized: str,
        corrections: list[Correction],
        state: dict[str, Any],
        tool_events: list[ToolEvent],
    ) -> TurnResult:
        profile = dict(state.get("profile") or {})
        field = str(call.arguments["field"])
        profile[field] = str(call.arguments["value"])
        state["profile"] = profile
        state.pop("last_plan", None)
        tool_events.append(ToolEvent("update_profile", f"Remembered {field}."))
        response = f"Profile updated: {field} = {profile[field]}."
        return self._result(normalized, corrections, response, state, tool_events, [], [], None, [], {})

    def _goal_turn(
        self,
        call: ToolCall,
        normalized: str,
        corrections: list[Correction],
        state: dict[str, Any],
        tool_events: list[ToolEvent],
    ) -> TurnResult:
        goals = normalize_goals(call.arguments.get("goals"), default=[])
        state["goals"] = goals
        state.pop("last_plan", None)
        idea = self._current_idea(state)
        if idea is not None:
            idea.goals = goals
            idea.score = None
            idea.artifact = None
            self._store_idea(state, idea)
            last_artifact = state.get("last_artifact")
            if isinstance(last_artifact, dict) and last_artifact.get("title") == idea.title:
                del state["last_artifact"]
        tool_events.append(ToolEvent("set_goals", f"Set {len(goals)} goals."))
        labels = [goal_label(goal) for goal in goals]
        response = "The seal will now bias toward: " + (", ".join(labels) or "no specific goals")
        return self._result(normalized, corrections, response, state, tool_events, [], [], None, [], {})

    def _idea_for_optional_id(self, call: ToolCall, state: dict[str, Any]) -> Idea | None:
        idea_id = str(call.arguments.get("id") or "")
        if idea_id:
            for item in state.get("ideas", []):
                if item.get("id") == idea_id:
                    return self._with_session_goals(Idea(**item), state)
        return self._current_idea(state)

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
                "response": response,
                "tool_resolution": state.get("last_tool_resolution") or {},
            }
        )
        state["trace"] = trace[-12:]
        if plan:
            state["last_plan"] = list(plan)
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

    def _align_score_from_state(self, score: ScoreCard, idea: Idea, state: dict[str, Any]) -> ScoreCard:
        artifact = state.get("last_artifact") or {}
        if artifact.get("title") == idea.title and artifact.get("verdict") == "UNWRITTEN":
            return replace(score, originality=max(score.originality, 8), verdict="UNWRITTEN")
        return score

    def _prioritize_unused_whitespace(
        self,
        items: list[WhitespaceItem],
        state: dict[str, Any],
    ) -> list[WhitespaceItem]:
        used_labels = {
            str(item.get("title") or "").strip().casefold()
            for item in state.get("ideas", [])
            if isinstance(item, dict)
        }
        current = state.get("current_whitespace")
        if isinstance(current, dict):
            used_labels.add(str(current.get("label") or "").strip().casefold())

        selected = next(
            (item for item in items if item.label.strip().casefold() not in used_labels),
            items[0],
        )
        return [selected, *[item for item in items if item.label != selected.label]]

    def _rank_ideas(self, state: dict[str, Any]) -> list[tuple[Idea, ScoreCard]]:
        ranked: list[tuple[Idea, ScoreCard]] = []
        for item in state.get("ideas", []):
            try:
                idea = self._with_session_goals(Idea(**item), state)
            except TypeError:
                continue
            score, _event = self.tools.score_idea(idea)
            score = self._align_score_from_state(score, idea, state)
            idea.score = score.to_dict()
            ranked.append((idea, score))
        return sorted(
            ranked,
            key=lambda pair: (
                pair[1].overall,
                pair[1].originality,
                pair[1].ai_necessity,
                pair[0].title.casefold(),
            ),
            reverse=True,
        )

    def _opening_response(self, projects: list[Project]) -> str:
        names = ", ".join(project.title for project in projects[:4])
        return (
            "The current map is open with "
            f"{len(self.index.projects)} project pages; the brightest current echoes include {names}. "
            "Describe one project idea and I will test where it overlaps, where it is quiet, and what to build next."
        )

    def _overlap_response(self, idea: Idea, projects: list[Project], score: ScoreCard) -> str:
        if score.verdict.startswith("UNWRITTEN"):
            nearby = ", ".join(project.title for project in projects[:2]) or "no close pages"
            return (
                f"The page for {idea.title} does not bleed much. I found {nearby}, but the seal reads "
                f"{score.verdict} at {score.overall}/10. Push the AI necessity harder: make the model decide, rank, "
                "or personalize something a static app cannot."
            )
        citations = "; ".join(
            f"page {hit.page_number}: {hit.project.title}" for hit in score.echoes[:3]
        )
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
            f"The wax seal for {idea.title} reads {score.overall}/10, {score.verdict}. "
            f"The build path is: {steps}"
        )

    def _compare_response(self, ranked: list[tuple[Idea, ScoreCard]], plan: list[str]) -> str:
        winner, score = ranked[0]
        rows = []
        for index, (idea, item_score) in enumerate(ranked[:4], start=1):
            rows.append(
                f"{index}. {idea.title} — {item_score.overall}/10, {item_score.verdict}, "
                f"originality {item_score.originality}/10"
            )
        next_step = plan[0] if plan else "Make the top idea concrete enough to demo in one before/after scene."
        return (
            "Ranked pages: "
            + " | ".join(rows)
            + f". Keep {winner.title}: it has the strongest current seal at {score.overall}/10. Next: {next_step}"
        )

    def _artifact(self, idea: Idea, score: ScoreCard) -> dict[str, Any]:
        return {
            "title": idea.title,
            "verdict": score.verdict,
            "overall": score.overall,
            "caption": f"Idea page: {idea.title} - {score.verdict}.",
            "seal": score.to_dict(),
            "wood_map": build_wood_map(self.index, idea, score),
        }
