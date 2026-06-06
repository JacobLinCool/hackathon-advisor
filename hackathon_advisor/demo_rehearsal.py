from __future__ import annotations

from typing import Any


DEMO_PROMPT = (
    "A local-first archive cartographer for family photos that runs offline, cites nearby Spaces, "
    "and exports Field Notes."
)
DEMO_PLAN_PROMPT = "make a build plan"
DEMO_PROFILE = {
    "skills": "frontend prototyping, Python, small-model evaluation",
    "time": "one focused weekend",
    "preferences": "auditable artifacts, local-first runtime, strong demo beat",
    "constraints": "CPU Space runtime; no proprietary inference API",
}
DEMO_TARGETS = [
    "Off the Grid",
    "Well-Tuned",
    "Off-Brand",
    "Sharing is Caring",
    "Field Notes",
]


def build_demo_rehearsal(engine: Any) -> dict[str, Any]:
    initial_state = {
        "profile": dict(DEMO_PROFILE),
        "targets": list(DEMO_TARGETS),
    }
    first = engine.turn(DEMO_PROMPT, initial_state)
    second = engine.turn(DEMO_PLAN_PROMPT, first.state)
    score = second.score or first.score
    artifact = second.artifact or first.artifact
    projects = first.projects or second.projects
    whitespace = first.whitespace or second.whitespace
    session = second.state
    return {
        "prompt": DEMO_PROMPT,
        "plan_prompt": DEMO_PLAN_PROMPT,
        "response": second.response,
        "session": session,
        "score": score.to_dict() if score else None,
        "plan": list(second.plan),
        "artifact": artifact,
        "projects": [project.to_public_dict() for project in projects],
        "whitespace": [item.to_dict() for item in whitespace],
        "turn_count": len(session.get("trace") or []),
        "export_ready": {
            "trace": bool(session.get("trace")),
            "notes": bool(session.get("trace")),
            "chapter": bool(session.get("ideas")),
            "lora_dataset": bool(session.get("trace")),
            "submission_packet": bool(session.get("trace")),
            "png": bool(artifact),
        },
    }
