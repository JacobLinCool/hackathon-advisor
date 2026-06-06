from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from gradio import Server

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.trace_export import build_trace_jsonl, trace_metadata


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
DATA_PATH = ROOT / "data" / "projects.json"
INDEX_PATH = ROOT / "data" / "project_index.json"

index = ProjectIndex.from_files(DATA_PATH, INDEX_PATH)
engine = AdvisorEngine(index)
app = Server()


def _json_event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/static/{path:path}")
def static_file(path: str) -> FileResponse:
    target = (STATIC_DIR / path).resolve()
    if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "projects": len(index.projects),
        **trace_metadata(index),
    }


@app.get("/api/bootstrap")
def bootstrap() -> dict:
    return {
        "project_count": len(index.projects),
        **trace_metadata(index),
        "top_projects": [project.to_public_dict() for project in index.top_projects(limit=8)],
        "whitespace": [item.to_dict() for item in index.find_whitespace(limit=5)],
    }


@app.api(name="trace_artifact", concurrency_limit=8)
def trace_artifact(session_json: str = "{}") -> str:
    try:
        session = json.loads(session_json or "{}")
    except json.JSONDecodeError:
        session = {}
    return build_trace_jsonl(session, trace_metadata(index))


@app.api(name="agent_turn", concurrency_limit=4, stream_every=0.04)
def agent_turn(message: str, session_json: str = "{}") -> Iterator[str]:
    try:
        session = json.loads(session_json or "{}")
    except json.JSONDecodeError:
        session = {}

    result = engine.turn(message, session)
    yield _json_event(
        {
            "type": "start",
            "corrections": [correction.to_dict() for correction in result.corrections],
            "normalized_text": result.normalized_text,
            "tool_events": [event.to_dict() for event in result.tool_events],
        }
    )

    for chunk in result.stream_chunks():
        yield _json_event({"type": "token", "text": chunk})

    yield _json_event(
        {
            "type": "done",
            "state": result.state,
            "response": result.response,
            "projects": [project.to_public_dict() for project in result.projects],
            "whitespace": [item.to_dict() for item in result.whitespace],
            "score": result.score.to_dict() if result.score else None,
            "plan": result.plan,
            "artifact": result.artifact,
        }
    )


if __name__ == "__main__":
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
