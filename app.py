from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from fastapi import Body, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from gradio import Server

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.artifact_bundle import BUNDLE_FILENAME, build_demo_bundle_zip
from hackathon_advisor.asr_runtime import create_asr_transcriber
from hackathon_advisor.chapter import build_chapter_markdown
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.demo_rehearsal import build_demo_rehearsal
from hackathon_advisor.field_notes import build_field_notes_markdown
from hackathon_advisor.lora_dataset import build_lora_dataset_jsonl
from hackathon_advisor.lora_training_kit import TRAINING_KIT_FILENAME, build_lora_training_kit_zip
from hackathon_advisor.png_export import artifact_png_filename, render_artifact_png
from hackathon_advisor.prize_ledger import prize_ledger
from hackathon_advisor.runtime_hooks import install_asyncio_cleanup_hook
from hackathon_advisor.submission_packet import build_submission_packet_markdown
from hackathon_advisor.tool_contracts import resolve_tool_call, tool_schemas
from hackathon_advisor.tools import GOALS, goal_profiles
from hackathon_advisor.trace_export import build_trace_jsonl, trace_metadata
from hackathon_advisor.zerogpu import gpu_task


install_asyncio_cleanup_hook()

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
DATA_PATH = ROOT / "data" / "projects.json"
INDEX_PATH = ROOT / "data" / "project_index.json"
PROFILE_FIELDS = ["skills", "time", "preferences", "constraints"]
MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
AUDIO_UPLOAD_SUFFIXES = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus", ".wav", ".webm"}

index = ProjectIndex.from_files(DATA_PATH, INDEX_PATH)
engine = AdvisorEngine(index)
voice_transcriber = create_asr_transcriber()
app = Server()


def _json_event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@gpu_task
def _engine_turn(message: str, session: dict[str, Any]):
    return engine.turn(message, session)


@gpu_task
def _transcribe_voice(audio_path: str) -> dict[str, Any]:
    return voice_transcriber.transcribe(Path(audio_path)).to_dict()


def _session_from_json(session_json: str = "{}") -> dict[str, Any]:
    try:
        session = json.loads(session_json or "{}")
    except json.JSONDecodeError:
        return {}
    return session if isinstance(session, dict) else {}


def _session_from_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    return _session_from_json(str(payload.get("session_json") or "{}"))


def _agent_turn_events(message: str, session_json: str = "{}") -> Iterator[str]:
    session = _session_from_json(session_json)
    result = _engine_turn(message, session)
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
        "runtime": engine.runtime_status(),
        "voice": voice_transcriber.status().to_dict(),
        **trace_metadata(index),
    }


@app.get("/api/bootstrap")
def bootstrap() -> dict:
    runtime_status = engine.runtime_status()
    return {
        "project_count": len(index.projects),
        "runtime": runtime_status,
        "voice": voice_transcriber.status().to_dict(),
        **trace_metadata(index),
        "top_projects": [project.to_public_dict() for project in index.top_projects(limit=8)],
        "whitespace": [item.to_dict() for item in index.find_whitespace(limit=5)],
        "goal_options": GOALS,
        "goal_profiles": goal_profiles(),
        "default_goals": GOALS[:3],
        "profile_fields": PROFILE_FIELDS,
    }


@app.get("/api/runtime")
def runtime() -> dict:
    return engine.runtime_status()


@app.get("/api/prize-ledger")
def prize_ledger_endpoint() -> dict:
    return prize_ledger(engine.runtime_status(), trace_metadata(index), voice_transcriber.status().to_dict())


@app.get("/api/tool-contracts")
def tool_contracts() -> dict:
    return {
        "tool_count": len(tool_schemas()),
        "tools": tool_schemas(),
    }


@app.get("/api/demo-session")
def demo_session() -> dict:
    return build_demo_rehearsal(engine)


@app.get("/api/demo-bundle.zip")
def demo_bundle() -> Response:
    runtime_status = engine.runtime_status()
    ledger = prize_ledger(runtime_status, trace_metadata(index), voice_transcriber.status().to_dict())
    metadata = {
        **trace_metadata(index),
        "project_count": len(index.projects),
    }
    content = build_demo_bundle_zip(build_demo_rehearsal(engine), metadata, ledger)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{BUNDLE_FILENAME}"'},
    )


@app.post("/api/artifact.png")
def artifact_png(artifact: dict[str, Any] | None = Body(default=None)) -> Response:
    artifact = artifact or {}
    filename = artifact_png_filename(artifact)
    return Response(
        content=render_artifact_png(artifact),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/agent-turn")
def agent_turn_stream(payload: dict[str, Any] | None = Body(default=None)) -> StreamingResponse:
    payload = payload or {}
    message = str(payload.get("message") or "")
    session_json = str(payload.get("session_json") or "{}")

    def stream() -> Iterator[str]:
        for event in _agent_turn_events(message, session_json):
            yield f"{event}\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> dict[str, Any]:
    content_type = str(audio.content_type or "")
    filename = Path(str(audio.filename or "voice-note")).name
    suffix = Path(filename).suffix.lower() or ".audio"
    if not _is_audio_upload(content_type, suffix):
        raise HTTPException(status_code=415, detail="Voice input must be an audio file.")
    with tempfile.TemporaryDirectory(prefix="advisor-upload-") as directory:
        source = Path(directory) / f"voice{suffix}"
        await _save_audio_upload(audio, source)
        return _transcribe_voice(str(source))


def _is_audio_upload(content_type: str, suffix: str) -> bool:
    if content_type.startswith("audio/"):
        return True
    if content_type in {"", "application/octet-stream"} and suffix in AUDIO_UPLOAD_SUFFIXES:
        return True
    return False


async def _save_audio_upload(upload: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_AUDIO_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Voice note is too large.")
            handle.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Voice note is empty.")


@app.post("/api/field-notes")
def field_notes_api(payload: dict[str, Any] | None = Body(default=None)) -> Response:
    session = _session_from_payload(payload)
    content = build_field_notes_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )
    return Response(content=content, media_type="text/markdown; charset=utf-8")


@app.post("/api/chapter")
def chapter_api(payload: dict[str, Any] | None = Body(default=None)) -> Response:
    session = _session_from_payload(payload)
    content = build_chapter_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )
    return Response(content=content, media_type="text/markdown; charset=utf-8")


@app.get("/api/lora-training-kit.zip")
def lora_training_kit() -> Response:
    runtime_status = engine.runtime_status()
    ledger = prize_ledger(runtime_status, trace_metadata(index), voice_transcriber.status().to_dict())
    metadata = {
        **trace_metadata(index),
        "project_count": len(index.projects),
    }
    demo = build_demo_rehearsal(engine)
    session = demo.get("session") if isinstance(demo.get("session"), dict) else {}
    content = build_lora_training_kit_zip(session, metadata, ledger)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{TRAINING_KIT_FILENAME}"'},
    )


@app.api(name="tool_contract_check", concurrency_limit=8)
def tool_contract_check(model_output: str, fallback_query: str = "") -> dict:
    return resolve_tool_call(model_output, fallback_query=fallback_query).to_dict()


@app.api(name="trace_artifact", concurrency_limit=8)
def trace_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    return build_trace_jsonl(session, trace_metadata(index))


@app.api(name="field_notes", concurrency_limit=8)
def field_notes_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    return build_field_notes_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )


@app.api(name="chapter", concurrency_limit=8)
def chapter_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    return build_chapter_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )


@app.api(name="lora_dataset", concurrency_limit=8)
def lora_dataset_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    return build_lora_dataset_jsonl(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
    )


@app.api(name="submission_packet", concurrency_limit=8)
def submission_packet_artifact(session_json: str = "{}") -> str:
    session = _session_from_json(session_json)
    runtime_status = engine.runtime_status()
    return build_submission_packet_markdown(
        session,
        {
            **trace_metadata(index),
            "project_count": len(index.projects),
        },
        prize_ledger(runtime_status, trace_metadata(index), voice_transcriber.status().to_dict()),
    )


@app.api(name="agent_turn", concurrency_limit=4, stream_every=0.04)
def agent_turn(message: str, session_json: str = "{}") -> Iterator[str]:
    yield from _agent_turn_events(message, session_json)


if __name__ == "__main__":
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
