#!/usr/bin/env python3
"""Modal wiring for the Codex trace privacy-filter publisher.

The user-facing entrypoint is `scripts/publish_codex_trace_dataset.py --location modal`,
which calls `run_modal` below. The publisher core (selection, redaction, dataset build,
upload) lives in `scripts.publish_codex_trace_dataset`; this module only owns the Modal
app/image/volume and the GPU remote function.

Local work: select project-relevant Codex session JSONL, upload raw files to a Modal
Volume, receive the filtered dataset zip, and upload it from local Hugging Face creds.
Remote work: run the same core, applying openai/privacy-filter on CUDA.
"""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import zipfile

import modal

from scripts.publish_codex_trace_dataset import (
    TextCaps,
    build_project_terms,
    default_session_roots,
    discover_session_files,
    display_path,
    session_matches_project,
    sha256_file,
    upload_dataset,
)

APP_NAME = "hackathon-advisor-codex-trace-publisher"
GPU = "A10G"
VOLUME_NAME = "hackathon-advisor-codex-trace-inputs"
VOLUME_MOUNT = "/codex-trace-inputs"

app = modal.App(APP_NAME)
input_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub>=1.5,<2",
        "torch>=2.8,<3",
        "transformers>=5.6,<6",
    )
    .add_local_python_source("scripts", copy=True)
)


def selected_sessions(project_root: Path, session_roots: list[Path], include_terms: list[str]) -> list[dict]:
    terms = build_project_terms(project_root, include_terms)
    selected: list[dict] = []
    for path in discover_session_files(session_roots):
        matched, reason = session_matches_project(path, terms)
        if not matched:
            continue
        selected.append(
            {
                "path": str(path),
                "filename": path.name,
                "source_path": display_path(path),
                "selected_reason": reason.replace(str(project_root), "$PROJECT_ROOT").replace(str(Path.home()), "~"),
                "source_sha256": sha256_file(path),
                "source_size_bytes": path.stat().st_size,
            }
        )
    if not selected:
        raise RuntimeError("no Codex session JSONL files matched the project terms")
    return selected


def upload_inputs_to_volume(run_id: str, sessions: list[dict]) -> None:
    with input_volume.batch_upload(force=True) as batch:
        batch.put_file(
            io.BytesIO(json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2).encode("utf-8")),
            f"/{run_id}/selected_sessions.json",
        )
        for item in sessions:
            batch.put_file(item.get("upload_path", item["path"]), f"/{run_id}/sessions/{item['filename']}")


def snapshot_sessions(run_id: str, sessions: list[dict], out_dir: Path) -> list[dict]:
    snapshot_dir = out_dir.parent / "codex-trace-modal-input" / run_id / "sessions"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshotted: list[dict] = []
    for item in sessions:
        source = Path(item["path"])
        target = snapshot_dir / item["filename"]
        shutil.copy2(source, target)
        copied = dict(item)
        copied["upload_path"] = str(target)
        copied["source_sha256"] = sha256_file(target)
        copied["source_size_bytes"] = target.stat().st_size
        snapshotted.append(copied)
    return snapshotted


@app.function(image=image, gpu=GPU, timeout=7200)
def smoke() -> dict:
    import torch

    return {
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }


@app.function(image=image, gpu=GPU, timeout=7200, volumes={VOLUME_MOUNT: input_volume})
def filter_remote(
    run_id: str,
    *,
    project_root: str,
    include_terms: list[str],
    repo_id: str,
    path_redaction_prefixes: list[str],
    privacy_filter_model: str,
    privacy_filter_min_score: float,
    privacy_filter_batch_size: int,
    privacy_filter_chunk_chars: int,
    record_batch_size: int,
    progress_interval_batches: int,
    text_caps_payload: dict,
) -> dict:
    from pathlib import Path
    import logging
    import zipfile

    from scripts.publish_codex_trace_dataset import (
        PrivacyFilterRedactor,
        TextCaps,
        build_dataset,
        dataset_card,
        model_revision,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    input_volume.reload()
    run_dir = Path(VOLUME_MOUNT) / run_id
    session_dir = run_dir / "sessions"
    selected_path = run_dir / "selected_sessions.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")).get("sessions", [])
    source_by_sha = {item["source_sha256"]: item for item in selected}
    out_dir = Path("/tmp") / f"codex-trace-dataset-{run_id}"
    revision = model_revision(privacy_filter_model)
    redactor = PrivacyFilterRedactor(
        privacy_filter_model,
        min_score=privacy_filter_min_score,
        batch_size=privacy_filter_batch_size,
        chunk_chars=privacy_filter_chunk_chars,
        device="cuda",
    )
    manifest = build_dataset(
        project_root=Path(project_root),
        session_roots=[session_dir],
        include_terms=[*include_terms, project_root],
        out_dir=out_dir,
        redactor=redactor,
        privacy_model_id=privacy_filter_model,
        privacy_model_revision=revision,
        privacy_device=redactor.device,
        min_score=privacy_filter_min_score,
        record_batch_size=record_batch_size,
        progress_interval_batches=progress_interval_batches,
        text_caps=TextCaps(**text_caps_payload),
        path_redaction_prefixes=path_redaction_prefixes,
    )
    for session in manifest["sessions"]:
        source = source_by_sha.get(session["source_sha256"])
        if source:
            session["source_path"] = source["source_path"]
            session["selected_reason"] = source["selected_reason"]
            session["source_size_bytes"] = source["source_size_bytes"]
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(dataset_card(manifest, repo_id), encoding="utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir).as_posix())
    return {
        "dataset_zip": buffer.getvalue(),
        "manifest": manifest,
    }


def run_modal(args) -> None:
    """Run the publisher on Modal GPU.

    Invoked by `publish_codex_trace_dataset.py --location modal` (a plain Python process),
    so this opens its own ephemeral Modal app context. The caller's local home is passed
    explicitly in `path_redaction_prefixes` because `Path.home()` inside the container is
    `/root`, not the user's machine.
    """
    project = args.project_root.expanduser().resolve()
    roots = args.session_roots or default_session_roots()
    include_terms = list(args.include or [])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = args.out_dir
    sessions = snapshot_sessions(run_id, selected_sessions(project, roots, include_terms), output)
    total_bytes = sum(int(item["source_size_bytes"]) for item in sessions)
    print(f"selected {len(sessions)} sessions ({total_bytes / 1024 / 1024:.1f} MiB raw)")
    for index, item in enumerate(sessions, start=1):
        print(f"  {index}. {item['source_path']} ({item['source_size_bytes'] / 1024 / 1024:.1f} MiB)")
    print(f"uploading raw sessions to Modal volume {VOLUME_NAME}/{run_id}")
    upload_inputs_to_volume(run_id, sessions)

    caps = TextCaps(
        message=args.max_message_chars,
        tool_argument=args.max_tool_argument_chars,
        tool_output=args.max_tool_output_chars,
        other=args.max_other_text_chars,
    )
    with app.run():
        result = filter_remote.remote(
            run_id,
            project_root=str(project),
            include_terms=include_terms,
            repo_id=args.repo_id,
            path_redaction_prefixes=[str(project), str(Path.home())],
            privacy_filter_model=args.privacy_filter_model,
            privacy_filter_min_score=args.privacy_filter_min_score,
            privacy_filter_batch_size=args.privacy_filter_batch_size,
            privacy_filter_chunk_chars=args.privacy_filter_chunk_chars,
            record_batch_size=args.record_batch_size,
            progress_interval_batches=args.progress_interval_batches,
            text_caps_payload=caps.__dict__,
        )

    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(result["dataset_zip"])) as zf:
        zf.extractall(output)
    manifest = result["manifest"]
    print(
        "filtered dataset: "
        f"{manifest['selected_session_count']} sessions, "
        f"{manifest['published_record_count']} records, "
        f"{manifest['redaction_count']} privacy redactions, "
        f"{manifest['truncated_field_count']} truncated fields"
    )
    if args.skip_upload:
        print(f"wrote dataset staging directory: {output}")
        return
    revision = upload_dataset(output, args.repo_id, manifest)
    print(f"published dataset https://huggingface.co/datasets/{args.repo_id}")
    print(f"revision: {revision}")
