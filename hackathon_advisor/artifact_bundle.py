from __future__ import annotations

from io import BytesIO
import json
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from hackathon_advisor.chapter import build_chapter_markdown
from hackathon_advisor.field_notes import build_field_notes_markdown
from hackathon_advisor.lora_dataset import build_lora_dataset_jsonl
from hackathon_advisor.lora_training_kit import build_lora_training_kit_zip
from hackathon_advisor.png_export import artifact_png_filename, render_artifact_png
from hackathon_advisor.submission_packet import build_submission_packet_markdown
from hackathon_advisor.trace_export import build_trace_jsonl
from hackathon_advisor._text import clean as _clean, utc_now


BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FILENAME = "hackathon-advisor-demo-bundle.zip"


def build_demo_bundle_zip(
    demo: dict[str, Any],
    metadata: dict[str, Any],
    ledger: dict[str, Any],
) -> bytes:
    session = demo.get("session") if isinstance(demo.get("session"), dict) else {}
    files = _bundle_files(session, metadata, ledger, demo)
    manifest = _manifest(files, metadata, ledger, demo)

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        archive.writestr("manifest.json", f"{manifest_json}\n")
        for filename, content in files.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


def _bundle_files(
    session: dict[str, Any],
    metadata: dict[str, Any],
    ledger: dict[str, Any],
    demo: dict[str, Any],
) -> dict[str, str | bytes]:
    artifact = demo.get("artifact") if isinstance(demo.get("artifact"), dict) else {}
    png_filename = artifact_png_filename(artifact)
    ledger_json = json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True)
    return {
        "demo-session.json": json.dumps(demo, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "prize-ledger.json": f"{ledger_json}\n",
        "trace.jsonl": build_trace_jsonl(session, metadata),
        "field-notes.md": build_field_notes_markdown(session, metadata),
        "almanac-chapter.md": build_chapter_markdown(session, metadata),
        "lora-sft.jsonl": build_lora_dataset_jsonl(session, metadata),
        "lora-training-kit.zip": build_lora_training_kit_zip(session, metadata, ledger),
        "submission-packet.md": build_submission_packet_markdown(session, metadata, ledger),
        png_filename: render_artifact_png(artifact),
    }


def _manifest(
    files: dict[str, str | bytes],
    metadata: dict[str, Any],
    ledger: dict[str, Any],
    demo: dict[str, Any],
) -> dict[str, Any]:
    badges = ledger.get("badges") if isinstance(ledger.get("badges"), list) else []
    return {
        "type": "demo_bundle_manifest",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "app": "hackathon-advisor",
        "turn_count": int(demo.get("turn_count") or 0),
        "file_count": len(files),
        "files": list(files),
        "runtime": ledger.get("runtime") if isinstance(ledger.get("runtime"), dict) else {},
        "badge_status": {
            str(badge.get("name")): str(badge.get("status"))
            for badge in badges
            if isinstance(badge, dict)
        },
        "index": {
            "algorithm": _clean(metadata.get("index_algorithm")),
            "snapshot_generated_at": _clean(metadata.get("snapshot_generated_at")),
            "index_generated_at": _clean(metadata.get("index_generated_at")),
            "snapshot_digest": _clean(metadata.get("snapshot_digest")),
        },
    }
