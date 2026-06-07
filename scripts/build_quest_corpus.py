#!/usr/bin/env python3
"""Download the real README.md and main app-file source for every project in the
snapshot, so the quest dataset can split each sample into a README segment and an
app-file segment of genuine content.

Output: data/quest_corpus.json — one record per project with the raw README, the
README body with YAML frontmatter stripped, the raw app-file source, the AST app
signals carried in the snapshot, and the character lengths used later for cleaning.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

from scripts.crawl_hf_spaces import readme_frontmatter

APP_SOURCE_CHAR_CAP = 24000
README_CHAR_CAP = 24000


def download_space_text(repo_id: str, filename: str) -> str:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="space",
            filename=filename,
            token=False,
            etag_timeout=30,
        )
    except (EntryNotFoundError, HfHubHTTPError, OSError):
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def strip_frontmatter(readme: str) -> str:
    lines = readme.splitlines()
    if not lines or lines[0].strip() != "---":
        return readme.strip()
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            return "\n".join(lines[index + 1 :]).strip()
    return readme.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl README + app-file source for the quest dataset.")
    parser.add_argument("--projects", default="data/projects.json", type=Path)
    parser.add_argument("--out", default="data/quest_corpus.json", type=Path)
    args = parser.parse_args()

    snapshot = json.loads(args.projects.read_text(encoding="utf-8"))
    projects = snapshot["projects"]
    records = []
    for position, project in enumerate(projects, start=1):
        repo_id = project["id"]
        app_file = (project.get("app_file") or "").strip()
        readme_raw = download_space_text(repo_id, "README.md")
        readme_body = strip_frontmatter(readme_raw)[:README_CHAR_CAP]
        frontmatter = readme_frontmatter(readme_raw)
        app_source = download_space_text(repo_id, app_file)[:APP_SOURCE_CHAR_CAP] if app_file else ""
        record = {
            "id": repo_id,
            "title": project.get("title") or "",
            "summary": project.get("summary") or "",
            "tags": project.get("tags") or [],
            "models": project.get("models") or [],
            "datasets": project.get("datasets") or [],
            "sdk": project.get("sdk") or "",
            "license": project.get("license") or "",
            "likes": int(project.get("likes") or 0),
            "url": project.get("url") or "",
            "app_file": app_file,
            "readme_raw": readme_raw,
            "readme_body": readme_body,
            "readme_frontmatter": frontmatter,
            "app_source": app_source,
            "app_signals": project.get("app_file_embedding_text") or "",
            "readme_len": len(readme_body),
            "app_source_len": len(app_source),
            "app_signals_len": len(project.get("app_file_embedding_text") or ""),
        }
        records.append(record)
        print(
            f"[{position:3d}/{len(projects)}] {repo_id:55s} "
            f"readme={record['readme_len']:5d} app={record['app_source_len']:5d}",
            flush=True,
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_snapshot": str(args.projects),
        "snapshot_generated_at": snapshot.get("generated_at") or "",
        "project_count": len(records),
        "projects": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} corpus records to {args.out}")


if __name__ == "__main__":
    main()
