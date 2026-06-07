#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from pathlib import PurePosixPath
import sys
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hackathon_advisor.data import extract_app_file_embedding_text


API = "https://huggingface.co/api"


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot public Spaces in a Hugging Face org.")
    parser.add_argument("--org", default="build-small-hackathon")
    parser.add_argument("--out", default="data/projects.json")
    args = parser.parse_args()

    projects = crawl_projects(args.org)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{API}/spaces?author={args.org}",
        "projects": sorted(projects, key=lambda project: project["id"].lower()),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(projects)} projects to {output}")


def crawl_projects(org: str) -> list[dict[str, Any]]:
    api = HfApi(token=False)
    spaces = api.list_spaces(author=org, full=True, token=False)
    return [
        project_from_space(space)
        for space in spaces
        if not bool(getattr(space, "private", False))
    ]


def project_from_space(space: Any) -> dict[str, Any]:
    card = card_data(space)
    space_id = str(space.id)
    siblings = sibling_names(space)
    readme = download_repo_text(space_id, "README.md") if "README.md" in siblings else ""
    frontmatter = readme_frontmatter(readme)
    app_file = validate_app_file(str(frontmatter.get("app_file") or ""), space_id=space_id)
    app_file_embedding_text = ""
    if app_file:
        if app_file not in siblings:
            raise RuntimeError(f"{space_id} README frontmatter points to missing app_file: {app_file}")
        app_file_embedding_text = extract_app_file_embedding_text(
            app_file,
            download_repo_text(space_id, app_file),
        )

    title = str(card.get("title") or humanize_slug(space_id.rsplit("/", 1)[-1]))
    summary = str(card.get("short_description") or card.get("description") or "")
    return {
        "id": space_id,
        "title": title,
        "summary": summary,
        "tags": sorted(set(str(tag) for tag in (card.get("tags") or getattr(space, "tags", None) or []))),
        "models": [str(model) for model in getattr(space, "models", None) or card.get("models") or []],
        "datasets": [
            str(dataset) for dataset in getattr(space, "datasets", None) or card.get("datasets") or []
        ],
        "likes": int(getattr(space, "likes", None) or 0),
        "sdk": str(card.get("sdk") or getattr(space, "sdk", None) or ""),
        "license": str(card.get("license") or ""),
        "created_at": isoformat(getattr(space, "created_at", None)),
        "last_modified": isoformat(getattr(space, "last_modified", None)),
        "host": host_url(space),
        "url": f"https://huggingface.co/spaces/{space_id}",
        "app_file": app_file,
        "app_file_embedding_text": app_file_embedding_text,
    }


def card_data(space: Any) -> dict[str, Any]:
    raw = getattr(space, "card_data", None) or getattr(space, "cardData", None) or {}
    if isinstance(raw, dict):
        return raw
    to_dict = getattr(raw, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {}


def sibling_names(space: Any) -> set[str]:
    return {str(sibling.rfilename) for sibling in getattr(space, "siblings", None) or []}


def download_repo_text(repo_id: str, filename: str) -> str:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="space",
        filename=filename,
        token=False,
        etag_timeout=30,
    )
    return Path(path).read_text(encoding="utf-8")


def readme_frontmatter(readme: str) -> dict[str, str]:
    lines = readme.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    values: dict[str, str] = {}
    closed = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped in {"---", "..."}:
            closed = True
            break
        if not stripped or stripped.startswith("#") or ":" not in line:
            continue
        if line[:1].isspace() or stripped.startswith("-"):
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if key:
            values[key] = yaml_scalar(raw_value)
    return values if closed else {}


def yaml_scalar(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
        return value[1:-1]
    return value


def validate_app_file(app_file: str, *, space_id: str) -> str:
    cleaned = app_file.strip()
    if not cleaned:
        return ""
    path = PurePosixPath(cleaned)
    if path.is_absolute() or ".." in path.parts or cleaned.endswith("/"):
        raise RuntimeError(f"{space_id} README frontmatter has an invalid app_file path: {app_file}")
    return path.as_posix()


def isoformat(value: Any) -> str:
    if value is None:
        return ""
    formatter = getattr(value, "isoformat", None)
    if callable(formatter):
        return formatter()
    return str(value)


def host_url(space: Any) -> str:
    host = str(getattr(space, "host", None) or "")
    if host:
        return host
    subdomain = str(getattr(space, "subdomain", None) or "")
    return f"https://{subdomain}.hf.space" if subdomain else ""


def humanize_slug(slug: str) -> str:
    return " ".join(part for part in slug.replace("_", "-").split("-") if part).title()


if __name__ == "__main__":
    main()
