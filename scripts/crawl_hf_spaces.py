#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


API = "https://huggingface.co/api"


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot public Spaces in a Hugging Face org.")
    parser.add_argument("--org", default="build-small-hackathon")
    parser.add_argument("--out", default="data/projects.json")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    spaces = fetch_json(f"{API}/spaces?author={quote(args.org)}&limit={args.limit}")
    projects = []
    for item in spaces:
        space_id = item["id"]
        detail = fetch_json(f"{API}/spaces/{quote(space_id, safe='/')}")
        projects.append(project_from_detail(detail))
        time.sleep(0.05)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{API}/spaces?author={args.org}&limit={args.limit}",
        "projects": sorted(projects, key=lambda project: project["id"].lower()),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(projects)} projects to {output}")


def fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "hackathon-advisor-crawler/0.1"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"failed to fetch {url}: {error.code}") from error


def project_from_detail(detail: dict[str, Any]) -> dict[str, Any]:
    card = detail.get("cardData") or {}
    space_id = str(detail["id"])
    title = str(card.get("title") or humanize_slug(space_id.rsplit("/", 1)[-1]))
    summary = str(card.get("short_description") or card.get("description") or "")
    tags = sorted(set(str(tag) for tag in (card.get("tags") or detail.get("tags") or [])))
    return {
        "id": space_id,
        "title": title,
        "summary": summary,
        "tags": tags,
        "models": [str(model) for model in detail.get("models") or card.get("models") or []],
        "datasets": [str(dataset) for dataset in detail.get("datasets") or card.get("datasets") or []],
        "likes": int(detail.get("likes") or 0),
        "sdk": str(card.get("sdk") or detail.get("sdk") or ""),
        "license": str(card.get("license") or ""),
        "created_at": str(detail.get("createdAt") or ""),
        "last_modified": str(detail.get("lastModified") or ""),
        "host": str(detail.get("host") or ""),
        "url": f"https://huggingface.co/spaces/{space_id}",
    }


def humanize_slug(slug: str) -> str:
    return " ".join(part for part in slug.replace("_", "-").split("-") if part).title()


if __name__ == "__main__":
    main()
