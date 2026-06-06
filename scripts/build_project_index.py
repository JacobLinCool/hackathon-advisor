#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hackathon_advisor.data import Project, build_index_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the offline project retrieval index.")
    parser.add_argument("--projects", default="data/projects.json")
    parser.add_argument("--out", default="data/project_index.json")
    args = parser.parse_args()

    project_path = Path(args.projects)
    data = json.loads(project_path.read_text(encoding="utf-8"))
    projects = [Project.from_dict(item) for item in data["projects"]]
    payload = build_index_payload(
        projects=projects,
        snapshot_generated_at=str(data.get("generated_at") or ""),
        source=str(data.get("source") or ""),
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "wrote "
        f"{payload['document_count']} docs, {payload['vocabulary_size']} terms "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
