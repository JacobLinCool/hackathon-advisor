#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hackathon_advisor.agent import AdvisorEngine
from hackathon_advisor.data import ProjectIndex
from hackathon_advisor.trace_export import build_trace_jsonl, trace_metadata


SAMPLE_TURNS = (
    "A local-first archive cartographer for family photos",
    "write bolder and find whitespace",
    "make a build plan",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a publishable sample agent trace JSONL.")
    parser.add_argument("--projects", default="data/projects.json")
    parser.add_argument("--index", default="data/project_index.json")
    parser.add_argument("--out", default="data/sample_trace.jsonl")
    args = parser.parse_args()

    index = ProjectIndex.from_files(Path(args.projects), Path(args.index))
    engine = AdvisorEngine(index)
    state = {}
    for turn in SAMPLE_TURNS:
        state = engine.turn(turn, state).state

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_trace_jsonl(state, trace_metadata(index)), encoding="utf-8")
    print(f"wrote {len(state.get('trace', []))} turns to {output}")


if __name__ == "__main__":
    main()
