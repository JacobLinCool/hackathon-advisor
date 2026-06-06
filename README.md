---
title: Hackathon Advisor
emoji: "📜"
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.16.0
python_version: "3.11"
app_file: app.py
pinned: true
license: mit
short_description: Originality advisor for Build Small.
tags:
  - gradio
  - build-small-hackathon
  - small-models
  - agent
  - originality
  - off-the-grid
---

# Hackathon Advisor

**Hackathon Advisor** is a text-first project advisor for the Build Small Hackathon. The user-facing experience is
**The Unwritten Almanac**: Mothback, an archivist of unwritten project pages, compares your idea against real Spaces in
the `build-small-hackathon` organization, finds under-explored territory, scores the idea, and drafts a practical build
plan.

The current milestone is a deployable, deterministic vertical slice:

- Local snapshot of public `build-small-hackathon` Spaces.
- Offline search over project titles, tags, models, and descriptions.
- Jargon correction for hackathon/model terms.
- One-turn advisor loop with overlap citations, whitespace suggestions, scoring, and plans.
- Custom `gradio.Server` frontend with streaming API events.

See [DESIGN.md](DESIGN.md) for the full product and model plan.

## Run Locally

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:7860>.

## Refresh The Project Snapshot

```bash
python scripts/crawl_hf_spaces.py --org build-small-hackathon --out data/projects.json
python scripts/build_project_index.py --projects data/projects.json --out data/project_index.json
python scripts/generate_sample_trace.py --projects data/projects.json --index data/project_index.json --out data/sample_trace.jsonl
```

The app uses `data/projects.json` and `data/project_index.json` at runtime. The index validates the snapshot timestamp,
source, project order, and digest before the app starts.

## Trace Artifact

The app exposes a `trace_artifact` Gradio API endpoint and a `JSONL` button in the UI. Both emit the same JSONL schema:
a manifest row followed by one row per agent turn. `data/sample_trace.jsonl` is a checked-in, Hub-published sample trace.

## Field Notes Artifact

The `field_notes` Gradio API endpoint and `Notes` button export a Markdown build note from the exact session state:
builder profile, target badges, idea board, cited Spaces, latest build plan, planner calls, and the share caption. This
keeps the Field Notes badge path tied to auditable app evidence instead of a separate hand-written summary.

## Wood Map

Every scored fate page now carries a deterministic `wood_map` artifact: background dots for inked Spaces, red dots for
the closest cited echoes, and a green/red "you" dot for the current idea. The live UI and PNG export render the same
map, so the share artifact visually proves whether the page sits in an empty margin or near existing work.

## Latency Watchdog

The custom frontend shows optimistic ink immediately after submit. If the first streamed token is slow, a lightweight
watchdog updates the page text so the demo never sits in a silent blank state during Space startup or model routing.

## Session Persistence

The frontend stores the current advisor session in browser `localStorage`: profile notes, selected targets, idea board,
trace, latest build plan, and last share artifact. Refreshing the Space restores the same cockpit state; the `Reset`
button clears the saved session and returns to the current snapshot defaults.

## Tool-Call Contract

`/api/tool-contracts` exposes the JSON schemas intended for MiniCPM-style tool calling. `tool_contract_check` accepts a
MiniCPM XML call such as `<function name="search_projects">{"query":"lullaby audio"}</function>`, validates it against
the schemas, and returns either the valid call or a safe default call for the UI watchdog path.

## Runtime Backend

The deployed Space defaults to `ADVISOR_MODEL_BACKEND=rules`, a deterministic planner that emits the same validated XML
tool calls as the MiniCPM path. To enable the optional MiniCPM adapter in a GPU environment, install the `model` extra
and set `ADVISOR_MODEL_BACKEND=minicpm-transformers` plus `ADVISOR_MODEL_ID=openbmb/MiniCPM5-1B`.

## Test

```bash
pytest
```
