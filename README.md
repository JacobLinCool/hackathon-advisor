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
short_description: Originality advisor for small-model project ideas.
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
**The Unwritten Almanac**: a journal-style workspace that compares your idea against real Spaces in the
`build-small-hackathon` organization, finds under-explored territory, scores the idea, and drafts a practical build plan.

The current milestone is a deployed ZeroGPU + MiniCPM5 LoRA advisor:

- Local snapshot of public `build-small-hackathon` Spaces.
- Offline search over project titles, tags, models, and descriptions.
- Jargon correction for hackathon/model terms.
- MiniCPM5 tool-call planning with a published PEFT LoRA adapter, plus deterministic local rules for tests and CPU-only
  development.
- One-turn advisor loop with overlap citations, whitespace suggestions, scoring, and plans.
- Custom `gradio.Server` frontend focused on the builder's idea workflow, with submission evidence kept in API exports.

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

The app exposes a `trace_artifact` Gradio API endpoint for submission evidence and debugging. It emits a manifest row
followed by one row per agent turn. `data/sample_trace.jsonl` is a checked-in, Hub-published sample trace. This endpoint
is intentionally kept out of the main user workflow.

## Field Notes Artifact

The `field_notes` Gradio API endpoint and `Notes` button export a Markdown build note from the exact session state:
builder profile, selected goals, idea board, cited Spaces, latest build plan, advisor actions, and the share caption. This
keeps the note tied to auditable app evidence instead of a separate hand-written summary.

## Chapter Artifact

The `chapter` Gradio API endpoint and `Chapter` button export the public-facing idea board as an Almanac chapter:
one idea page per saved direction, each with verdict, score, selected goals, and closest cited pages. It is the
shareable companion to the working notes artifact.

## Idea Board Compare

The `Compare` command rescans the saved idea board, recalculates each seal against the selected goals, selects the
strongest page as the active idea, and drafts the next build step. The app then moves that page to the top of the Idea
Board and refreshes the seal, wood map, plan, and PNG artifact around the chosen direction.
Users can also click any Idea Board page to make it current before pressing `Plan`.
If the board is empty, `Plan` and `Compare` do not create placeholder pages; they prompt the user to write an idea or
press `Gap` first.

## Gap Exploration

The `Gap` command walks through unused whitespace candidates instead of repeating the same first suggestion. Each chosen
gap becomes a new Idea Board page, so users can compare several genuinely different directions before ranking or
planning.

## Profile-Aware Plans

The `Profile` panel is part of the planning loop. Skills, time, preferences, and constraints are stored in the session
and inserted into `Plan` and `Compare` build paths, so the app can turn "one evening", "frontend prototyping", or
"CPU-only Space" into concrete scoping steps instead of generic advice.

## LoRA Dataset Artifact

The `lora_dataset` Gradio API endpoint exports a compact chat JSONL dataset from successful session turns. Each included
turn yields a tool-call example and an advisor-response example for `openbmb/MiniCPM5-1B`, with the selected goals,
parsed XML tool call, tool observations, and score context preserved. This is the dataset format used to train the
published MiniCPM5 LoRA adapter.

## LoRA Training Kit

`/api/lora-training-kit.zip` exports the training kit for the deterministic demo session: SFT JSONL, training recipe,
adapter model card, and the exact training command. The included `scripts/train_minicpm_lora.py` entrypoint supports a
dependency-light `--dry-run` validation path and a real `transformers + PEFT` training path that can publish the adapter
to `build-small-hackathon/hackathon-advisor-minicpm5-lora` with `--push-to-hub`.

## Submission Packet

The `submission_packet` Gradio API endpoint exports a Markdown submission bundle for the current session: live links,
snapshot provenance, a timed demo script, artifact checklist, Prize Ledger evidence, model budget, session trace
summary, social post draft, and open badge gaps. This keeps the final submission story tied to the same auditable state
as the app instead of a separate hand-curated checklist.

## Demo Rehearsal

`/api/demo-session` and the `Example` button load a deterministic two-turn sample: a complete project idea, profile,
selected goals, score seal, build plan, trace, and wood map. It is built by running the same advisor engine as a normal
user session, so the visible app stays focused on the builder's idea while API exports remain available for submission
evidence.

## Demo Evidence Bundle

`/api/demo-bundle.zip` downloads a server-built ZIP for the deterministic demo session. The bundle includes a manifest,
demo session JSON, Prize Ledger JSON, trace JSONL, Field Notes, Almanac chapter, LoRA SFT JSONL, LoRA training kit,
Submission Packet, and the rendered fate-page PNG. This gives judges or collaborators one auditable package without
depending on browser `localStorage`.

## Prize Ledger

`/api/prize-ledger` exposes submission evidence: the documented model stack, total parameter budget, Tiny Titan
eligibility, runtime backend, and badge readiness. It is kept as an API artifact rather than a primary in-app panel so
the user-facing app stays centered on idea evaluation. The main `/api/bootstrap` payload does not include the ledger.

## Wood Map

Every scored fate page now carries a deterministic `wood_map` artifact: background dots for inked Spaces, red dots for
the closest cited echoes, and a green/red "you" dot for the current idea. The live UI and PNG export render the same
map, so the share artifact visually proves whether the page sits in an empty margin or near existing work.
The `PNG` button posts the current artifact to `/api/artifact.png`, which uses the same Pillow renderer as
`/api/demo-bundle.zip`, so browser downloads and bundled evidence cannot drift into different layouts.

## Latency Watchdog

The custom frontend shows optimistic ink immediately after submit. If the first streamed token is slow, a lightweight
watchdog updates the page text so the demo never sits in a silent blank state during Space startup or model routing.

## Session Persistence

The frontend stores the current advisor session in browser `localStorage`: profile notes, selected goals, idea board,
trace, latest build plan, and last share artifact. Refreshing the Space restores the same cockpit state; the `Reset`
button clears the saved session and returns to the current snapshot defaults.

## Tool-Call Contract

`/api/tool-contracts` exposes the JSON schemas intended for MiniCPM-style tool calling. `tool_contract_check` accepts a
MiniCPM XML call such as `<function name="search_projects">{"query":"lullaby audio"}</function>`, validates it against
the schemas, and returns either the valid call or a safe default call for the UI watchdog path.

## Runtime Backend

The deployed Space is configured for ZeroGPU inference with:

```bash
ADVISOR_ZERO_GPU=1
ADVISOR_ZERO_GPU_DURATION=60
ADVISOR_MODEL_BACKEND=minicpm-transformers
ADVISOR_MODEL_ID=openbmb/MiniCPM5-1B
ADVISOR_ADAPTER_ID=build-small-hackathon/hackathon-advisor-minicpm5-lora
ADVISOR_ADAPTER_REVISION=25de69bcde397e1bcdd852923b56a42f10222650
```

`agent_turn` wraps the engine call with `spaces.GPU` when `ADVISOR_ZERO_GPU=1`, so model loading and generation run on
the ZeroGPU allocation. Local tests and CPU-only development still default to `ADVISOR_MODEL_BACKEND=rules`.

## Test

```bash
pytest
```
