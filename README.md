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
```

The app uses `data/projects.json` at runtime, so deployed builds remain usable without live crawl calls.

## Test

```bash
pytest
```
