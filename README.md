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
  - track:wood
  - sponsor:openbmb
  - sponsor:openai
  - sponsor:nvidia
  - sponsor:modal
  - achievement:offgrid
  - achievement:welltuned
  - achievement:offbrand
  - achievement:llama
  - achievement:sharing
  - achievement:fieldnotes
  - tiny-titan
  - best-demo
  - best-agent
  - bonus-quest-champion
models:
  - openbmb/MiniCPM5-1B
  - build-small-hackathon/hackathon-advisor-minicpm5-lora
  - build-small-hackathon/hackathon-advisor-quest-minicpm5-lora
  - ggml-org/embeddinggemma-300m-qat-q8_0-GGUF
  - nvidia/nemotron-speech-streaming-en-0.6b
datasets:
  - build-small-hackathon/hackathon-advisor-quest-dataset
  - build-small-hackathon/hackathon-advisor-codex-traces
---

# Hackathon Advisor

**Hackathon Advisor** is a live map of the Build Small Hackathon and a small-model originality coach for builders. It
opens on an atlas of public `build-small-hackathon` Spaces, then lets a builder search the field, inspect project
clusters, see quest evidence, and open **The Unwritten Almanac** to evaluate an idea against the work already on the
trail.

The [Build Small Hackathon](https://huggingface.co/build-small-hackathon) asks participants to build under a 32B
parameter cap, solve a concrete problem for someone nearby or make a delightful AI-native experience, and submit a Space,
demo video, and social post. Hackathon Advisor treats that setting as the data surface: every public Space becomes part
of a continuously refreshed project atlas, and every advisor response is grounded in that shared map.

## Demo

- Live app: <https://build-small-hackathon-hackathon-advisor.hf.space>
- Hugging Face Space: <https://huggingface.co/spaces/build-small-hackathon/hackathon-advisor>
- Source code (GitHub): <https://github.com/JacobLinCool/hackathon-advisor>
- Demo video: <https://youtu.be/Gq-FUiL-ZPw>
- Social post: <https://x.com/JacobLinCool/status/2066156056724848965>
- Submission notes: [`docs/submission-notes.md`](docs/submission-notes.md)
- Start at the Idea Map, search for a theme, click nearby projects, hover quest badges for evidence, and open the
  advisor when you are ready to test an idea.

## What This Establishes

Builders enter a fast-moving hackathon with limited context. A promising idea can already be crowded, a quiet niche can
be hard to see, and prize alignment can be scattered across READMEs, tags, and app files. Hackathon Advisor turns the
field itself into the starting point. The app shows where projects cluster, which submissions sit near each other, which
quests they appear to satisfy, and where a new idea may still have room to breathe.

The project also exists because the dataset is already there. A public hackathon organization is a living corpus of
Spaces, READMEs, model declarations, app files, tags, and demos. Once that corpus is indexed, builders can explore what
others are making, which methods they are using, and which results are emerging across the field.

That visibility gives an online hackathon some of the creative force of OpenAI's Parameter Golf challenge: the event
becomes a shared surface for ideas to interact while people are still building. Participants can find adjacent work,
recognize overlap, borrow useful patterns, extend ideas into new domains, and connect with others working on similar
problems. The result is a faster open-source feedback loop, where good ideas become easier to find, improve, and build
on together.

The atlas is the default experience because the map is the evidence. The advisor is available behind `Open advisor`,
where it uses the same project snapshot to cite overlap, propose whitespace, score the idea, draft a build plan, and
export the session evidence.

## Hackathon Submission

Hackathon Advisor is submitted primarily for the **Thousand Token Wood** track. The product is useful to builders, but
its core form is an AI-native field guide: a living map, an almanac voice, quest evidence, and shareable artifacts that
make the hackathon field itself explorable.

The demo video is part of that submission evidence. It is built from real app footage of the atlas and advisor flows. Codex helped draft the storyboard, drive the app, capture the screen, generate voice-over, compose the cut, and verify frames and ASR transcripts against the intended narration.

The Space is also targeting the official sponsor and achievement tags shown in the README front matter:

- `sponsor:openbmb`: MiniCPM5-1B is the central planner and quest-classifier base model.
- `sponsor:openai`: Codex served as the engineering partner across the build. It helped translate the hackathon
  requirements into implementation slices, inspect and revise the repository, implement the atlas
  refresh/storage/search paths, add the quest-evidence UI, run tests and browser checks, review deployed Space behavior,
  prepare commits and deployment updates, produce the demo storyboard/app-footage/voice-over checks, and refine the
  README narrative. The redacted project-facing Codex session traces are published as a Hugging Face dataset.
- `sponsor:nvidia`: voice input runs `nvidia/nemotron-speech-streaming-en-0.6b` through NVIDIA NeMo ASR.
- `sponsor:modal`: Modal is used for development compute, including the quest LoRA training path and remote index-build
  path, and is documented in this README.
- `achievement:offgrid`: runtime inference is local to the Space process; no proprietary cloud inference API is called.
- `achievement:welltuned`: the advisor and quest classifier use published MiniCPM LoRA adapters.
- `achievement:offbrand`: the app uses a custom `gradio.Server` frontend instead of the stock Gradio interface.
- `achievement:llama`: retrieval embeddings run through llama.cpp via `llama-cpp-python`.
- `achievement:sharing`: redacted Codex build traces are published as a Hugging Face dataset.
- `achievement:fieldnotes`: the repo includes build reports and field-note exports.

Additional prize evidence for Tiny Titan, Best Agent, Best Demo, and Bonus Quest Champion is summarized in
[`docs/submission-notes.md`](docs/submission-notes.md).

## What You Can Do

- Explore a full-screen t-SNE atlas of public hackathon Spaces, with KMeans clusters and nearest-neighbor links.
- Search projects with BM25 over titles, slugs, summaries, tags, declared models, cluster labels, quest evidence, README
  text, and declared app-file source.
- Filter by cluster or quest, then inspect the selected project's summary, Space link, tags, quest matches, and evidence
  hints.
- Chat with the atlas: the "Ask the atlas" drawer answers questions like "who completed the most quests" or "what
  clusters exist" through the BASE MiniCPM5-1B model's native tool calling, with thinking enabled — the reasoning
  trace streams live into a collapsible block. Verified tool results render as cards and can filter or highlight the
  map directly; the model's prose is grounded on a compact digest of the same result.
- Refresh the atlas from the Space backend; validated artifacts are written to the mounted cache directory and swapped
  into the live app atomically.
- Open the advisor workspace for idea comparison, gap exploration, score seals, profile-aware plans, voice input, and
  shareable exports.
- Export from the workspace UI: build notes, the Almanac chapter, and the page PNG. Further reviewer artifacts — trace
  JSONL, demo bundle, submission packet, LoRA dataset, and LoRA training kit — are served through the API endpoints
  listed below.

## How It Works

The refresh path snapshots public Spaces in the `build-small-hackathon` organization, reads each README and declared
main app file, rebuilds the EmbeddingGemma project index, resolves official `track:*`, `sponsor:*`, and
`achievement:*` metadata tags first, then asks MiniCPM only for quest evidence that metadata did not declare. The active
dashboard contains project points, nearest links, clusters, quest coverage, provenance, and refresh state.

`ADVISOR_CACHE_DIR` is the artifact store. On Hugging Face Spaces it points to the mounted Storage Bucket; locally it can
be a normal directory such as `.cache/advisor-dashboard`. Each refresh writes
`runs/{run_id}/projects.json`, `project_index.json`, `dashboard.json`, `quest_analysis.json`, and `manifest.json`, then
updates `latest.json` through an atomic swap. Quest analysis is cached per project using the metadata-first inference
prompt hash, taxonomy hash, MiniCPM model id, adapter id/revision, local adapter digest, and generation config.

The app can start an hourly scheduler when `ADVISOR_CACHE_DIR` is configured. The submitted Space keeps scheduled
refresh disabled and updates the mounted bucket from local builds, so index construction and quest analysis do not run
inside the public demo runtime. Manual and scheduled refreshes still acquire `$ADVISOR_CACHE_DIR/refresh.lock`,
heartbeat while active, and leave the current validated dashboard in place if a new run fails validation.

## Models And Data

| Role | Model | Runtime | Evidence |
| --- | --- | --- | --- |
| Advisor | [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) + [`build-small-hackathon/hackathon-advisor-minicpm5-lora`](https://huggingface.co/build-small-hackathon/hackathon-advisor-minicpm5-lora) | ZeroGPU, Transformers, PEFT | A 1.08B OpenBMB model plans which tool to call each turn; advisor prose is rendered from deterministic templates grounded in the retrieved tool results. |
| Quest analysis | Official Space metadata + [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) + [`build-small-hackathon/hackathon-advisor-quest-minicpm5-lora`](https://huggingface.co/build-small-hackathon/hackathon-advisor-quest-minicpm5-lora) | ZeroGPU, Transformers, PEFT | The analyzer trusts official README tags for declared tracks, sponsor prizes, and badges, then uses a task-specific MiniCPM LoRA to classify remaining README/app-file evidence into strict quest JSON. |
| Project retrieval | [`ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF) | Local llama.cpp index build plus llama.cpp query embeddings | The atlas and retrieval index use a GGUF embedding model through llama.cpp. |
| Voice input | [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) | ZeroGPU; NVIDIA NeMo ASR | Voice notes are transcribed with NVIDIA NeMo using the same Nemotron model in local and deployed runs. |

MiniCPM is loaded following the official demo shape (`trust_remote_code=True`, `bfloat16`, and
`apply_chat_template(..., enable_thinking=False)`) for stable tool calls and strict quest JSON.

| Data / released material | Link | How it is used |
| --- | --- | --- |
| Hackathon project corpus | [`build-small-hackathon`](https://huggingface.co/build-small-hackathon) | Public Spaces are crawled as the live field for the atlas, search, advisor citations, and quest coverage. |
| Project snapshot | [`data/projects.json`](https://huggingface.co/spaces/build-small-hackathon/hackathon-advisor/blob/main/data/projects.json) | Stores Space metadata, README text, declared models/datasets, tags, and declared app-file evidence. |
| Project embedding index | [`data/project_index.json`](https://huggingface.co/spaces/build-small-hackathon/hackathon-advisor/blob/main/data/project_index.json) | Stores normalized EmbeddingGemma vectors and retrieval metadata for map construction and advisor search. |
| Quest SFT dataset | [`build-small-hackathon/hackathon-advisor-quest-dataset`](https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-quest-dataset) | Trains the MiniCPM quest classifier from README/app-file prompts with source-attributed quest labels. |
| Codex session traces | [`build-small-hackathon/hackathon-advisor-codex-traces`](https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-codex-traces) | Publishes real Codex session logs for this project after selection, minimization, and OpenAI Privacy Filter redaction. |
| Advisor LoRA examples | `lora_dataset` and [`/api/lora-training-kit.zip`](https://build-small-hackathon-hackathon-advisor.hf.space/api/lora-training-kit.zip) | Regenerates chat JSONL examples, recipe metadata, and the adapter card from exact advisor sessions. |

## How Codex Was Used

[Codex](https://developers.openai.com/codex) served as the engineering partner for the project. It helped translate the
hackathon requirements into implementation slices, inspect the existing codebase, build the atlas refresh/storage/cache
path, add the dashboard search and quest-evidence UI, run local tests and browser checks, review deployed Space behavior,
prepare commits and deployment updates, and revise the README into a submission narrative. This was an evidence loop:
Codex could read the repository, operate the local or deployed app, inspect the result, and then revise the
implementation or presentation from what it observed. The live app runtime uses the models and data listed above; Codex
appears in the development record as the assistant that helped design, implement, validate, and document the system.

The demo video used the same agentic workflow. Codex helped draft the narrated storyboard, drive the live app, capture
real screen footage, generate voice-over, compose the final video, and check the artifact by reading frames and ASR
transcripts. That media workflow ran outside Hackathon Advisor's submitted runtime and is not counted in the model
budget, but it documents how Codex was used to produce and verify submission materials rather than only code changes.

The redacted session-level Codex traces are published as a Hugging Face dataset at
[`build-small-hackathon/hackathon-advisor-codex-traces`](https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-codex-traces).

The full development history is public at <https://github.com/JacobLinCool/hackathon-advisor>.

## Prize Evidence

This submission targets the **Thousand Token Wood** main track, plus the OpenBMB, OpenAI/Codex, NVIDIA, and Modal
sponsor awards and the six bonus-quest badges.

| Prize path | Implemented evidence |
| --- | --- |
| Thousand Token Wood | The Almanac and Idea Map make the AI output visible as a playful, evidence-grounded exploration surface; the embedding index and the MiniCPM tool loop are load-bearing for the whitespace and originality experience. |
| Off the Grid | Every model runs from open weights on the Space's own GPU/CPU (or a local box); no third-party inference API is called at runtime, and retrieval vectors are local and embedded through llama.cpp. |
| Well-Tuned | Two MiniCPM5-1B PEFT LoRA adapters (advisor + quest classifier) are published publicly on the Hub; the local quest adapter is byte-identical to its published repo, and the training kit reproduces them. |
| Off-Brand | The custom `gradio.Server` frontend ships a bespoke atlas and Almanac experience, with no default Gradio UI in the runtime path. |
| Llama Champion | EmbeddingGemma GGUF vectors and every runtime query embedding run through llama.cpp; the index validator rejects any non-llama.cpp runtime. |
| Sharing is Caring | Real Codex session logs for this project are published on the Hub at [`build-small-hackathon/hackathon-advisor-codex-traces`](https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-codex-traces); the publisher selects project-relevant sessions, minimizes internal metadata, applies [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter), and records source hashes for audit. |
| Field Notes | A build report on the quest-classifier fine-tune is published at [`docs/quest-classification-lora.md`](docs/quest-classification-lora.md), and the app exports session Field Notes as markdown. |
| Tiny Titan | The largest single model is MiniCPM5-1B at ~1.08B — well under the 4B Tiny Titan ceiling; the full runtime stack totals ≈1.98B, far under the 32B cap. |
| Best Demo | The YouTube demo and public X post are linked from this README, and a backup MP4 is committed at [`static/assets/hackathon-advisor-demo.mp4`](static/assets/hackathon-advisor-demo.mp4). The video uses real app footage and a Codex-assisted production loop for storyboard, app driving, screen capture, voice-over generation, final composition, and frame/ASR verification. |
| OpenBMB | MiniCPM5-1B is the central language model for both tool planning and quest classification. |
| OpenAI Codex | Codex acted as an engineering partner across implementation, debugging, documentation, deployment review, demo production, and submission preparation. The public GitHub and Space histories use `Co-authored-by: Codex <codex@openai.com>` trailers, and selected project-facing session traces are published after minimization and privacy-filter redaction. |
| NVIDIA Nemotron | Voice input runs `nvidia/nemotron-speech-streaming-en-0.6b` through NVIDIA NeMo. |
| Modal | Modal trains the quest-classifier LoRA (`scripts/modal_train_quest_lora.py`), and a Modal remote index-build path is provided; the index shipped in this repo was built locally. |
| Best Agent | Each turn MiniCPM5 selects one tool; the engine then orchestrates the search → whitespace → score → plan chain over the live project field. |

## Run Locally

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdir -p .cache/advisor-dashboard
ADVISOR_CACHE_DIR=.cache/advisor-dashboard \
ADVISOR_MODEL_BACKEND=minicpm-transformers \
ADVISOR_MODEL_ID=openbmb/MiniCPM5-1B \
ADVISOR_ADAPTER_ID=build-small-hackathon/hackathon-advisor-minicpm5-lora \
ADVISOR_ADAPTER_REVISION=25de69bcde397e1bcdd852923b56a42f10222650 \
ADVISOR_QUEST_ANALYZER_BACKEND=minicpm-transformers \
python app.py
```

Then open <http://127.0.0.1:7860>. The atlas refresh button runs locally with the same artifact swap path used in
deployment. It writes refreshed runs under `.cache/advisor-dashboard/runs/` and atomically updates
`.cache/advisor-dashboard/latest.json`.

## Refresh The Project Snapshot

```bash
python scripts/crawl_hf_spaces.py --org build-small-hackathon --out data/projects.json
python scripts/build_project_index.py --location modal --projects data/projects.json --out data/project_index.json
```

The checked-in development snapshot lives in `data/projects.json` and `data/project_index.json`. A configured
`ADVISOR_CACHE_DIR` supplies the latest validated dashboard artifacts.

## Publish Codex Trace Dataset

Local privacy-filter run:

```bash
uv run --with 'transformers>=5.6,<6' --with 'torch>=2.8,<3' \
  python scripts/publish_codex_trace_dataset.py \
  --project-root . \
  --repo-id build-small-hackathon/hackathon-advisor-codex-traces \
  --verbose
```

Faster Modal GPU run:

```bash
python scripts/publish_codex_trace_dataset.py --location modal \
  --project-root . \
  --repo-id build-small-hackathon/hackathon-advisor-codex-traces
```

The publisher scans `~/.codex/sessions` and `~/.codex/archived_sessions`, selects sessions that mention this project,
keeps project-facing Codex events, removes system/developer prompts and compaction internals, normalizes local paths,
caps long tool-output text with truncation counts in the manifest, applies OpenAI Privacy Filter to the published log
text, writes `codex_sessions.jsonl` and `dataset_manifest.json`, then uploads the filtered data to the configured
Hugging Face dataset. The Modal wrapper uploads the selected raw JSONL files to a private Modal Volume, runs the same
publisher core on a GPU, returns the filtered dataset to local disk, and performs the Hugging Face upload from local
credentials.

## API And Artifacts

| Surface | Purpose |
| --- | --- |
| `GET /api/dashboard` | Atlas points, links, clusters, quest report, provenance, and refresh status. |
| `GET /api/dashboard/search?q=...` | BM25 search over project, cluster, quest, README, and app-file text. |
| `POST /api/dashboard/chat` | Atlas chat turn (NDJSON stream): base-model tool call, verified result + map action, grounded answer. |
| `POST /api/dashboard/refresh` | Starts one background refresh job. |
| `GET /api/dashboard/refresh` | Reports refresh stage, result, and status. |
| `POST /api/transcribe` | Transcribes uploaded voice notes with NVIDIA NeMo and Nemotron ASR. |
| `GET /api/prize-ledger` | Model stack, parameter budget, runtime status, and prize evidence. |
| `GET /api/demo-bundle.zip` | Demo session JSON, prize ledger, trace, notes, chapter, LoRA files, submission packet, and PNG. |
| `GET /api/lora-training-kit.zip` | SFT data, recipe, adapter card, and training command. |

The Gradio API also exposes `trace_artifact`, `field_notes`, `chapter`, `lora_dataset`, and `submission_packet` for
submission evidence and reviewer inspection.

## Advisor Workspace

The advisor workspace preserves the working loop from the original app. `Ink` compares the current idea against the
project index, `Gap` rotates through unused whitespace candidates, `Plan` drafts a practical build path, and `Compare`
rescans the saved idea board to select the strongest page. The `Profile` panel adds skills, time, preferences, and
constraints to the plan so the output can reflect "one evening", "frontend prototyping", or "CPU-only Space" as real
scoping facts.

Each scored page includes a deterministic `wood_map`: background dots for indexed Spaces, red dots for closest cited
echoes, and a green/red point for the current idea. The live UI and PNG export use the same Pillow renderer.

## Runtime Backend

The deployed Space is configured for ZeroGPU inference with:

```bash
ADVISOR_ZERO_GPU=1
ADVISOR_ZERO_GPU_DURATION=120
ADVISOR_MODEL_BACKEND=minicpm-transformers
ADVISOR_MODEL_ID=openbmb/MiniCPM5-1B
ADVISOR_ADAPTER_ID=build-small-hackathon/hackathon-advisor-minicpm5-lora
ADVISOR_ADAPTER_REVISION=25de69bcde397e1bcdd852923b56a42f10222650
ADVISOR_QUEST_ANALYZER_BACKEND=minicpm-transformers
ADVISOR_QUEST_ADAPTER_ID=build-small-hackathon/hackathon-advisor-quest-minicpm5-lora
ADVISOR_QUEST_ANALYSIS_BATCH_SIZE=8
ADVISOR_CACHE_DIR=/data/advisor-cache
ADVISOR_DISABLE_SCHEDULED_REFRESH=1
ADVISOR_REFRESH_COMPUTE=cpu
ADVISOR_SCHEDULED_REFRESH_COMPUTE=cpu
ADVISOR_REFRESH_INTERVAL_SECONDS=3600
ADVISOR_REFRESH_INITIAL_DELAY_SECONDS=300
ADVISOR_REFRESH_LOCK_TTL_SECONDS=7200
ADVISOR_REFRESH_EMBEDDING_TIMEOUT_SECONDS=1800
ADVISOR_EMBEDDING_MODEL_REPO=ggml-org/embeddinggemma-300m-qat-q8_0-GGUF
ADVISOR_EMBEDDING_MODEL_FILE=embeddinggemma-300m-qat-Q8_0.gguf
ADVISOR_EMBEDDING_N_CTX=2048
ADVISOR_ASR_MODEL_ID=nvidia/nemotron-speech-streaming-en-0.6b
```

The retrieval query embedder downloads the GGUF model through `huggingface_hub` unless
`ADVISOR_EMBEDDING_MODEL_PATH` points to a local file. `/api/transcribe` uses the same ZeroGPU wrapper for Nemotron ASR.
On macOS local runs, the app automatically runs llama.cpp query embedding in a worker process so the MiniCPM PyTorch
runtime and llama.cpp stay isolated from each other's OpenMP runtime. Dashboard refresh also builds the GGUF embedding
index in a subprocess before returning to the app process for MiniCPM quest analysis. When
`ADVISOR_CACHE_DIR` is set and `HF_HOME` is not, the refresh subprocess stores Hugging Face downloads under
`$ADVISOR_CACHE_DIR/huggingface` so the mounted bucket keeps the embedding model cache across refreshes and restarts.

## Test

```bash
pytest
```
