# AGENTS.md

Operating manual for coding agents working in this repo.

---

## What this is

**Hackathon Advisor** is a Gradio `gradio.Server` (FastAPI subclass) Space for the
[Build Small Hackathon](https://huggingface.co/build-small-hackathon). It is a small-model (**≤32B**, largest single
model **≤4B**) originality coach: it crawls the public `build-small-hackathon` org into a live project atlas, then lets a
builder search the field and open **The Unwritten Almanac** advisor to test an idea against existing work.

The engine in `hackathon_advisor/` is **UI-agnostic**; `app.py` and `static/` are one possible front door.

**Model stack (all open-weight, all local):**

| Role | Model | Runtime |
| --- | --- | --- |
| Advisor brain (tool planning) | `openbmb/MiniCPM5-1B` + advisor LoRA | Transformers + PEFT, ZeroGPU |
| Quest classifier | `openbmb/MiniCPM5-1B` + quest LoRA | Transformers + PEFT, ZeroGPU |
| Retrieval / atlas | `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF` | llama.cpp (llama-cpp-python) |
| Voice input (ASR) | `nvidia/nemotron-speech-streaming-en-0.6b` | NVIDIA NeMo |

---

## Setup & commands

- **Python** `>=3.11,<3.13`. Dependency manager is **uv** (`uv.lock` is the source of truth).
- **System packages** (`packages.txt`): `ffmpeg`, `libsndfile1`.

```bash
uv sync                       # or: pip install -r requirements.txt
uv run pytest                 # run the test suite (fast, NO GPU/weights needed — heavy models are mocked)
uvx ruff check .              # lint   (config: pyproject.toml [tool.ruff], line-length 100, py311; ruff is not a pinned dep)
uvx ruff format .             # format
```

Run the app locally (greedy CPU/MPS path, no ZeroGPU):

```bash
mkdir -p .cache/advisor-dashboard
ADVISOR_CACHE_DIR=.cache/advisor-dashboard \
ADVISOR_MODEL_BACKEND=minicpm-transformers \
ADVISOR_QUEST_ANALYZER_BACKEND=minicpm-transformers \
python app.py                 # → http://127.0.0.1:7860
```

`ADVISOR_MODEL_BACKEND=rules` swaps the LLM for a deterministic planner — use it for UI/plumbing work without loading
MiniCPM.

`pytest` config lives in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["."]`). **Always run it before
committing** — there are 26 test files and they are the contract.

---

## Repo map

```
app.py                  gr.Server entry: static UI + FastAPI /api/* + @app.api() client endpoints + refresh scheduler
hackathon_advisor/      the engine package (UI-agnostic — keep it that way)
static/                 bespoke frontend (index.html / app.js / styles.css) — the Off-Brand custom UI
scripts/                offline pipelines (crawl, Modal index/LoRA build, Hub publish) — NOT runtime
data/                   checked-in snapshots: projects.json, project_index.json, sample_trace.jsonl, quest dataset
artifacts/quest-lora/   local quest-LoRA training output (gitignored; loaded from the Hub repo at runtime)
docs/                   build reports (e.g. quest-classification-lora.md)
tests/                  pytest suite (mirrors module names: test_<module>.py)
```

### Engine package (`hackathon_advisor/`)

| Module | Responsibility |
| --- | --- |
| `agent.py` | `AdvisorEngine.turn()` / `turn_stream()`. **One** LLM tool-pick per turn, then deterministic Python orchestration (`search → whitespace → score → plan`). Advisor prose is built from **f-string templates** here, not by the model. |
| `model_runtime.py` | `ToolPlanner` backends. `create_tool_planner()` selects via `ADVISOR_MODEL_BACKEND`: `minicpm-transformers` (MiniCPM5-1B + advisor LoRA, device ladder `auto/CUDA → MPS → CPU`) or `rules` (`RuleBasedPlanner`). |
| `tool_contracts.py` | `TOOL_SPECS` typed schema; `parse_xml_tool_call()`; `resolve_tool_call()` returns `valid` or a `defaulted` call (the tool-call **degradation ladder**). |
| `tools.py` | Tool implementations over `ProjectIndex` (search, whitespace, score, plan, profile, …). Heavy logic lives here, not in the model. |
| `aliases.py` | Jargon normalization (fuzzy-maps "neutron" → Nemotron, "mini cpm" → MiniCPM5, …) applied **before** tool routing. |
| `data.py` | `ProjectIndex`: loads the snapshot + embedding index, `_embed_query()` via llama.cpp, cosine search. |
| `llama_embedding.py` | `LlamaCppEmbedder` — EmbeddingGemma GGUF through llama-cpp-python (the Llama Champion path). |
| `dashboard.py` / `dashboard_storage.py` / `dashboard_search.py` | Atlas payload (t-SNE / KMeans / nearest links), BM25 search, and the refresh **lease + heartbeat + atomic `latest.json` swap**. |
| `quest_analysis.py` / `quest_taxonomy.py` / `quest_cache.py` | MiniCPM quest LoRA → strict quest JSON; the taxonomy; per-project cache keyed on prompt/taxonomy/model/adapter hashes. |
| `scoring.py` | Deterministic idea rubric (the model only triggers + verbalizes it). |
| `wood_map.py` / `png_export.py` | PCA projection + Pillow render of the shareable page PNG. |
| `field_notes.py` / `chapter.py` / `trace_export.py` / `submission_packet.py` / `artifact_bundle.py` / `demo_rehearsal.py` | Export surfaces (notes, chapter, agent trace, submission packet, demo bundle). |
| `prize_ledger.py` | Model stack + parameter budget + badge ledger reported at `/api/prize-ledger`. |
| `zerogpu.py` | `gpu_task()` decorator (no-op unless `ADVISOR_ZERO_GPU=1`) + GPU-quota error detection for the CPU fallback. |
| `runtime_hooks.py` / `profiling.py` | Process/runtime helpers and turn profiling. |

### Routes (`app.py`)

First-party FastAPI routes power the visible app; `@app.api()` endpoints stay available for Gradio/Python clients.

| Route | Purpose |
| --- | --- |
| `GET /` , `GET /static/{path}` | Serve the bespoke `static/` frontend |
| `POST /api/agent-turn` | The advisor turn — **NDJSON stream**; this is the `@spaces.GPU` boundary |
| `POST /api/transcribe` | Voice note → transcript (NeMo, see ASR gotcha) |
| `GET /api/dashboard` · `GET /api/dashboard/search` | Atlas payload · BM25 search |
| `POST/GET /api/dashboard/refresh` | Start / poll one background refresh job |
| `GET /api/bootstrap` · `GET /api/runtime` · `GET /api/prize-ledger` · `GET /api/tool-contracts` | Frontend bootstrap, runtime status, prize ledger, tool schema |
| `GET /api/demo-bundle.zip` · `GET /api/lora-training-kit.zip` · `POST /api/artifact.png` · `POST /api/field-notes` · `POST /api/chapter` | Exports |
| `GET /health` | Liveness |

---

## Gotchas (the things that bite agents here)

1. **The 1B model only emits ONE XML tool call per turn.** All user-facing prose is templated Python (`agent.py`
   `_*_response`), and multi-step flows are orchestrated in code — not a model-driven ReAct loop. Do **not** "make the
   model write the response" or add multi-hop tool loops; route through `tool_contracts.py` instead.
2. **Off the Grid is a hard constraint.** No proprietary cloud inference API may touch the runtime path. All three
   engines run locally from open weights. Don't add `InferenceClient`, `openai`, etc. to runtime code.
3. **Parameter budget.** Total ≤32B, largest single model ≤4B (Tiny Titan). Don't introduce a larger model;
   `prize_ledger.py` documents the ~1.98B stack.
4. **MiniCPM (PyTorch) and llama.cpp clash on OpenMP.** Query embedding runs in a **worker subprocess** on macOS, and
   dashboard refresh builds the GGUF index in a subprocess before returning to the MiniCPM process. Keep these isolated;
   don't import both heavy runtimes into the same hot path.
5. **Decoding is greedy.** `enable_thinking=False`, `temperature=0` for tool calls and strict quest JSON. Keep tool
   schemas small and single-hop (1B discipline).
6. **Never write `latest.json` directly.** Refreshes write `runs/{run_id}/…` then do an **atomic swap** under
   `$ADVISOR_CACHE_DIR/refresh.lock` with a heartbeat; a failed run leaves the last validated dashboard in place.
7. **Tests must stay GPU-free.** The suite mocks torch/transformers/llama.cpp — `pytest` runs with no GPU and no model
   weights. Don't add module-top heavy imports that break CPU-only test collection.
8. **ASR backend.** `asr_runtime.py` requires NVIDIA NeMo ASR for `nvidia/nemotron-speech-streaming-en-0.6b`; missing
   NeMo is a hard runtime error, locally and on the deployed Space. `status()` reports the configured Nemotron backend.

---

## Offline pipelines (`scripts/`, build-time only)

Runtime never calls these — they keep the Space self-contained.

```bash
python scripts/crawl_hf_spaces.py --org build-small-hackathon --out data/projects.json   # crawl the field
python scripts/build_project_index.py --projects data/projects.json --out data/project_index.json   # local llama.cpp index
python scripts/build_project_index.py --location modal ...   # same build, on Modal (one CLI, --location switches where it runs)
modal run scripts/modal_train_quest_lora.py ...           # train the quest LoRA on Modal
python scripts/publish_quest_adapter.py ... / publish_quest_dataset.py ...   # push adapter / dataset to the Hub
```

---

## Commits & reviews

- **Conventional commits**, one concern per commit. Observed history: `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`.
- **Gate before committing:** `uv run pytest` green, `uvx ruff check .` clean, and the README updated if behavior
  changed.
- Keep the engine package UI-agnostic; if you touch a runtime model path, re-check gotchas 2–4 (Off the Grid, param
  budget, OpenMP isolation).

---

## Key environment variables

| Variable | Default | Use |
| --- | --- | --- |
| `ADVISOR_CACHE_DIR` | — | Artifact store (mounted bucket on Spaces); enables the refresh scheduler when set |
| `ADVISOR_MODEL_BACKEND` | `minicpm-transformers` | Advisor planner: `minicpm-transformers` or `rules` |
| `ADVISOR_MODEL_ID` / `ADVISOR_ADAPTER_ID` / `ADVISOR_ADAPTER_REVISION` | MiniCPM5-1B + advisor LoRA | Advisor model + pinned LoRA |
| `ADVISOR_QUEST_ANALYZER_BACKEND` / `ADVISOR_QUEST_ADAPTER_ID` | `minicpm-transformers` / `build-small-hackathon/hackathon-advisor-quest-minicpm5-lora` | Quest classifier |
| `ADVISOR_ZERO_GPU` / `ADVISOR_ZERO_GPU_DURATION` | off / `120` | Wrap the engine turn in `@spaces.GPU` on the deployed Space |
| `ADVISOR_ASR_MODEL_ID` | Nemotron | Voice ASR model |
| `ADVISOR_EMBEDDING_MODEL_REPO` / `ADVISOR_EMBEDDING_MODEL_FILE` | EmbeddingGemma GGUF | llama.cpp retrieval model |
| `ADVISOR_REFRESH_COMPUTE` / `ADVISOR_REFRESH_INTERVAL_SECONDS` | `cpu` / `3600` | Scheduled refresh compute + cadence |

See `## Runtime Backend` in `README.md` for the full deployed configuration.
