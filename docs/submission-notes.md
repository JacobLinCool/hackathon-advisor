# Hackathon Submission Notes

Prepared on 2026-06-14 for the Build Small Hackathon submission deadline of 2026-06-15 23:59 UTC.

Official guide: <https://build-small-hackathon-field-guide.hf.space/submit>  
Live app: <https://build-small-hackathon-hackathon-advisor.hf.space>  
Space: <https://huggingface.co/spaces/build-small-hackathon/hackathon-advisor>  
GitHub: <https://github.com/JacobLinCool/hackathon-advisor>  
Demo video: <https://build-small-hackathon-hackathon-advisor.hf.space/static/assets/hackathon-advisor-demo.mp4>

## Official Checklist

| Requirement | Status | Evidence / action |
| --- | --- | --- |
| Stay under 32B | Ready | All runtime models are below 4B; the full stack is documented at `/api/prize-ledger` and in `README.md`. |
| Ship a Gradio app in the official org | Ready | Public Gradio Space: `build-small-hackathon/hackathon-advisor`; `/health` returns 200. |
| Record a demo | Ready | Final demo video is committed at `static/assets/hackathon-advisor-demo.mp4` and served by the Space. |
| Post it | Needs user action | Draft is in `docs/social-post.md`; publish it manually and replace the README draft link with the public post URL. |
| Mind the GPU limit | Ready | Uses one ZeroGPU Space (`zero-a10g`) for this submission. |
| Tag README | Ready | README front matter uses official `track:*`, `sponsor:*`, and `achievement:*` tags. |
| Short write-up of idea and tech | Ready | README plus `docs/article.md` describe the problem, experience, architecture, quests, and Codex usage. |

## Target Track

| Track | Eligible? | Rationale |
| --- | --- | --- |
| Backyard AI | Not primary | The app is useful to hackathon builders, but it is not framed around one everyday personal task for a named nearby user. |
| Thousand Token Wood | Yes | The project is an AI-native field guide: map, almanac, quest evidence, grounded originality coaching, and shareable artifacts. |

README tag: `track:wood`.

## Sponsor Prizes

| Prize | Eligible? | Requirement | Evidence |
| --- | --- | --- | --- |
| Best MiniCPM Build | Yes | Build with MiniCPM models. | `openbmb/MiniCPM5-1B` is the advisor planner and quest-classifier base model. |
| Best Use of Codex | Yes | Codex-attributed commits in connected GitHub repo or Space. | Public GitHub history includes Codex co-author trailers; README explains Codex role; trace dataset is published. |
| Nemotron Hardware Prize | Yes | Build with Nemotron models. | `/api/transcribe` uses `nvidia/nemotron-speech-streaming-en-0.6b` through NVIDIA NeMo ASR. |
| Best Use of Modal | Yes | Use Modal for development or runtime and note it in README. | Modal scripts train the quest LoRA and support remote index builds; README notes Modal development compute. |

README tags: `sponsor:openbmb`, `sponsor:openai`, `sponsor:nvidia`, `sponsor:modal`.

## Achievement Tags From The Submit Tool

| Achievement | Eligible? | Evidence |
| --- | --- | --- |
| Off the Grid | Yes | Runtime inference uses local open weights on the Space; no proprietary remote inference API is in the runtime path. |
| Well-Tuned | Yes | Advisor and quest-classifier MiniCPM LoRA adapters are public on Hugging Face. |
| Off-Brand | Yes | Custom `gradio.Server` frontend in `static/`, not stock Gradio UI. |
| Llama Champion | Yes | EmbeddingGemma GGUF retrieval runs through llama.cpp via `llama-cpp-python`. |
| Sharing is Caring | Yes | Redacted Codex traces are published on the Hub, and the app exports shareable notes/PNG/bundles. |
| Field Notes | Yes | `docs/quest-classification-lora.md`, `docs/blog-quest-lora.md`, `docs/article.md`, and field-note exports document the build. |

README tags: `achievement:offgrid`, `achievement:welltuned`, `achievement:offbrand`, `achievement:llama`,
`achievement:sharing`, `achievement:fieldnotes`.

## Other Prize Table Fit

| Prize / badge | Eligible? | Status |
| --- | --- | --- |
| Tiny Titan | Yes | Largest runtime model is MiniCPM5-1B at about 1.08B, well below 4B. |
| Best Agent | Yes | MiniCPM selects tools; Python orchestrates search, whitespace, scoring, plans, exports, and atlas actions. |
| Best Demo | Mostly ready | Demo video is ready; public social post still needs manual publication. |
| Bonus Quest Champion | Yes | The submission satisfies the six achievement tags and multiple sponsor/extra prize criteria. |
| Judges' Wildcard | Automatic | No separate entry required. |

## Deployment Verification

- `https://build-small-hackathon-hackathon-advisor.hf.space/health` returned HTTP 200 on 2026-06-14.
- Space API reports `sdk: gradio`, `private: false`, `runtime.stage: RUNNING`, `hardware.current: zero-a10g`.
- Runtime endpoint reports `backend: minicpm-transformers`, `model_id: openbmb/MiniCPM5-1B`, and `device: cuda`.
- Mounted cache/storage is configured at `/data`.

## Repository Hygiene

- GitHub repository is public: <https://github.com/JacobLinCool/hackathon-advisor>.
- `.env`, `.venv`, `.cache`, build artifacts, `node_modules`, local demo work files, and local frames are ignored.
- `.env.example` contains safe non-secret defaults.
- A secret-pattern scan found only placeholder/example strings and test fixtures, not real committed credentials.

## Manual Remaining Step

Publish one social post using `docs/social-post.md`, then update the README `Social post draft` line with the public URL
before completing the official submission form.

## Submission Form Summary

Hackathon Advisor is a live atlas of the Build Small Hackathon and a small-model originality coach. It crawls public
Build Small Spaces into a searchable map, shows project clusters and quest evidence, lets builders ask the atlas what
exists, and opens The Unwritten Almanac to compare a new idea against nearby work, find whitespace, score originality,
and export shareable field notes.

The runtime is local/open-weight: MiniCPM5-1B plus LoRA for advisor tool planning, MiniCPM5-1B plus LoRA for quest
classification, EmbeddingGemma GGUF through llama.cpp for retrieval, and Nemotron Speech ASR through NVIDIA NeMo for
voice input. Every model is under 4B and the full stack is far below the 32B cap. Codex helped build, test, debug,
document, and prepare the submission, with Codex co-author attribution in the public Git history.
