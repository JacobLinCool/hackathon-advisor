# Hackathon Advisor: A Living Field Guide for Build Small

Watch the demo video: <https://youtu.be/Gq-FUiL-ZPw>

Live app: <https://build-small-hackathon-hackathon-advisor.hf.space>  
Repository: <https://github.com/JacobLinCool/hackathon-advisor>  
Space: <https://huggingface.co/spaces/build-small-hackathon/hackathon-advisor>

## Overview

Hackathon Advisor is a live atlas of the Build Small Hackathon and a small-model originality coach for builders. It
turns the public `build-small-hackathon` organization into an evidence surface: every public Space contributes to a map
of the field, a searchable project index, quest-evidence summaries, and an advisor that helps a builder test where a new
idea may still have room to grow.

The project addresses a practical problem in compressed creative work. During a hackathon, builders need to understand
the surrounding field quickly. They need to know which ideas are crowded, which themes are emerging, which quests a
project might satisfy, and how a proposal can become more specific before time runs out. Hackathon Advisor makes those
questions answerable from the public work already being built around them.

## Contribution

The central claim of the project is simple: originality improves when builders can see the field they are entering. The
app presents that field first. A full-screen Idea Map places projects by embedding similarity, draws nearest-neighbor
links, and exposes clusters that would be difficult to infer from a feed of individual Spaces. Search and filters make
the same evidence usable for targeted questions, such as "voice assistants", "local-first", or "quest classifier".

The advisor, called The Unwritten Almanac, uses the same project snapshot to compare a proposed idea against nearby
work. It cites overlapping projects, identifies whitespace, scores the idea with a deterministic rubric, and drafts a
build plan. The output is meant to be useful before implementation begins: it helps a builder sharpen the audience,
choose a tractable scope, and reduce accidental duplication.

## User Experience

The experience begins with exploration. A builder can search the atlas, inspect a cluster, open a project card, and see
the evidence behind detected quest matches. The map is intentionally the first surface because it gives the advisor's
later recommendations visible context.

When the builder opens The Unwritten Almanac, the app shifts from field reading to idea development. The workspace keeps
an idea board, profile constraints, score seals, whitespace candidates, and export actions in one place. A session can
produce field notes, an Almanac chapter, a shareable PNG, a demo bundle, and LoRA training materials. These artifacts
become reusable records for review and submission.

The "Ask the atlas" drawer adds a second mode of interaction. It lets a builder ask structured questions about the
current dashboard. Verified repository results appear before the model-written answer, and map actions can highlight or
filter projects directly. The model's prose is grounded in a compact digest of the verified result.

## Technical Design

Hackathon Advisor is deployed as a Gradio `gradio.Server`, a FastAPI application with Gradio API endpoints. The visible
interface is a custom HTML, CSS, and JavaScript frontend served from `static/`; the engine in `hackathon_advisor/`
remains UI-agnostic.

The runtime model stack is open-weight and local to the Space:

- Advisor planning: `openbmb/MiniCPM5-1B` with a public advisor LoRA.
- Quest classification: `openbmb/MiniCPM5-1B` with a public quest-classifier LoRA.
- Retrieval: `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF` through llama.cpp.
- Voice input: `nvidia/nemotron-speech-streaming-en-0.6b` through NVIDIA NeMo ASR.

The advisor gives the 1B model a narrow, inspectable role. MiniCPM selects one advisor action per turn; Python then
carries out the deterministic sequence over search, whitespace discovery, scoring, planning, profile constraints, and
exports. This design keeps the user-facing response tied to retrieved evidence while preserving the small-model
discipline of the hackathon.

The atlas refresh method crawls public Spaces in the hackathon organization, reads each README and declared app file,
builds a llama.cpp embedding index, runs quest analysis, validates the dashboard payload, and atomically swaps the new
snapshot into the mounted Space cache. The last validated atlas remains available when a refresh fails, and many
inspection routes remain usable while heavier models are unloaded.

## Validation Challenges

The main engineering challenge was trust. A map is useful only when builders and judges can understand where its signals
come from. The refresh process therefore preserves project metadata, README evidence, app-file evidence, embedding
provenance, quest-analysis outputs, and manifest data for each validated snapshot.

Quest classification required additional care. Early prompt-only runs could rename quests, emit explanatory prose, or
misread local-inference criteria. The final classifier is a supervised MiniCPM LoRA trained on real public Spaces, with
a strict JSON schema and invariant checks behind every refresh. The write-up in
[`docs/quest-classification-lora.md`](docs/quest-classification-lora.md) describes the dataset and validation path.

The runtime also had to separate MiniCPM's PyTorch stack from llama.cpp on systems where OpenMP runtimes conflict. Query
embedding on macOS runs in a worker subprocess, and dashboard refresh builds the GGUF index in a subprocess before
returning to MiniCPM quest analysis.

## Codex Development Record

Codex acted as an engineering collaborator throughout the build. It helped inspect the codebase, turn requirements into
implementation slices, add the dashboard storage and search paths, build the quest-evidence UI, run tests, review
deployed Space behavior, prepare the demo materials, and refine the submission documents.

The project also preserves Codex's contribution as evidence. The public Git history includes Codex co-author trailers,
and redacted Codex session traces are published at
<https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-codex-traces>.

## Fit For Build Small

Hackathon Advisor fits the Build Small constraints through its model budget and its product form. Every runtime model is
under 4B parameters, the full stack is far below the 32B cap, and inference runs from open weights inside the Space
process. MiniCPM is central to the advisor and quest classifier, llama.cpp powers retrieval, Nemotron supports voice
input, Modal supports development compute, and Codex is part of the documented build record.

The project is submitted for Thousand Token Wood because it makes the hackathon field navigable as an AI-native
landscape. The atlas, Almanac, quest evidence, and exports give builders a way to see the surrounding work and produce a
more deliberate idea from that evidence.
