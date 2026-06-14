# Hackathon Advisor: A Living Field Guide for Build Small

Demo: <https://build-small-hackathon-hackathon-advisor.hf.space>  
Demo video: <https://youtu.be/Gq-FUiL-ZPw>  
Repository: <https://github.com/JacobLinCool/hackathon-advisor>  
Space: <https://huggingface.co/spaces/build-small-hackathon/hackathon-advisor>

## The Problem

Build Small moves fast. Every builder is trying to make something local, useful, strange, or delightful under a strict
model budget, but the field changes hour by hour. A good idea can already be crowded. A quiet niche can be invisible.
Quest and sponsor fit can be buried in READMEs, model cards, and app files. The result is that builders spend too much
of their scarce time guessing where their project sits.

Hackathon Advisor turns the hackathon itself into the starting point. It is a live atlas of public
`build-small-hackathon` Spaces and a small-model originality coach that helps a builder ask: what exists, what is nearby,
what is still unwritten, and how could this idea become a focused submission?

## The Experience

The app opens on the Idea Map, not a chatbot. Each point is a public hackathon Space. Clusters show the shape of the
field; nearest-neighbor links show which projects echo each other. A builder can search for a theme, filter by quest,
open a project, and inspect evidence before asking the advisor for help.

The advisor side is called The Unwritten Almanac. It compares an idea against the current project atlas, cites nearby
projects, proposes whitespace, scores the idea, and drafts a build plan. The workspace can export field notes, an
Almanac chapter, a page PNG, a demo bundle, and LoRA training materials.

There is also an "Ask the atlas" drawer. It uses the base MiniCPM5-1B model's native tool calling to answer questions
about the dashboard. Verified tool results render before the prose, and map actions can highlight or filter the atlas.

## The Implementation

The visible Space is a Gradio `gradio.Server`, which is a FastAPI server with Gradio API endpoints. The frontend is a
custom HTML/CSS/JS interface served from `static/`; the engine in `hackathon_advisor/` stays UI-agnostic.

The runtime model stack is fully open-weight and local to the Space:

- Advisor planning: `openbmb/MiniCPM5-1B` plus a public advisor LoRA.
- Quest classification: `openbmb/MiniCPM5-1B` plus a public quest-classifier LoRA.
- Retrieval: `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF` through llama.cpp.
- Voice input: `nvidia/nemotron-speech-streaming-en-0.6b` through NVIDIA NeMo ASR.

The advisor deliberately keeps the 1B model's job small. MiniCPM chooses one tool call per turn. Python then handles the
deterministic orchestration: search, whitespace, scoring, planning, profile handling, exports, and response templates.
This keeps behavior inspectable and avoids asking a tiny model to run an uncontrolled multi-hop ReAct loop.

The atlas refresh path crawls public Spaces in the hackathon organization, reads each README and declared app file,
rebuilds the llama.cpp embedding index, runs quest analysis, validates the dashboard payload, and swaps the result
atomically into the mounted Space cache. A failed refresh leaves the last validated dashboard in place.

## What Was Hard

The hardest part was not drawing a map. It was making the map trustworthy enough for judges and builders.

MiniCPM and llama.cpp can clash on OpenMP when loaded into the same hot path, so query embedding on macOS runs in a
worker subprocess and dashboard refresh builds the GGUF index in a subprocess before returning to MiniCPM quest analysis.

Quest classification also needed discipline. A prompt-only classifier would rename quests, emit extra prose, or award
Off the Grid to projects that used remote inference APIs. We built a supervised dataset from real public Spaces,
fine-tuned a MiniCPM LoRA, and kept schema validation plus hard invariants in the refresh path.

Finally, the Space needed to be useful even when GPU quota is tight. The atlas, search, exports, cached dashboard, and
many inspection routes remain available without loading the heavy models.

## How Codex Helped

Codex acted as the engineering partner across the project. It helped inspect the codebase, break the work into
implementation slices, build the dashboard storage and search paths, add quest evidence UI, write and adjust tests,
debug deployed Space behavior, prepare commit history, publish redacted build traces, and turn the project into a
submission-ready story.

The project also uses Codex as evidence. Redacted Codex session traces are published at
<https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-codex-traces>, and commits in the public GitHub
repo include Codex co-author trailers.

## Why It Fits Build Small

Hackathon Advisor is small in the way the hackathon asks for: every model is under 4B, the full stack is far below the
32B cap, and the runtime uses open weights instead of proprietary inference APIs. It is also complete: a public Space,
a public repo, a demo video, a model/data trail, a custom interface, and a clear use case.

The project belongs in Thousand Token Wood because it turns a hackathon into a navigable, AI-native landscape. The map,
Almanac, quest evidence, and exports make the field feel alive while still grounding every suggestion in real public
projects.
