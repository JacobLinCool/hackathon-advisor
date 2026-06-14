# Hackathon Advisor: A Living Field Guide for Hackathon Builders

Watch the demo video: <https://youtu.be/Gq-FUiL-ZPw>

Live app: <https://build-small-hackathon-hackathon-advisor.hf.space>  
Repository: <https://github.com/JacobLinCool/hackathon-advisor>  
Space: <https://huggingface.co/spaces/build-small-hackathon/hackathon-advisor>

## Overview

Hackathon Advisor is a live atlas of the Build Small Hackathon and a small-model originality coach for builders. It turns the public `build-small-hackathon` organization into an evidence surface: every public Space contributes to a map of the field, a searchable project index, quest-evidence summaries, and an advisor that helps a builder test where a new idea may still have room to grow.

The project addresses a practical problem in compressed creative work. During a hackathon, builders need to understand the surrounding field quickly. They need to know which ideas are crowded, which themes are emerging, which quests a project might satisfy, and how a proposal can become more specific before time runs out. Hackathon Advisor makes those questions answerable from the public work already being built around them.

The motivation starts with the data itself. Build Small already produces a ready-made corpus: public Spaces, READMEs, model declarations, app files, tags, and demos. That corpus invites exploration. Once it is made searchable and comparable, participants can see what others are building, which methods they are using, and what kinds of results are emerging across the community.

## Contribution

The central claim of the project is simple: originality improves when builders can see the field they are entering. The app presents that field first. A full-screen Idea Map places projects by embedding similarity, draws nearest-neighbor links, and exposes clusters that would be difficult to infer from a feed of individual Spaces. Search and filters make the same evidence usable for targeted questions, such as "voice assistants", "local-first", or "quest classifier".

This kind of visibility gives an online hackathon some of the generative character of OpenAI's [Parameter Golf](https://github.com/openai/parameter-golf) challenge: the event becomes more than a submission queue. It becomes a shared surface where participants can discover adjacent work, recognize partial overlap, borrow useful patterns, and extend ideas into new domains. A project similar to one's own can become evidence of a nearby direction; a project in another domain can suggest a feature, interface, or evaluation strategy that would otherwise remain invisible.

The advisor, called The Unwritten Almanac, uses the same project snapshot to compare a proposed idea against nearby work. It cites overlapping projects, identifies whitespace, scores the idea with a deterministic rubric, and drafts a build plan. The output is meant to be useful before implementation begins: it helps a builder sharpen the audience, choose a tractable scope, and reduce accidental duplication.

The broader value is collective intelligence. When participants can find related work quickly, similar teams can learn from each other's approaches and the community can combine ideas while the event is still unfolding. The atlas helps good ideas become easier to find, easier to extend, and easier to improve.

## User Experience

The experience begins with exploration. A builder can search the atlas, inspect a cluster, open a project card, and see the evidence behind detected quest matches. The map is intentionally the first surface because it gives the advisor's later recommendations visible context.

When the builder opens The Unwritten Almanac, the app shifts from field reading to idea development. The workspace keeps an idea board, profile constraints, score seals, whitespace candidates, and export actions in one place. A session can produce field notes, an Almanac chapter, a shareable PNG, a demo bundle, and LoRA training materials. These artifacts become reusable records for review and submission.

The "Ask the atlas" drawer adds a second mode of interaction. It lets a builder ask structured questions about the current dashboard. Verified repository results appear before the model-written answer, and map actions can highlight or filter projects directly. The model's prose is grounded in a compact digest of the verified result.

## Technical Design

Hackathon Advisor is deployed as a Gradio `gradio.Server`, a FastAPI application with Gradio API endpoints. The visible interface is a custom HTML, CSS, and JavaScript frontend served from `static/`; the engine in `hackathon_advisor/` remains UI-agnostic.

The runtime model stack is open-weight and local to the Space:

- Advisor planning: [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) with the public [`build-small-hackathon/hackathon-advisor-minicpm5-lora`](https://huggingface.co/build-small-hackathon/hackathon-advisor-minicpm5-lora) adapter.
- Quest classification: official Space metadata first, then [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) with the public [`build-small-hackathon/hackathon-advisor-quest-minicpm5-lora`](https://huggingface.co/build-small-hackathon/hackathon-advisor-quest-minicpm5-lora) adapter for evidence not declared by tags.
- Retrieval: [`ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF) through llama.cpp.
- Voice input: [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) through NVIDIA NeMo ASR.

The advisor gives the 1B model a narrow, inspectable role. MiniCPM selects one advisor action per turn; Python then carries out the deterministic sequence over search, whitespace discovery, scoring, planning, profile constraints, and exports. This design keeps the user-facing response tied to retrieved evidence while preserving the small-model discipline of the hackathon.

The atlas refresh method crawls public Spaces in the hackathon organization, reads each README and declared app file, builds a llama.cpp embedding index, resolves official `track:*`, `sponsor:*`, and `achievement:*` tags before invoking the quest LoRA for remaining evidence, validates the dashboard payload, and atomically swaps the new snapshot into the mounted Space cache. The last validated atlas remains available when a refresh fails, and many inspection routes remain usable while heavier models are unloaded.

## Validation Challenges

The main engineering challenge was trust. A map is useful only when builders and judges can understand where its signals come from. The refresh process therefore preserves project metadata, README evidence, app-file evidence, embedding provenance, quest-analysis outputs, and manifest data for each validated snapshot.

Quest classification required additional care. Early prompt-only runs could rename quests, emit explanatory prose, or misread local-inference criteria. The final classifier is a supervised MiniCPM LoRA trained on real public Spaces, with a strict JSON schema and invariant checks behind every refresh. The write-up in [`docs/quest-classification-lora.md`](docs/quest-classification-lora.md) describes the dataset and validation path.

The runtime also had to separate MiniCPM's PyTorch stack from llama.cpp on systems where OpenMP runtimes conflict. Query embedding on macOS runs in a worker subprocess, and dashboard refresh builds the GGUF index in a subprocess before returning to MiniCPM quest analysis.

## Codex Development

Codex acted as an engineering collaborator throughout the build. It helped inspect the codebase, turn requirements into implementation slices, add the dashboard storage and search paths, build the quest-evidence UI, run tests, review deployed Space behavior, prepare the demo materials, and refine the submission documents. The workflow was not only "ask for code, receive code"; it became a loop where Codex could read the repository, operate the app, inspect the result, and then revise the implementation or presentation from evidence.

The demo video was part of that loop. I used Codex with the [`studio-demo-video`](https://github.com/JacobLinCool/video-studio-mlx/tree/main/.agents/skills/studio-demo-video) skill to write a narrated storyboard, drive the live app, capture real screen footage, generate a voice-over, compose the final video, and check the result by reading extracted frames and ASR transcripts. That process used Codex's computer-use and vision abilities as practical production tools: it could operate UI surfaces, inspect whether focus rings and captions landed correctly, and verify that the spoken lines matched the intended scenes.

The local media backend behind this workflow was my in-progress Swift/MLX project, [Video Studio MLX](https://github.com/JacobLinCool/video-studio-mlx). That workflow points to a related lesson from the build: small local models are not only useful inside the submitted app; they can also support the way an agent builds, documents, and checks the app. I do not count this MLX tooling as part of Hackathon Advisor's hackathon runtime or model budget because it is not built with Gradio. It is still worth mentioning because it shows a possible agentic production path: write the script, use the product, generate narration, edit the video, and verify the artifact with simple visual and speech checks.

The project also preserves Codex's contribution as evidence. The public Git history includes Codex co-author trailers, and redacted Codex session traces are published at <https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-codex-traces>.
