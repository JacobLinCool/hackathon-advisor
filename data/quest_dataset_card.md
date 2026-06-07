---
license: apache-2.0
task_categories:
- text-classification
- text-generation
language:
- en
tags:
- hackathon-advisor
- quest-classification
- lora-sft
- minicpm5
pretty_name: Hackathon Advisor Quest Classification SFT
size_categories:
- n<1K
---

# Hackathon Advisor — Quest Classification SFT Dataset

Supervised fine-tuning data that teaches MiniCPM5-1B to classify a Build Small
Hackathon project against 13 judging dimensions from a two-segment README + app-file
prompt, emitting strict JSON with short, source-attributed evidence. Trains the LoRA at
[`build-small-hackathon/hackathon-advisor-quest-minicpm5-lora`](https://huggingface.co/build-small-hackathon/hackathon-advisor-quest-minicpm5-lora).

## Format (`quest_sft.jsonl`)

Chat-JSONL. The **first line** is a `lora_sft_manifest`; every following line is a
`lora_sft_example` with a `messages` list (system / user / assistant). The assistant
turn is exactly one JSON object:

```json
{"matches":[{"quest":"...","confidence":0.0,"evidence":"...","source":"readme|app_file"}]}
```

No markdown, no prose, no renamed quests; an empty `matches` list when no dimension has
clear evidence. The user turn splits the project into a `[README]` segment and an
`[APP_FILE]` segment so the model judges product description and implementation
evidence separately and attributes each match to its source.

## Quest dimensions (13)

Six merit badges (Off the Grid, Well-Tuned, Off-Brand, Llama Champion, Sharing is
Caring, Field Notes), two tracks (Backyard AI, Thousand Token Wood), and five
sponsor / special awards (OpenBMB, Nemotron, Modal, Tiny Titan, Best Agent).

## Examples: 156 (14 with empty matches)

| variant | count |
| --- | --- |
| natural | 108 |
| app_only | 16 |
| missing_app_file | 16 |
| noisy_metadata | 8 |
| contradiction | 6 |
| empty | 2 |

Positive examples per quest:

| quest | examples |
| --- | --- |
| Off the Grid | 87 |
| Off-Brand | 59 |
| Tiny Titan | 58 |
| Thousand Token Wood | 49 |
| Llama Champion | 35 |
| Backyard AI | 35 |
| Well-Tuned | 31 |
| OpenBMB | 26 |
| Sharing is Caring | 19 |
| Nemotron | 18 |
| Field Notes | 15 |
| Modal | 14 |
| Best Agent | 14 |

## Provenance

Built from the real public Spaces of the `build-small-hackathon` org: 125 crawled
projects → deduped + length-filtered to 108 content-rich ones → labelled by a
teacher-then-adversarial-verifier multi-agent workflow → plus targeted augmentations
(app-only, readme-only / missing app file, README↔app contradictions, empty matches,
noisy metadata). `labeled.json` holds the per-project verified labels. Examples are
derived from public hackathon submissions for research and hackathon use; each project
remains under its own Space license.
