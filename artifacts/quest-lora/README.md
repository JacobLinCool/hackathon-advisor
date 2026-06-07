---
base_model: openbmb/MiniCPM5-1B
library_name: peft
datasets:
- build-small-hackathon/hackathon-advisor-quest-dataset
tags:
- lora
- hackathon-advisor
- quest-classification
license: apache-2.0
---

# Hackathon Advisor — Quest Classification LoRA (MiniCPM5-1B)

PEFT LoRA adapter that classifies a Build Small Hackathon project against 13 judging
dimensions (6 merit badges + 2 tracks + 5 sponsor/special awards) from a two-segment
README + app-file prompt, emitting strict JSON:

```json
{"matches":[{"quest":"...","confidence":0.0,"evidence":"...","source":"readme|app_file"}]}
```

Load it in the deployed Space by setting `ADVISOR_QUEST_ADAPTER_ID` to this repo.
The backend revalidates every dashboard refresh and will not swap on schema failure.

## Recipe

- Base model: `openbmb/MiniCPM5-1B`
- Task: `hackathon_advisor_quest_classification`
- Method: LoRA SFT (completion-only loss)
- Examples: 146
- Epochs: 6.0
- LoRA rank/alpha/dropout: 16/32/0.05
- Max seq length: 2560
- GPU: A10G

## Dataset

[`build-small-hackathon/hackathon-advisor-quest-dataset`](https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-quest-dataset) — 156 chat-JSONL examples built from real `build-small-hackathon` Spaces: 108 teacher-
labelled + adversarially-verified projects plus targeted augmentations (app-only,
readme-only / missing app file, README↔app contradictions, empty matches, noisy
metadata). All 13 quests covered.

## Self-eval at training time: 10/10 held-out prompts produced schema-valid JSON.
