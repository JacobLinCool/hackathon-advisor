# Quest-classification LoRA

The dashboard refresh asks MiniCPM5-1B to classify every crawled hackathon project
against the Build Small Hackathon judging dimensions and to quote short evidence for
each match. A pure prompt drifts (truncated JSON, renamed quests, runaway evidence),
so we fine-tune a small LoRA that fixes the task to a strict JSON contract. The
backend still validates every refresh and refuses to swap the dashboard on a schema
failure; the checked-in adapter is the product default, and validation is the
correctness gate.

## Label space (`hackathon_advisor/quest_taxonomy.py`)

13 dimensions, each detectable from a README and an app file:

- Six merit badges: Off the Grid, Well-Tuned, Off-Brand, Llama Champion, Sharing is
  Caring, Field Notes.
- Two main tracks: Backyard AI, Thousand Token Wood.
- Sponsor / special awards: OpenBMB, Nemotron, Modal, Tiny Titan, Best Agent.

Output schema (one JSON object, nothing else):

```json
{"matches": [{"quest": "...", "confidence": 0.0, "evidence": "...", "source": "readme|app_file"}]}
```

`render_quest_prompt` is the single prompt renderer shared by the dataset and the
live analyzer, so the model sees the same two-segment shape (README + APP_FILE) at
train and inference time, with the same `QUEST_SYSTEM_PROMPT`.

## Dataset pipeline

1. `scripts/build_quest_corpus.py` — download the real README.md and main app-file
   source for all 125 crawled projects into `data/quest_corpus.json`.
2. Selection (`data/quest_selection.json`) — drop near-identical template clones
   (embedding cosine + identical app hash) and the shortest, signal-free tail;
   108 content-rich projects survive (app-only / readme-only / both profiles kept).
3. Teacher labelling — a multi-agent workflow labels each project (one agent) then
   adversarially verifies and corrects it (a second agent): drops matches whose
   evidence is not in the cited segment, fixes `source`, kills Off-the-Grid on a
   cloud-API app, kills Tiny Titan on >4B models. Output: `data/quest_labels/labeled.json`.
4. `scripts/build_quest_sft.py` — one natural example per project plus targeted
   augmentations so every case is represented:
   app-only, readme-only / missing app file, README↔app contradictions, empty
   matches, and noisy metadata. Writes `data/quest_sft.jsonl`
   (`hackathon_advisor/quest_dataset.py` formats and validates it).

156 chat-JSONL examples (108 natural + 48 augmented), 14 with empty matches, all 13
quests covered; ~93% of match evidence is literally present in its cited segment
(the rest is Off-the-Grid absence-of-cloud-API reasoning).

Published as a Hub dataset:
[`build-small-hackathon/hackathon-advisor-quest-dataset`](https://huggingface.co/datasets/build-small-hackathon/hackathon-advisor-quest-dataset)
(`scripts/publish_quest_dataset.py`). The trained adapter lives at
[`build-small-hackathon/hackathon-advisor-quest-minicpm5-lora`](https://huggingface.co/build-small-hackathon/hackathon-advisor-quest-minicpm5-lora).

## Training (`scripts/modal_train_quest_lora.py`)

```bash
modal run scripts/modal_train_quest_lora.py::smoke              # check the GPU
modal run scripts/modal_train_quest_lora.py --dataset data/quest_sft.jsonl --epochs 6
```

LoRA SFT on an A10G: rank 16, alpha 32, completion-only loss (the prompt is masked
to -100 so only the strict JSON is supervised), `max_seq_length=2560`, chat template
with `enable_thinking=False` to match inference. The container self-evaluates on a
held-out slice (does the adapter emit schema-valid JSON?) and returns the adapter as
a zip that the local entrypoint unpacks under `artifacts/quest-lora/`.

## Serving

`MiniCPMQuestAnalyzer` loads the checked-in `artifacts/quest-lora` adapter by
default. `ADVISOR_QUEST_ADAPTER_ID` and `ADVISOR_QUEST_ADAPTER_REVISION` may point
to a replacement adapter, while an explicit empty adapter id runs the base model
for controlled experiments. `validate_matches_by_project` enforces the schema
before the dashboard is swapped.
