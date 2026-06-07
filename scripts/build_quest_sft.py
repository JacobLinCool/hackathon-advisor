#!/usr/bin/env python3
"""Assemble the quest-classification SFT dataset from verified teacher labels.

Inputs:
  data/quest_labels/labeled.json   - verified matches per project (from the Workflow)
  data/quest_labels/in/<slug>.json - the exact README / APP_FILE segments shown to the labeller

Builds one natural example per project plus targeted augmentations so every case the
prompt must handle is represented: app-only signal, readme-only signal, a missing app
file, README/app contradictions, empty matches, and noisy metadata. Writes
data/quest_sft.jsonl (manifest + examples) and prints a coverage report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hackathon_advisor.quest_dataset import build_dataset_jsonl, build_example, parse_quest_dataset_jsonl
from hackathon_advisor.quest_taxonomy import normalize_match, render_quest_prompt

NO_README = "(no README description provided)"
NO_APP = "(no app file available)"
IN_DIR = ROOT / "data" / "quest_labels" / "in"


def load_input(slug: str) -> dict:
    return json.loads((IN_DIR / f"{slug}.json").read_text(encoding="utf-8"))


def prompt_for(meta: dict, readme: str, app: str) -> str:
    return render_quest_prompt(
        title=meta.get("title", ""),
        sdk=meta.get("sdk", ""),
        declared_models=meta.get("declared_models", []),
        tags=meta.get("tags", []),
        readme_segment=readme,
        app_file_name=meta.get("app_file", ""),
        app_file_segment=app,
    )


def example(meta: dict, readme: str, app: str, matches: list[dict], *, variant: str) -> dict:
    return build_example(
        prompt_for(meta, readme, app),
        [normalize_match(m) for m in matches],
        meta={"kind": "quest_classification", "project_id": meta.get("id", ""), "variant": variant},
    )


# --- synthetic README/app contradictions: README screams "local/offline" but the app
#     clearly calls a proprietary cloud API, so Off the Grid must NOT be awarded. ---
CONTRADICTIONS = [
    {
        "id": "synthetic/contradiction-1",
        "title": "PocketScribe — fully local notes",
        "declared_models": [],
        "tags": ["gradio"],
        "app_file": "app.py",
        "readme": "# PocketScribe\nPocketScribe is a 100% offline, fully local note-taking assistant. "
                  "No API keys, no cloud, runs entirely on your own laptop for total privacy.",
        "app": "import gradio as gr\nfrom openai import OpenAI\nclient = OpenAI()\n\n"
               "def summarize(note):\n    r = client.chat.completions.create(model='gpt-4o-mini', "
               "messages=[{'role':'user','content':note}])\n    return r.choices[0].message.content\n\n"
               "gr.Interface(summarize, 'text', 'text').launch()",
        "matches": [
            {"quest": "Backyard AI", "confidence": 0.55, "evidence": "personal note-taking assistant", "source": "readme"},
        ],
    },
    {
        "id": "synthetic/contradiction-2",
        "title": "HomeVet offline pet advisor",
        "declared_models": [],
        "tags": ["gradio", "pets"],
        "app_file": "app.py",
        "readme": "# HomeVet\nAn offline, local-first pet-care helper for my own dog. Works without the "
                  "internet and keeps everything on-device. Built for a real person: my family.",
        "app": "import gradio as gr\nimport anthropic\nclient = anthropic.Anthropic()\n\n"
               "def advise(symptom):\n    msg = client.messages.create(model='claude-3-5-sonnet-20241022', "
               "max_tokens=300, messages=[{'role':'user','content':symptom}])\n    return msg.content[0].text\n\n"
               "with gr.Blocks() as demo:\n    gr.Markdown('# HomeVet')\n    inp = gr.Textbox()\n    out = gr.Textbox()\n"
               "    gr.Button('Ask').click(advise, inp, out)\ndemo.launch()",
        "matches": [
            {"quest": "Backyard AI", "confidence": 0.7, "evidence": "pet-care helper for my own dog", "source": "readme"},
        ],
    },
    {
        "id": "synthetic/contradiction-3",
        "title": "GridFree storyteller",
        "declared_models": [],
        "tags": ["gradio", "story"],
        "app_file": "app.py",
        "readme": "# GridFree\nA delightful local, no-cloud bedtime-story generator. Runs off the grid, "
                  "no proprietary APIs, entirely on your machine.",
        "app": "import gradio as gr, requests, os\n\nAPI='https://api.openai.com/v1/chat/completions'\n"
               "def story(theme):\n    r=requests.post(API, headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},"
               " json={'model':'gpt-4o','messages':[{'role':'user','content':theme}]})\n    return r.json()\n\n"
               "gr.Interface(story,'text','text', css='.gradio-container{background:#102}').launch()",
        "matches": [
            {"quest": "Thousand Token Wood", "confidence": 0.6, "evidence": "bedtime-story generator", "source": "readme"},
            {"quest": "Off-Brand", "confidence": 0.5, "evidence": "custom css background styling", "source": "app_file"},
        ],
    },
    {
        "id": "synthetic/contradiction-4",
        "title": "LocalLlama claim vs Gemini app",
        "declared_models": [],
        "tags": ["gradio"],
        "app_file": "app.py",
        "readme": "# QuietDesk\nRuns llama.cpp locally with GGUF weights — completely offline, your data never leaves "
                  "the device. A calm local-first desktop assistant.",
        "app": "import gradio as gr\nimport google.generativeai as genai\ngenai.configure(api_key='...')\n"
               "model = genai.GenerativeModel('gemini-1.5-flash')\n\n"
               "def reply(q):\n    return model.generate_content(q).text\n\n"
               "gr.ChatInterface(reply).launch()",
        "matches": [],
    },
    {
        "id": "synthetic/contradiction-5",
        "title": "Edge claim, cohere app",
        "declared_models": ["CohereForAI/command-r"],
        "tags": ["gradio"],
        "app_file": "app.py",
        "readme": "# EdgeMind\nEdgeMind is an on-device, fully local agent. No external services. Includes a write-up of "
                  "every build decision in our field notes below.\n## Field Notes\nDay 1: chose a tiny model...",
        "app": "import gradio as gr, cohere\nco = cohere.Client('KEY')\n\n"
               "def run(q):\n    return co.chat(message=q, model='command-r').text\n\n"
               "gr.Interface(run,'text','text').launch()",
        "matches": [
            {"quest": "Field Notes", "confidence": 0.7, "evidence": "write-up of every build decision", "source": "readme"},
        ],
    },
    {
        "id": "synthetic/contradiction-6",
        "title": "README understates a clearly local app",
        "declared_models": ["openbmb/MiniCPM5-1B"],
        "tags": ["gradio"],
        "app_file": "app.py",
        "readme": "# Helper\nA small helper app. (No further description.)",
        "app": "import gradio as gr\nfrom llama_cpp import Llama\n"
               "llm = Llama.from_pretrained('openbmb/MiniCPM5-1B-GGUF', filename='*Q4_K_M.gguf')\n\n"
               "def chat(m):\n    return llm.create_chat_completion(messages=[{'role':'user','content':m}])\n\n"
               "gr.Interface(chat,'text','text').launch()",
        "matches": [
            {"quest": "Off the Grid", "confidence": 0.85, "evidence": "local llama_cpp GGUF inference", "source": "app_file"},
            {"quest": "Llama Champion", "confidence": 0.9, "evidence": "from llama_cpp import Llama", "source": "app_file"},
            {"quest": "OpenBMB", "confidence": 0.85, "evidence": "openbmb/MiniCPM5-1B-GGUF", "source": "app_file"},
            {"quest": "Tiny Titan", "confidence": 0.75, "evidence": "MiniCPM5-1B is ~1B params", "source": "app_file"},
        ],
    },
]

# A couple of fully-empty-signal samples beyond whatever empties occur naturally.
EMPTY_SAMPLES = [
    {
        "id": "synthetic/empty-1",
        "title": "My Build Small Hackathon",
        "declared_models": [],
        "tags": ["gradio", "region:us"],
        "app_file": "app.py",
        "readme": "Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference",
        "app": "import gradio as gr\n\ndef greet(name):\n    return 'Hello ' + name\n\n"
               "gr.Interface(fn=greet, inputs='text', outputs='text').launch()",
    },
    {
        "id": "synthetic/empty-2",
        "title": "todo",
        "declared_models": [],
        "tags": ["gradio"],
        "app_file": "",
        "readme": "todo",
        "app": NO_APP,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the quest SFT dataset.")
    parser.add_argument("--labels", default="data/quest_labels/labeled.json", type=Path)
    parser.add_argument("--out", default="data/quest_sft.jsonl", type=Path)
    parser.add_argument("--app-only", type=int, default=16)
    parser.add_argument("--readme-only", type=int, default=16)
    parser.add_argument("--noisy", type=int, default=8)
    args = parser.parse_args()

    labeled = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = labeled["results"] if isinstance(labeled, dict) else labeled
    examples: list[dict] = []
    counts: dict[str, int] = {}

    def add(ex: dict) -> None:
        examples.append(ex)
        counts[ex["variant"]] = counts.get(ex["variant"], 0) + 1

    # 1) natural example per labeled project
    by_slug = {}
    for row in rows:
        slug = row["slug"]
        meta = load_input(slug)
        matches = row.get("matches") or []
        by_slug[slug] = (meta, matches)
        add(example(meta, meta["README"], meta["APP_FILE"], matches, variant="natural"))

    # rank projects by richness of each source for augmentation selection
    app_rich = sorted(
        ((s, m, ms) for s, (m, ms) in by_slug.items() if any(x["source"] == "app_file" for x in ms)),
        key=lambda t: -sum(1 for x in t[2] if x["source"] == "app_file"),
    )
    readme_rich = sorted(
        ((s, m, ms) for s, (m, ms) in by_slug.items() if any(x["source"] == "readme" for x in ms)),
        key=lambda t: -sum(1 for x in t[2] if x["source"] == "readme"),
    )

    # 2) app-only: strip README, keep only app_file-sourced matches
    for slug, meta, ms in app_rich[: args.app_only]:
        kept = [m for m in ms if m["source"] == "app_file"]
        add(example(meta, NO_README, meta["APP_FILE"], kept, variant="app_only"))

    # 3) readme-only / missing app file: blank the app file, keep only readme-sourced matches
    for slug, meta, ms in readme_rich[: args.readme_only]:
        kept = [m for m in ms if m["source"] == "readme"]
        add(example(meta, meta["README"], NO_APP, kept, variant="missing_app_file"))

    # 4) noisy metadata: inject garbled tags + scrambled title, gold unchanged
    noisy_pool = sorted(
        ((s, m, ms) for s, (m, ms) in by_slug.items() if ms),
        key=lambda t: -len(t[2]),
    )
    for slug, meta, ms in noisy_pool[: args.noisy]:
        noisy_meta = dict(meta)
        noisy_meta["tags"] = list(meta.get("tags", [])) + ["asdf123", "xx", "region:us", "untitled", "draft"]
        noisy_meta["title"] = (meta.get("title", "") + " ::: TODO copy of template (do not read title)").strip()
        add(example(noisy_meta, meta["README"], meta["APP_FILE"], ms, variant="noisy_metadata"))

    # 5) synthetic contradictions
    for spec in CONTRADICTIONS:
        add(example(spec, spec["readme"], spec["app"], spec["matches"], variant="contradiction"))

    # 6) explicit empties
    for spec in EMPTY_SAMPLES:
        add(example(spec, spec["readme"], spec["app"], [], variant="empty"))

    text = build_dataset_jsonl(examples, source_note="build_small_hackathon real projects + targeted augmentations")
    manifest, parsed = parse_quest_dataset_jsonl(text)  # validates the whole file
    args.out.write_text(text, encoding="utf-8")

    print(f"wrote {len(parsed)} examples to {args.out}")
    print("variant counts:", json.dumps(counts, ensure_ascii=False))
    print("empty-match examples:", manifest["empty_match_examples"])
    print("quest positive counts:")
    for quest, n in sorted(manifest["quest_positive_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {quest}")


if __name__ == "__main__":
    main()
