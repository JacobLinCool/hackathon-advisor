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
import re
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


# Real projects (kept in the corpus) whose app calls a REMOTE inference endpoint.
# Their teacher labels already exclude Off the Grid; app-only variants force the model
# to judge the remote-inference app directly instead of leaning on its strong prior.
REMOTE_INFERENCE_SLUGS = [
    "GTROX", "ai-study-buddy", "come-and-compare", "AI-agent-Evaluation-pipeline",
    "Sprout-And-Spoon", "The-Shrine", "Backyard-Demo-Builder", "persona-atlas",
    "Structured-Data-Rescuer", "nutrilens", "ux-crime-scene", "wpl-discovery",
    "legawa", "business-order-assistant", "cloud-parade-cabinet", "gitopadesh",
]


# Hand-authored contrastive hard negatives for two observed failure modes:
#  (1) a REMOTE inference call (InferenceClient / endpoints / replicate / *.modal.run)
#      must NOT earn Off the Grid, whatever model it names;
#  (2) OpenBMB belongs only to openbmb/ models and Tiny Titan only to <=4B models,
#      so a non-openbmb / large model id must not trigger them. Positive anchors keep
#      the model from over-correcting on genuinely local openbmb / small models.
HARD_NEGATIVES = [
    {
        "id": "synthetic/remote-gptoss-empty",
        "title": "Chat Demo", "declared_models": [], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# Chat Demo\nA simple chat space.",
        "app": "import gradio as gr\nfrom huggingface_hub import InferenceClient\n"
               "client = InferenceClient(model=\"openai/gpt-oss-20b\")\n\n"
               "def respond(m, history):\n    return client.chat_completion(m).choices[0].message.content\n\n"
               "gr.ChatInterface(respond).launch()",
        "matches": [],
    },
    {
        "id": "synthetic/remote-qwen-offbrand",
        "title": "NeonChat", "declared_models": [], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# NeonChat\nA chat UI with a neon theme.",
        "app": "import gradio as gr\nfrom huggingface_hub import InferenceClient\n"
               "client = InferenceClient(model=\"Qwen/Qwen2.5-72B-Instruct\")\n"
               "CUSTOM_CSS = '.gradio-container{background:#0a0a14} .msg{box-shadow:0 0 12px #0ff}'\n\n"
               "def reply(m, h):\n    return client.chat_completion(m).choices[0].message.content\n\n"
               "demo = gr.Blocks(css=CUSTOM_CSS)\n",
        "matches": [
            {"quest": "Off-Brand", "confidence": 0.78, "evidence": "gr.Blocks(css=CUSTOM_CSS) neon custom styling", "source": "app_file"},
        ],
    },
    {
        "id": "synthetic/remote-endpoint-backyard",
        "title": "PillReader", "declared_models": [], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# PillReader\nHelps my grandmother read the small print on her medication labels and "
                  "set reminders, so she can manage her prescriptions without calling me every day.",
        "app": "import requests, gradio as gr\n"
               "ENDPOINT = \"https://abc123.endpoints.huggingface.cloud\"\n\n"
               "def read(image):\n    return requests.post(ENDPOINT, files={'image': image}).json()['text']\n\n"
               "gr.Interface(read, 'image', 'text').launch()",
        "matches": [
            {"quest": "Backyard AI", "confidence": 0.85, "evidence": "helps my grandmother read medication labels", "source": "readme"},
        ],
    },
    {
        "id": "synthetic/remote-replicate-ttw",
        "title": "DreamPostcards", "declared_models": [], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# DreamPostcards\nA whimsical generator that turns a sentence about your day into a "
                  "dreamy illustrated postcard from an imaginary seaside town.",
        "app": "import replicate, gradio as gr\n\n"
               "def make(prompt):\n    return replicate.run('black-forest-labs/flux-schnell', input={'prompt': prompt})\n\n"
               "gr.Interface(make, 'text', 'image').launch()",
        "matches": [
            {"quest": "Thousand Token Wood", "confidence": 0.8, "evidence": "dreamy illustrated postcard generator", "source": "readme"},
        ],
    },
    {
        "id": "synthetic/remote-together-empty",
        "title": "AskAnything", "declared_models": [], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# AskAnything\nAsk a question.",
        "app": "import gradio as gr\nfrom together import Together\nclient = Together()\n\n"
               "def ask(q):\n    return client.chat.completions.create(model='openai/gpt-oss-120b', "
               "messages=[{'role':'user','content':q}]).choices[0].message.content\n\n"
               "gr.Interface(ask, 'text', 'text').launch()",
        "matches": [],
    },
    {
        "id": "synthetic/remote-modalrun-modal",
        "title": "FastSummarizer", "declared_models": [], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# FastSummarizer\nSummarizes long text. The model is served on Modal.",
        "app": "import requests, gradio as gr\n"
               "MODAL_URL = \"https://myorg--summarizer-serve.modal.run\"\n\n"
               "def summarize(text):\n    return requests.post(MODAL_URL, json={'text': text}).json()['summary']\n\n"
               "gr.Interface(summarize, 'text', 'text').launch()",
        "matches": [
            {"quest": "Modal", "confidence": 0.85, "evidence": "model served at *.modal.run endpoint", "source": "app_file"},
        ],
    },
    {
        "id": "synthetic/remote-gradioclient-empty",
        "title": "Proxy Chat", "declared_models": [], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# Proxy Chat\nChat front-end.",
        "app": "import gradio as gr\nfrom gradio_client import Client\n"
               "client = Client(\"someorg/big-llm-space\")\n\n"
               "def chat(m):\n    return client.predict(m, api_name='/chat')\n\n"
               "gr.Interface(chat, 'text', 'text').launch()",
        "matches": [],
    },
    {
        "id": "synthetic/remote-openrouter-empty",
        "title": "RouterBot", "declared_models": [], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# RouterBot\nA chatbot.",
        "app": "import gradio as gr\nfrom openai import OpenAI\n"
               "client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key='...')\n\n"
               "def reply(m):\n    return client.chat.completions.create(model='meta-llama/llama-3.1-8b', "
               "messages=[{'role':'user','content':m}]).choices[0].message.content\n\n"
               "gr.Interface(reply, 'text', 'text').launch()",
        "matches": [],
    },
    {
        "id": "synthetic/local-gptoss20b",
        "title": "LocalGPTOSS", "declared_models": ["openai/gpt-oss-20b"], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# LocalGPTOSS\nRuns gpt-oss locally.",
        "app": "import gradio as gr\nimport torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
               "model = AutoModelForCausalLM.from_pretrained(\"openai/gpt-oss-20b\", torch_dtype='auto', device_map='cuda')\n"
               "tok = AutoTokenizer.from_pretrained(\"openai/gpt-oss-20b\")\n\n"
               "def gen(p):\n    ids = tok(p, return_tensors='pt').to('cuda')\n    return tok.decode(model.generate(**ids)[0])\n\n"
               "gr.Interface(gen, 'text', 'text').launch()",
        "matches": [
            {"quest": "Off the Grid", "confidence": 0.88, "evidence": "AutoModelForCausalLM.from_pretrained, in-process, no remote call", "source": "app_file"},
        ],
    },
    {
        "id": "synthetic/local-qwen7b",
        "title": "Qwen7B Helper", "declared_models": ["Qwen/Qwen2.5-7B-Instruct"], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# Qwen7B Helper\nA local assistant.",
        "app": "import gradio as gr\nfrom transformers import pipeline\n"
               "pipe = pipeline('text-generation', model=\"Qwen/Qwen2.5-7B-Instruct\", device_map='auto')\n\n"
               "def run(p):\n    return pipe(p)[0]['generated_text']\n\n"
               "gr.Interface(run, 'text', 'text').launch()",
        "matches": [
            {"quest": "Off the Grid", "confidence": 0.85, "evidence": "local transformers pipeline, no remote inference", "source": "app_file"},
        ],
    },
    {
        "id": "synthetic/local-llamacpp-qwen",
        "title": "Pocket Qwen", "declared_models": ["Qwen/Qwen2.5-7B-Instruct-GGUF"], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# Pocket Qwen\nRuns a GGUF model on your laptop.",
        "app": "import gradio as gr\nfrom llama_cpp import Llama\n"
               "llm = Llama.from_pretrained(\"Qwen/Qwen2.5-7B-Instruct-GGUF\", filename=\"*Q4_K_M.gguf\")\n\n"
               "def chat(m):\n    return llm.create_chat_completion(messages=[{'role':'user','content':m}])\n\n"
               "gr.Interface(chat, 'text', 'text').launch()",
        "matches": [
            {"quest": "Llama Champion", "confidence": 0.95, "evidence": "from llama_cpp import Llama GGUF weights", "source": "app_file"},
            {"quest": "Off the Grid", "confidence": 0.88, "evidence": "local llama_cpp GGUF inference, no remote call", "source": "app_file"},
        ],
    },
    {
        "id": "synthetic/local-llama3b-tiny",
        "title": "Tiny Llama Buddy", "declared_models": ["meta-llama/Llama-3.2-3B-Instruct"], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# Tiny Llama Buddy\nA small local helper.",
        "app": "import gradio as gr\nfrom transformers import AutoModelForCausalLM\n"
               "model = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-3.2-3B-Instruct\", device_map='cuda')\n\n"
               "def gen(p):\n    return model_generate(p)\n\n"
               "gr.Interface(gen, 'text', 'text').launch()",
        "matches": [
            {"quest": "Off the Grid", "confidence": 0.85, "evidence": "local from_pretrained, in-process inference", "source": "app_file"},
            {"quest": "Tiny Titan", "confidence": 0.82, "evidence": "Llama-3.2-3B is a 3B model", "source": "app_file"},
        ],
    },
    {
        "id": "synthetic/local-openbmb-positive",
        "title": "Pocket MiniCPM", "declared_models": ["openbmb/MiniCPM5-1B-GGUF"], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# Pocket MiniCPM\nRuns MiniCPM locally via llama.cpp.",
        "app": "import gradio as gr\nfrom llama_cpp import Llama\n"
               "llm = Llama.from_pretrained(\"openbmb/MiniCPM5-1B-GGUF\", filename=\"*Q4_K_M.gguf\")\n\n"
               "def chat(m):\n    return llm.create_chat_completion(messages=[{'role':'user','content':m}])\n\n"
               "gr.Interface(chat, 'text', 'text').launch()",
        "matches": [
            {"quest": "Llama Champion", "confidence": 0.95, "evidence": "from llama_cpp import Llama", "source": "app_file"},
            {"quest": "OpenBMB", "confidence": 0.95, "evidence": "openbmb/MiniCPM5-1B-GGUF model", "source": "app_file"},
            {"quest": "Off the Grid", "confidence": 0.9, "evidence": "local llama_cpp GGUF, no remote call", "source": "app_file"},
            {"quest": "Tiny Titan", "confidence": 0.82, "evidence": "MiniCPM5-1B is a 1B model", "source": "app_file"},
        ],
    },
    {
        "id": "synthetic/local-minicpmv-positive",
        "title": "Vision Notes", "declared_models": ["openbmb/MiniCPM-V-4_6"], "tags": ["gradio"], "app_file": "app.py",
        "readme": "# Vision Notes\nReads images with MiniCPM-V locally.",
        "app": "import gradio as gr\nfrom transformers import AutoModel\n"
               "model = AutoModel.from_pretrained(\"openbmb/MiniCPM-V-4_6\", trust_remote_code=True, device_map='cuda')\n\n"
               "def caption(img):\n    return model.chat(image=img, msgs=[])\n\n"
               "gr.Interface(caption, 'image', 'text').launch()",
        "matches": [
            {"quest": "OpenBMB", "confidence": 0.95, "evidence": "openbmb/MiniCPM-V-4_6 model", "source": "app_file"},
            {"quest": "Off the Grid", "confidence": 0.88, "evidence": "local AutoModel.from_pretrained, no remote call", "source": "app_file"},
        ],
    },
]


_REMOTE_RE = re.compile(
    r"InferenceClient|endpoints\.huggingface|\breplicate\b|\btogether\b|openrouter|gradio_client|"
    r"\.modal\.run|api\.openai|api\.anthropic|generativeai|cohere\.Client",
    re.I,
)
# OpenBMB == the openbmb org or its MiniCPM/OpenCPM family (the award is "use their model").
_OPENBMB_RE = re.compile(r"openbmb/|minicpm|opencpm", re.I)


def _check_invariants(examples: list[dict]) -> None:
    """Fail the build on the crisp gold violations behind the GTROX failure modes:
    a remote inference call must not earn Off the Grid, and OpenBMB belongs only to
    openbmb / MiniCPM-family models. (A reliable >4B check for Tiny Titan is left to
    the labeller — parameter counts in code are too noisy: 1.7B, commented models,
    multi-model apps all defeat a regex.)"""
    problems: list[str] = []
    for e in examples:
        user = e["messages"][1]["content"]
        body = user.split("METADATA:", 1)[-1]  # skip the quest list so its prose can't false-positive
        app = body.split("[APP_FILE]", 1)[-1]
        quests = {m["quest"] for m in json.loads(e["messages"][2]["content"])["matches"]}
        pid = e.get("project_id", "?")
        if _REMOTE_RE.search(app) and "Off the Grid" in quests:
            problems.append(f"{pid}: remote inference in app but Off the Grid awarded")
        if "OpenBMB" in quests and not _OPENBMB_RE.search(body):
            problems.append(f"{pid}: OpenBMB awarded without an openbmb / MiniCPM model in the content")
    if problems:
        raise SystemExit("invariant violations:\n  " + "\n  ".join(problems))


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

    # 7) app-only variants of the real remote-inference projects (forces judging the
    #    remote app directly; their gold already excludes Off the Grid)
    covered_app_only = {s for s, _, _ in app_rich[: args.app_only]}
    for slug in REMOTE_INFERENCE_SLUGS:
        if slug not in by_slug or slug in covered_app_only:
            continue
        meta, ms = by_slug[slug]
        kept = [m for m in ms if m["source"] == "app_file"]
        add(example(meta, NO_README, meta["APP_FILE"], kept, variant="remote_app_only"))

    # 8) hand-authored contrastive hard negatives (remote!=local; org-prefix gates)
    for spec in HARD_NEGATIVES:
        add(example(spec, spec["readme"], spec["app"], spec["matches"], variant="hard_negative"))

    _check_invariants(examples)

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
