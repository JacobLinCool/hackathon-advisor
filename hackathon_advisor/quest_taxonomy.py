"""Shared taxonomy and prompt format for quest classification.

The dashboard refresh asks MiniCPM5-1B to classify each hackathon project against
the Build Small Hackathon judging dimensions. Beyond the six merit-badge side
quests the advisor already tracks, the contest also runs two main tracks and a set
of sponsor / special awards that are equally detectable from a project's README and
app file (which model it loads, whether it runs on Modal, whether it is agentic).
This module is the single source of truth for that label space and for the strict
two-segment prompt, so the LoRA training data and the live analyzer stay aligned.

Model output schema (one JSON object, nothing else):
    {"matches": [{"quest": str, "confidence": 0.0-1.0, "evidence": str,
                  "source": "readme" | "app_file"}]}

The live analyzer also accepts deterministic ``metadata`` matches produced from
official Space README tags. Those are not model outputs; they are structured
submission intent recorded by the hackathon submitter.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any


SOURCE_README = "readme"
SOURCE_APP_FILE = "app_file"
SOURCE_METADATA = "metadata"
QUEST_SOURCES = (SOURCE_README, SOURCE_APP_FILE, SOURCE_METADATA)

# Canonical system prompt shared by the SFT dataset and the live analyzer so the
# model is trained and served under the exact same instruction.
QUEST_SYSTEM_PROMPT = (
    "You classify hackathon projects against fixed quest dimensions. "
    "Return exactly one strict JSON object and nothing else. "
    "The first character must be { and the last character must be }. "
    "Each match needs quest, confidence, evidence, and source (readme or app_file). "
    "Never emit markdown, prose, a top-level array, extra keys, or an unknown or rephrased quest name."
)

# README / app-file budgets used when rendering a project into the prompt. Kept
# small enough that prompt + completion fit the LoRA max_seq_length with headroom.
README_PROMPT_CHAR_LIMIT = 1500
APP_PROMPT_CHAR_LIMIT = 1900


# Ordered label space. The first six ids match the merit-badge GOALS the advisor
# already uses elsewhere; the rest are the tracks and sponsor / special awards.
QUEST_PROFILES: tuple[dict[str, str], ...] = (
    {
        "id": "Off the Grid",
        "label": "Local-first",
        "description": "Runs the model on-device with no remote inference call: weights load locally and "
        "inference happens in-process, not over a hosted API.",
        "signals": "AWARD on a local in-process load: from_pretrained / pipeline / llama_cpp / diffusers / "
        "vLLM / ONNX, GGUF weights, @spaces.GPU. DISQUALIFY (do NOT award) on ANY remote inference call, even "
        "via huggingface_hub: InferenceClient, HF Inference API/Endpoints, gradio_client to a remote Space, "
        "replicate/together/openrouter/fal/groq, a *.modal.run or other HTTP inference endpoint, or "
        "openai/anthropic/gemini/cohere clients. A remote call disqualifies regardless of which model it names.",
    },
    {
        "id": "Well-Tuned",
        "label": "Fine-tuned",
        "description": "Uses or publishes a fine-tuned or LoRA-adapted model rather than only stock checkpoints.",
        "signals": "LoRA/PEFT adapter, fine-tuned model repo, training script, words like fine-tune, adapter, SFT, distilled.",
    },
    {
        "id": "Off-Brand",
        "label": "Custom frontend",
        "description": "Ships a custom interface beyond default Gradio styling, with a memorable look or voice.",
        "signals": "custom CSS/HTML/JS, gr.HTML, gr.Blocks theme/css=, gr.Server, custom components, bespoke theming.",
    },
    {
        "id": "Llama Champion",
        "label": "llama.cpp path",
        "description": "Runs a model through the llama.cpp runtime.",
        "signals": "llama-cpp-python, from llama_cpp import Llama, GGUF file, llama.cpp, Llama( constructor.",
    },
    {
        "id": "Sharing is Caring",
        "label": "Shareable artifact",
        "description": "Produces an output people can save, post, or compare, or publishes an agent trace to the Hub.",
        "signals": "download/export button, gr.File/gr.DownloadButton, save PNG/PDF/JSON, push_to_hub of a trace or dataset.",
    },
    {
        "id": "Field Notes",
        "label": "Build notes",
        "description": "Documents the build itself with notes, a write-up, or a blog/report link.",
        "signals": "README has a substantial build write-up, devlog, lessons learned, or a blog/report/Notion link.",
    },
    {
        "id": "Backyard AI",
        "label": "Real problem for one person",
        "description": "Solves a concrete real-world problem for a specific, named person or persona.",
        "signals": "README frames a real user and task (caregiving, a relative, a job, a household chore), practical utility.",
    },
    {
        "id": "Thousand Token Wood",
        "label": "Delightful & creative",
        "description": "A delightful, playful, or artistic experience that would not exist without AI.",
        "signals": "story/game/art/whimsy framing, generative characters or worlds, playful tone, creative novelty.",
    },
    {
        "id": "OpenBMB",
        "label": "OpenBMB model",
        "description": "Uses a model published by OpenBMB (the openbmb org), such as the MiniCPM family.",
        "signals": "The model id org prefix must be exactly openbmb/ (openbmb/MiniCPM*, OpenCPM). A model from "
        "any other org is NOT OpenBMB: openai/gpt-oss, Qwen/..., meta-llama/..., google/..., nvidia/..., "
        "microsoft/..., mistralai/... do NOT count just because a model id is present.",
    },
    {
        "id": "Codex",
        "label": "OpenAI Codex",
        "description": "Uses Codex during development with Codex-attributed commits in the Space history or "
        "a linked GitHub repository.",
        "signals": "Codex co-author trailers, Codex-attributed commits, README or article describes Codex as "
        "part of implementation, debugging, documentation, deployment, or demo preparation.",
    },
    {
        "id": "Nemotron",
        "label": "NVIDIA Nemotron",
        "description": "Uses an NVIDIA Nemotron model (Nemotron LLM, Parakeet, Nemotron-Speech, Canary).",
        "signals": "model repo nvidia/...nemotron..., Parakeet, nemotron-speech, Canary ASR.",
    },
    {
        "id": "Modal",
        "label": "Modal-powered",
        "description": "Uses Modal for training, inference, or background compute.",
        "signals": "import modal, modal.App, @app.function, Modal endpoint/volume, README cites Modal compute.",
    },
    {
        "id": "Tiny Titan",
        "label": "Small model (<=4B)",
        "description": "Runs on a genuinely small model of about four billion parameters or fewer.",
        "signals": "AWARD when the model name says <=4B: 0.5B/1B/1.5B/2B/3B/4B or tiny/small/nano/mini "
        "(Qwen2.5-1.5B, MiniCPM5-1B, gemma-2b). Do NOT award for 7B/8B/12B/13B/20B/27B/35B+ models "
        "(e.g. gpt-oss-20b, Qwen2.5-7B); a version number like V-4.6 is not a parameter count.",
    },
    {
        "id": "Best Agent",
        "label": "Agentic",
        "description": "An agentic build: tool use, function calling, planning, or an autonomous multi-step loop.",
        "signals": "tool/function calling, an agent/planner loop, multiple orchestrated tools, ReAct, multi-step reasoning over tools.",
    },
)

QUESTS: tuple[str, ...] = tuple(profile["id"] for profile in QUEST_PROFILES)
QUEST_PROFILE_BY_ID: dict[str, dict[str, str]] = {profile["id"]: profile for profile in QUEST_PROFILES}


def _quest_key(raw: Any) -> str:
    text = " ".join(str(raw or "").replace("&", " and ").casefold().split())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


_QUEST_ALIASES: dict[str, str] = {}
for _profile in QUEST_PROFILES:
    _QUEST_ALIASES[_quest_key(_profile["id"])] = _profile["id"]
    _QUEST_ALIASES[_quest_key(_profile["label"])] = _profile["id"]
    _QUEST_ALIASES[_quest_key(f"Best {_profile['id']}")] = _profile["id"]
    _QUEST_ALIASES[_quest_key(f"Best {_profile['label']}")] = _profile["id"]
    _QUEST_ALIASES[_quest_key(f"Best Use of {_profile['id']}")] = _profile["id"]
    _QUEST_ALIASES[_quest_key(f"Best Use of {_profile['label']}")] = _profile["id"]
_QUEST_ALIASES.update(
    {
        _quest_key("Best MiniCPM Build"): "OpenBMB",
        _quest_key("MiniCPM Build"): "OpenBMB",
        _quest_key("MiniCPM"): "OpenBMB",
        _quest_key("OpenBMB / MiniCPM"): "OpenBMB",
        _quest_key("Best Use of Codex"): "Codex",
        _quest_key("OpenAI Codex"): "Codex",
        _quest_key("OpenAI"): "Codex",
        _quest_key("Codex"): "Codex",
        _quest_key("Offbrand"): "Off-Brand",
        _quest_key("Offgrid"): "Off the Grid",
        _quest_key("Llama Champion badge"): "Llama Champion",
        _quest_key("Modal first"): "Modal",
        _quest_key("Nemotron 3 Nano 4B"): "Nemotron",
        _quest_key("Nemotron Mini 4B"): "Nemotron",
        _quest_key("Small model <=4B"): "Tiny Titan",
        _quest_key("Small model under 4B"): "Tiny Titan",
        _quest_key("Shareable output"): "Sharing is Caring",
        _quest_key("Custom UI"): "Off-Brand",
        _quest_key("Custom interface"): "Off-Brand",
        _quest_key("Local first"): "Off the Grid",
        _quest_key("Fine tuned"): "Well-Tuned",
        _quest_key("Fine tune"): "Well-Tuned",
    }
)


TRACK_METADATA_TAG_TO_QUEST: dict[str, str] = {
    "track:backyard": "Backyard AI",
    "track:wood": "Thousand Token Wood",
}
TRACK_QUESTS: tuple[str, ...] = tuple(TRACK_METADATA_TAG_TO_QUEST.values())

OFFICIAL_METADATA_TAG_TO_QUEST: dict[str, str] = {
    **TRACK_METADATA_TAG_TO_QUEST,
    "sponsor:openbmb": "OpenBMB",
    "sponsor:openai": "Codex",
    "sponsor:nvidia": "Nemotron",
    "sponsor:modal": "Modal",
    "achievement:offgrid": "Off the Grid",
    "achievement:welltuned": "Well-Tuned",
    "achievement:offbrand": "Off-Brand",
    "achievement:llama": "Llama Champion",
    "achievement:sharing": "Sharing is Caring",
    "achievement:fieldnotes": "Field Notes",
    # These tags are not emitted by the current submitter, but they appear in
    # hand-authored Space metadata and map cleanly to existing official prizes.
    "tiny-titan": "Tiny Titan",
    "best-agent": "Best Agent",
}


def quest_profiles(quest_ids: Sequence[str] | None = None) -> list[dict[str, str]]:
    return [
        {"id": profile["id"], "label": profile["label"], "description": profile["description"]}
        for profile in _selected_quest_profiles(quest_ids)
    ]


def quest_label(quest: str) -> str:
    return QUEST_PROFILE_BY_ID.get(quest, {}).get("label", quest)


def canonical_quest_id(raw_quest: Any) -> str:
    quest = " ".join(str(raw_quest or "").split())
    if quest in QUEST_PROFILE_BY_ID:
        return quest
    alias = _QUEST_ALIASES.get(_quest_key(quest))
    if alias:
        return alias
    folded = quest.casefold()
    for known in QUESTS:
        known_folded = known.casefold()
        if folded == known_folded:
            return known
        if folded.startswith(f"{known_folded} (") or folded.startswith(f"{known_folded} - "):
            return known
    raise ValueError(f"unknown quest: {quest!r}")


def canonical_quest_ids(raw_quest: Any) -> tuple[str, ...]:
    quest = " ".join(str(raw_quest or "").split())
    try:
        return (canonical_quest_id(quest),)
    except ValueError as original_error:
        parts = [part.strip() for part in re.split(r"\s*/\s*", quest) if part.strip()]
        if len(parts) <= 1:
            raise original_error
    canonical: list[str] = []
    for part in parts:
        try:
            quest_id = canonical_quest_id(part)
        except ValueError as error:
            raise ValueError(f"unknown quest in composite {quest!r}: {part!r}") from error
        if quest_id not in canonical:
            canonical.append(quest_id)
    return tuple(canonical)


def declared_quest_matches_from_tags(tags: Sequence[Any]) -> list[dict[str, Any]]:
    """Return deterministic quest matches declared by official Space metadata tags."""
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_tag in tags or ():
        tag = " ".join(str(raw_tag or "").split())
        quest = OFFICIAL_METADATA_TAG_TO_QUEST.get(tag.casefold())
        if not quest or quest in seen:
            continue
        seen.add(quest)
        matches.append(
            normalize_match(
                {
                    "quest": quest,
                    "confidence": 1.0,
                    "evidence": f"official Space tag {tag.casefold()}",
                    "source": SOURCE_METADATA,
                }
            )
        )
    return matches


def metadata_suppressed_quests_from_tags(tags: Sequence[Any]) -> set[str]:
    """Return quest ids that metadata should remove from model inference."""
    suppressed: set[str] = set()
    declared_tags = {" ".join(str(raw_tag or "").split()).casefold() for raw_tag in tags or ()}
    if declared_tags & set(TRACK_METADATA_TAG_TO_QUEST):
        suppressed.update(TRACK_QUESTS)
    return suppressed


def _selected_quest_profiles(quest_ids: Sequence[str] | None) -> tuple[dict[str, str], ...]:
    if quest_ids is None:
        return QUEST_PROFILES
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_quest in quest_ids:
        quest = canonical_quest_id(raw_quest)
        if quest in seen:
            continue
        seen.add(quest)
        selected.append(QUEST_PROFILE_BY_ID[quest])
    return tuple(selected)


def _clip(text: str, limit: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + " ..."


_IMPORT_RE = re.compile(r"^\s*(?:import\s+\w|from\s+\w[\w.]*\s+import)\b")
_REPO_ID_RE = re.compile(r"\b[\w-]+/[\w.\-]+\b")


def build_readme_segment(readme_body: str) -> str:
    return " ".join(str(readme_body or "").split())[: README_PROMPT_CHAR_LIMIT * 2]


def build_app_segment(app_source: str, app_signals: str = "") -> str:
    """Compose an app-file view that keeps imports and asset ids inside budget.

    Gradio apps front-load the decisive quest signals (which library is imported,
    which model repo is loaded) but a deep model id can fall outside a head slice,
    so imports are hoisted and any repo-id-looking tokens from the AST signals that
    are still missing are appended as a compact ASSETS line. The SFT dataset and the
    live analyzer both call this so the model sees the same app view either way.
    """
    source = str(app_source or "")
    if not source.strip() and not str(app_signals or "").strip():
        return ""
    imports = [line.strip() for line in source.splitlines() if _IMPORT_RE.match(line)]
    seen: set[str] = set()
    ordered_imports = [imp for imp in imports if not (imp in seen or seen.add(imp))][:40]
    head_budget = APP_PROMPT_CHAR_LIMIT * 2
    parts: list[str] = []
    if ordered_imports:
        parts.append("\n".join(ordered_imports))
    parts.append(source)
    composed = "\n\n".join(parts)[:head_budget]
    repo_ids = {token for token in _REPO_ID_RE.findall(app_signals or "") if "/" in token}
    missing = sorted(rid for rid in repo_ids if rid not in composed)
    if missing:
        composed = f"{composed}\n\nASSETS: {', '.join(missing[:12])}"
    return composed


def render_quest_prompt(
    *,
    title: str,
    sdk: str,
    declared_models: Sequence[str],
    tags: Sequence[str],
    readme_segment: str,
    app_file_name: str,
    app_file_segment: str,
    include_signals: bool = True,
    quest_ids: Sequence[str] | None = None,
) -> str:
    """Render the canonical two-segment classification prompt.

    The same renderer feeds both the SFT dataset and the live analyzer so the model
    never sees a different shape at training and inference time.
    """
    profiles = _selected_quest_profiles(quest_ids)
    quest_lines = [f"- {profile['id']}: {profile['description']}" for profile in profiles]
    if include_signals:
        quest_lines = [
            f"- {profile['id']}: {profile['description']} Signals: {profile['signals']}"
            for profile in profiles
        ]
    readme_text = _clip(readme_segment, README_PROMPT_CHAR_LIMIT) or "(no README description provided)"
    app_label = app_file_name.strip() or "(unknown)"
    app_text = _clip(app_file_segment, APP_PROMPT_CHAR_LIMIT) or "(no app file available)"
    metadata = {
        "title": (title or "").strip(),
        "sdk": (sdk or "").strip(),
        "declared_models": [str(model) for model in declared_models or []],
        "tags": [str(tag) for tag in tags or []],
    }
    return "\n".join(
        [
            "Classify this hackathon project against the quest dimensions below.",
            "Read the two evidence segments (README and APP_FILE) and judge each quest only from them.",
            "",
            "Quests (copy the id on the left verbatim):",
            *quest_lines,
            "",
            "Rules:",
            "- Include a quest only when a segment gives clear, specific evidence.",
            "- quest must be one id from the list above, copied exactly. Never invent or rephrase a quest name.",
            "- confidence is a number between 0 and 1.",
            "- evidence is a 3-to-12 word quote or tight paraphrase taken from the segment you cite.",
            '- source is "readme" when the evidence is in the README segment, "app_file" when it is in the APP_FILE segment.',
            "- At most one match per quest. Sort matches by confidence, highest first.",
            "- If no quest has clear evidence, return an empty matches list.",
            '- Output exactly one JSON object: {"matches":[{"quest":"...","confidence":0.0,"evidence":"...","source":"readme"}]}.',
            "- No markdown, no code fences, no commentary, no extra keys.",
            "",
            f"METADATA: {json.dumps(metadata, ensure_ascii=False)}",
            "",
            "[README]",
            readme_text,
            "",
            f"[APP_FILE] {app_label}",
            app_text,
        ]
    )


def normalize_match(match: Mapping[str, Any], *, evidence_limit: int = 360) -> dict[str, Any]:
    """Validate and canonicalize one match dict. Raises ValueError on schema drift."""
    quest = canonical_quest_id(match.get("quest"))
    try:
        confidence = float(match.get("confidence"))
    except (TypeError, ValueError) as error:
        raise ValueError("confidence must be numeric") from error
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be greater than 0 and no more than 1")
    evidence = " ".join(str(match.get("evidence") or "").split())
    if not evidence:
        raise ValueError("evidence must not be empty")
    if _looks_like_prompt_taxonomy(evidence):
        raise ValueError("evidence must come from README or APP_FILE, not quest instructions")
    source = str(match.get("source") or "")
    if source not in QUEST_SOURCES:
        raise ValueError(f"source must be one of {QUEST_SOURCES}, got {source!r}")
    return {
        "quest": quest,
        "confidence": round(confidence, 3),
        "evidence": evidence[:evidence_limit],
        "source": source,
    }


def _looks_like_prompt_taxonomy(evidence: str) -> bool:
    normalized = " ".join(evidence.casefold().split())
    if "signals:" in normalized:
        return True
    return any(
        normalized.startswith(" ".join(profile[field].casefold().split())[:80])
        for profile in QUEST_PROFILES
        for field in ("description",)
    )
