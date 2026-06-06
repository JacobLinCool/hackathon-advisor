from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


@dataclass(frozen=True)
class Correction:
    original: str
    canonical: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "canonical": self.canonical,
            "confidence": round(self.confidence, 3),
        }


ALIASES: dict[str, tuple[str, ...]] = {
    "Nemotron": ("nemotron", "nemo tron", "neutron", "nemotran", "nemo-tron"),
    "MiniCPM5": ("minicpm5", "mini cpm5", "mini cpm", "open cpm", "opencpm5", "cpm five"),
    "EmbeddingGemma": ("embedding gemma", "embeddinggemma", "gemma embedding", "embedded gemma"),
    "ZeroGPU": ("zero gpu", "zerogpu", "zero-gpu", "zero g p u"),
    "Gradio Server": ("gradio server", "gradio.server", "server mode"),
    "Build Small Hackathon": ("build small", "build-small", "small hackathon"),
    "Off the Grid": ("off the grid", "off-grid", "offline badge"),
    "Well-Tuned": ("well tuned", "well-tuned", "fine tune", "finetune", "lora"),
    "Tiny Titan": ("tiny titan", "tiny tight end", "tiny-titan"),
    "Llama Champion": ("llama champion", "llama.cpp", "llama cpp", "llama badge"),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)?", re.IGNORECASE)


def normalize_text(text: str) -> tuple[str, list[Correction]]:
    normalized = text
    corrections: list[Correction] = []
    spans = _candidate_spans(text)
    used: set[str] = set()

    for canonical, aliases in ALIASES.items():
        best: tuple[str, float] | None = None
        for alias in aliases:
            for span in spans:
                confidence = _similarity(alias, span)
                if confidence >= 0.88 and (best is None or confidence > best[1]):
                    best = (span, confidence)
        if not best:
            continue

        original, confidence = best
        if original.lower() in used or original == canonical:
            continue
        used.add(original.lower())
        normalized = re.sub(re.escape(original), canonical, normalized, count=1, flags=re.IGNORECASE)
        corrections.append(Correction(original=original, canonical=canonical, confidence=confidence))

    return normalized, corrections


def _candidate_spans(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    spans = set(tokens)
    for size in (2, 3):
        for index in range(max(0, len(tokens) - size + 1)):
            spans.add(" ".join(tokens[index : index + size]))
    return sorted(spans, key=len, reverse=True)


def _similarity(left: str, right: str) -> float:
    compact_left = re.sub(r"[^a-z0-9]", "", left.lower())
    compact_right = re.sub(r"[^a-z0-9]", "", right.lower())
    if not compact_left or not compact_right:
        return 0.0
    if compact_left == compact_right:
        return 1.0
    return SequenceMatcher(None, compact_left, compact_right).ratio()
