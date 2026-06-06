from __future__ import annotations

from dataclasses import dataclass

from hackathon_advisor.data import ProjectIndex, SearchHit, tokenize


@dataclass(frozen=True)
class ScoreCard:
    originality: int
    delight: int
    ai_necessity: int
    feasibility: int
    prize_fit: int
    verdict: str
    echoes: tuple[SearchHit, ...]

    @property
    def overall(self) -> float:
        return round(
            (
                self.originality * 0.30
                + self.delight * 0.20
                + self.ai_necessity * 0.20
                + self.feasibility * 0.15
                + self.prize_fit * 0.15
            ),
            1,
        )

    def to_dict(self) -> dict:
        return {
            "originality": self.originality,
            "delight": self.delight,
            "ai_necessity": self.ai_necessity,
            "feasibility": self.feasibility,
            "prize_fit": self.prize_fit,
            "overall": self.overall,
            "verdict": self.verdict,
            "echoes": [
                {
                    "score": round(hit.score, 3),
                    "page_number": hit.page_number,
                    "matched_terms": list(hit.matched_terms),
                    "project": hit.project.to_public_dict(),
                }
                for hit in self.echoes
            ],
        }


def score_idea(index: ProjectIndex, title: str, pitch: str, targets: list[str] | None = None) -> ScoreCard:
    text = f"{title} {pitch}".strip()
    hits = index.search(text, limit=4)
    top_overlap = hits[0].score if hits else 0.0
    tokens = set(tokenize(text))
    targets = targets or []

    originality = clamp_score(10 - round(top_overlap * 18))
    delight = clamp_score(4 + _keyword_count(tokens, {"story", "visual", "game", "ritual", "share", "voice"}) * 2)
    ai_necessity = clamp_score(
        3
        + _keyword_count(tokens, {"agent", "model", "embed", "search", "personal", "speech", "local"}) * 2
    )
    complexity_penalty = _keyword_count(tokens, {"realtime", "video", "multiplayer", "payments", "social"})
    feasibility = clamp_score(8 - complexity_penalty)
    prize_fit = clamp_score(
        4
        + _keyword_count(tokens, {"local", "offline", "small", "llama", "fine", "trace", "gradio"}) * 2
        + min(len(targets), 3)
    )
    verdict = "UNWRITTEN" if top_overlap < 0.16 else f"ECHO x{sum(1 for hit in hits if hit.score >= 0.12)}"
    return ScoreCard(
        originality=originality,
        delight=delight,
        ai_necessity=ai_necessity,
        feasibility=feasibility,
        prize_fit=prize_fit,
        verdict=verdict,
        echoes=tuple(hits),
    )


def clamp_score(value: int) -> int:
    return max(1, min(10, value))


def _keyword_count(tokens: set[str], keywords: set[str]) -> int:
    return sum(1 for keyword in keywords if any(token.startswith(keyword) for token in tokens))
