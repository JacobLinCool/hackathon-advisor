from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from hackathon_advisor.data import ProjectIndex, normalize_vector, tokenize


def load_test_index() -> ProjectIndex:
    return ProjectIndex.from_files(
        Path("data/projects.json"),
        Path("data/project_index.json"),
        query_embedder=test_query_embedder,
    )


def test_query_embedder(text: str) -> tuple[float, ...]:
    vector = [0.0] * 768
    for token in tokenize(text):
        digest = sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % len(vector)
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    if not any(vector):
        vector[0] = 1.0
    return normalize_vector(vector)
