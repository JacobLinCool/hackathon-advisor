from app import bootstrap, health, index


def test_health_exposes_index_metadata() -> None:
    payload = health()

    assert payload["ok"] is True
    assert payload["projects"] == len(index.projects)
    assert payload["index_algorithm"] == "tfidf-sparse-v1"
    assert len(payload["snapshot_digest"]) == 64


def test_bootstrap_exposes_index_metadata() -> None:
    payload = bootstrap()

    assert payload["index_algorithm"] == "tfidf-sparse-v1"
    assert payload["index_generated_at"]
    assert payload["snapshot_digest"]
    assert payload["top_projects"]
