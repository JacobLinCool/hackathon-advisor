from hackathon_advisor.dashboard import build_dashboard_payload
from hackathon_advisor.data import Project, ProjectIndex, build_index_payload
from hackathon_advisor.dashboard_search import DashboardSearchIndex


def test_dashboard_search_bm25_ranks_text_matches() -> None:
    project_index = fake_project_index()
    payload = build_dashboard_payload(project_index, generated_at="2026-06-08T00:00:00+00:00")
    search_index = DashboardSearchIndex(project_index.projects, payload)

    result = search_index.search("project 4 summary", limit=3)

    assert result["algorithm"] == "bm25-text-v1"
    assert result["results"]
    assert result["results"][0]["project_id"] == "build-small-hackathon/project-4"
    assert result["results"][0]["score"] == 1.0
    assert result["results"][0]["snippets"]


def test_dashboard_search_splits_slug_tokens() -> None:
    project_index = fake_project_index()
    payload = build_dashboard_payload(project_index, generated_at="2026-06-08T00:00:00+00:00")
    search_index = DashboardSearchIndex(project_index.projects, payload)

    result = search_index.search("project-7", limit=3)

    assert result["results"][0]["project_id"] == "build-small-hackathon/project-7"
    assert "project-7" in result["results"][0]["matched_terms"]


def fake_project_index() -> ProjectIndex:
    projects = [
        Project(
            id=f"build-small-hackathon/project-{index}",
            title=f"Project {index}",
            summary=f"Offline project planner {index}",
            tags=("gradio", "local-first"),
            models=("tiny-model",),
            datasets=(),
            likes=index,
            sdk="gradio",
            license="mit",
            created_at="2026-06-01T00:00:00+00:00",
            last_modified=f"2026-06-{index + 1:02d}T00:00:00+00:00",
            host=f"https://project-{index}.hf.space",
            url=f"https://huggingface.co/spaces/build-small-hackathon/project-{index}",
            app_file="app.py",
            app_file_embedding_text=f"local inference gradio small model artifact project {index}",
            readme_body=f"README evidence for project {index}",
        )
        for index in range(10)
    ]
    embeddings = []
    for index in range(10):
        vector = [0.0] * 10
        vector[index] = 1.0
        embeddings.append(vector)
    generated_at = "2026-06-08T00:00:00+00:00"
    source = "https://example.test/spaces"
    return ProjectIndex(
        projects=projects,
        generated_at=generated_at,
        source=source,
        index_payload=build_index_payload(projects, generated_at, source, embeddings),
    )
