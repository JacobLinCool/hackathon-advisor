from hackathon_advisor.dashboard import build_dashboard_payload
from hackathon_advisor.dashboard_repository import DashboardRepository
from hackathon_advisor.dashboard_search import DashboardSearchIndex
from hackathon_advisor.data import Project, ProjectIndex, build_index_payload


def test_overview_reports_counts_clusters_and_quests() -> None:
    repository = analyzed_repository()

    overview = repository.overview()

    assert overview["project_count"] == 10
    assert overview["cluster_count"] >= 1
    assert overview["quest_status"] == "analyzed"
    assert overview["top_clusters"][0]["project_count"] >= 1
    assert overview["top_quests"][0]["project_count"] >= 1
    assert overview["most_liked"][0]["title"] == "Project 9"


def test_overview_handles_not_analyzed_quests() -> None:
    repository = not_analyzed_repository()

    overview = repository.overview()

    assert overview["quest_status"] == "not_analyzed"
    assert overview["top_quests"] == []


def test_list_clusters_returns_labels_not_ids() -> None:
    repository = analyzed_repository()

    listing = repository.list_clusters()

    assert listing["cluster_count"] == len(listing["clusters"])
    for cluster in listing["clusters"]:
        assert cluster["label"]
        assert not cluster["label"].startswith("cluster-")
        assert cluster["project_count"] >= 1


def test_cluster_detail_resolves_fuzzy_label_and_id() -> None:
    repository = analyzed_repository()
    first = repository.list_clusters()["clusters"][0]

    by_label = repository.cluster_detail(first["label"])
    by_lower = repository.cluster_detail(first["label"].lower())
    by_id = repository.cluster_detail("cluster-1")

    assert by_label is not None and by_label["label"] == first["label"]
    assert by_lower is not None and by_lower["label"] == first["label"]
    assert by_id is not None
    assert by_label["examples"]
    assert by_label["examples"][0]["id"].startswith("build-small-hackathon/")


def test_cluster_detail_returns_none_for_unknown_label() -> None:
    repository = analyzed_repository()

    assert repository.cluster_detail("totally unrelated nonsense xyz") is None
    assert repository.cluster_detail("") is None


def test_list_quests_sorted_by_coverage() -> None:
    repository = analyzed_repository()

    listing = repository.list_quests()

    counts = [quest["project_count"] for quest in listing["quests"]]
    assert listing["status"] == "analyzed"
    assert counts == sorted(counts, reverse=True)
    assert counts[0] >= 1


def test_quest_detail_resolves_aliases() -> None:
    repository = analyzed_repository()

    detail = repository.quest_detail("local first")

    assert detail is not None
    assert detail["id"] == "Off the Grid"
    assert detail["project_count"] >= 1
    assert detail["examples"]


def test_quest_detail_rejects_unknown_quest() -> None:
    repository = analyzed_repository()

    assert repository.quest_detail("not a quest at all") is None


def test_quest_detail_spots_label_inside_a_question() -> None:
    repository = analyzed_repository()

    detail = repository.quest_detail("tell me about the Tiny Titan quest")

    assert detail is not None
    assert detail["id"] == "Tiny Titan"


def test_top_by_quests_ranks_projects_by_quest_count() -> None:
    repository = analyzed_repository()

    leaderboard = repository.top_by_quests(limit=5)

    assert leaderboard["status"] == "analyzed"
    assert leaderboard["rows"]
    counts = [row["quest_count"] for row in leaderboard["rows"]]
    assert counts == sorted(counts, reverse=True)
    assert leaderboard["rows"][0]["quest_count"] == 2
    assert leaderboard["rows"][0]["id"] == "build-small-hackathon/project-9"


def test_top_by_quests_empty_when_not_analyzed() -> None:
    repository = not_analyzed_repository()

    leaderboard = repository.top_by_quests()

    assert leaderboard["status"] == "not_analyzed"
    assert leaderboard["rows"] == []
    assert leaderboard["projects_with_quests"] == 0


def test_search_delegates_to_bm25_index() -> None:
    repository = analyzed_repository()

    result = repository.search("project 4 summary", limit=3)

    assert result["total"] >= 1
    assert result["results"][0]["id"] == "build-small-hackathon/project-4"
    assert result["results"][0]["url"]


def test_project_detail_returns_readme_and_app_excerpts() -> None:
    repository = analyzed_repository()

    by_title = repository.project_detail("Project 4")
    by_slug = repository.project_detail("project-4")
    embedded = repository.project_detail("how does Project 4 work?")

    assert by_title is not None
    assert by_title["id"] == "build-small-hackathon/project-4"
    assert "README evidence for project 4" in by_title["readme_excerpt"]
    assert "import gradio" in by_title["app_excerpt"]
    assert by_title["app_file"] == "app.py"
    assert by_title["cluster_label"]
    assert by_slug is not None and by_slug["id"] == by_title["id"]
    assert embedded is not None and embedded["id"] == by_title["id"]


def test_project_detail_returns_none_for_unknown_project() -> None:
    repository = analyzed_repository()

    assert repository.project_detail("totally unknown thing") is None
    assert repository.project_detail("") is None


def test_repository_survives_a_sparse_payload() -> None:
    """The docstring promises empty-but-well-formed results on degraded snapshots."""
    project_index = fake_project_index()
    full_payload = build_dashboard_payload(project_index, generated_at="2026-06-08T00:00:00+00:00")
    search_index = DashboardSearchIndex(project_index.projects, full_payload)
    repository = DashboardRepository({}, search_index)

    assert repository.overview()["project_count"] == 0
    assert repository.list_clusters()["clusters"] == []
    assert repository.list_quests()["quests"] == []
    assert repository.top_by_quests()["rows"] == []
    assert repository.recent_activity()["projects"] == []
    assert repository.cluster_detail("anything") is None
    assert repository.quest_detail("Off the Grid") is not None
    assert repository.search("planner")["results"]


def test_recent_activity_sorts_by_last_modified() -> None:
    repository = analyzed_repository()

    recent = repository.recent_activity(limit=3)

    assert [project["id"] for project in recent["projects"]] == [
        "build-small-hackathon/project-9",
        "build-small-hackathon/project-8",
        "build-small-hackathon/project-7",
    ]
    assert recent["projects"][0]["cluster_label"]


def analyzed_repository() -> DashboardRepository:
    project_index = fake_project_index()
    quest_matches = {project.id: [] for project in project_index.projects}
    quest_matches["build-small-hackathon/project-9"] = [
        {"quest": "Off the Grid", "confidence": 0.9, "evidence": "loads weights locally", "source": "readme"},
        {"quest": "Tiny Titan", "confidence": 0.8, "evidence": "MiniCPM5-1B model", "source": "app_file"},
    ]
    quest_matches["build-small-hackathon/project-4"] = [
        {"quest": "Off the Grid", "confidence": 0.7, "evidence": "local llama.cpp runtime", "source": "readme"},
    ]
    payload = build_dashboard_payload(
        project_index,
        quest_matches=quest_matches,
        quest_source="test-quest-run",
        generated_at="2026-06-08T00:00:00+00:00",
    )
    return DashboardRepository(payload, DashboardSearchIndex(project_index.projects, payload))


def not_analyzed_repository() -> DashboardRepository:
    project_index = fake_project_index()
    payload = build_dashboard_payload(project_index, generated_at="2026-06-08T00:00:00+00:00")
    return DashboardRepository(payload, DashboardSearchIndex(project_index.projects, payload))


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
            app_file_source=f'import gradio as gr\n\ndemo = gr.Interface(fn=str)  # project {index}\ndemo.launch()',
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
