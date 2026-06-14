from __future__ import annotations

from hackathon_advisor.dashboard import (
    CLUSTER_LABEL_ALGORITHM,
    build_dashboard_payload,
    validate_dashboard_payload,
)
from hackathon_advisor.data import Project, ProjectIndex, build_index_payload
from hackathon_advisor.quest_analysis import (
    MiniCPMQuestAnalyzer,
    QuestAnalysisError,
    _extract_json_object,
    create_quest_analyzer,
    render_project_quest_prompt,
    remaining_project_quest_ids,
    validate_quest_analysis_payload,
)
from hackathon_advisor.quest_taxonomy import declared_quest_matches_from_tags
from hackathon_advisor.tools import GOALS


def test_dashboard_builder_projects_embeddings_with_tsne_and_clusters() -> None:
    index = fake_index()
    quest_matches = {
        project.id: [
            {
                "quest": GOALS[project_index % len(GOALS)],
                "confidence": 0.75,
                "evidence": "project evidence matches the quest",
                "source": "readme" if project_index % 2 == 0 else "app_file",
            }
        ]
        for project_index, project in enumerate(index.projects)
    }

    payload = build_dashboard_payload(
        index,
        quest_matches=quest_matches,
        quest_source="fake-strict-analyzer",
        generated_at="2026-06-08T00:00:00+00:00",
    )

    validate_dashboard_payload(payload)
    assert payload["layout"]["algorithm"] == "tsne"
    assert payload["layout"]["metric"] == "cosine"
    assert len(payload["points"]) == len(index.projects)
    assert len(payload["clusters"]) == 6
    assert payload["links"]
    assert payload["quest_report"]["status"] == "analyzed"
    assert all(0 <= point["x"] <= 100 and 0 <= point["y"] <= 100 for point in payload["points"])
    assert all(point["quest_ids"] for point in payload["points"])
    assert payload["cluster_label_algorithm"] == CLUSTER_LABEL_ALGORITHM


def test_dashboard_builder_is_deterministic_for_fixed_vectors() -> None:
    index = fake_index()

    left = build_dashboard_payload(index, generated_at="2026-06-08T00:00:00+00:00")
    right = build_dashboard_payload(index, generated_at="2026-06-08T00:00:00+00:00")

    assert [(point["id"], point["x"], point["y"]) for point in left["points"]] == [
        (point["id"], point["x"], point["y"]) for point in right["points"]
    ]
    assert left["clusters"] == right["clusters"]


def test_dashboard_cluster_labels_ignore_hackathon_wide_noise() -> None:
    index = noisy_cluster_label_index()

    payload = build_dashboard_payload(index, generated_at="2026-06-08T00:00:00+00:00")

    banned = {"ai", "build-small-hackathon", "gradio", "hackathon", "project", "region", "us"}
    keywords = {keyword for cluster in payload["clusters"] for keyword in cluster["keywords"]}
    assert keywords.isdisjoint(banned)
    assert {"dream", "family", "garden", "notice", "order", "repair"} & keywords
    assert all("region:us" not in point["tags"] for point in payload["points"])


def test_quest_analysis_validation_accepts_strict_project_coverage() -> None:
    projects = fake_projects(4)
    raw = {
        "projects": [
            {
                "project_id": project.id,
                "matches": [
                    {
                        "quest": "Off the Grid",
                        "confidence": 0.9,
                        "evidence": "runs without proprietary inference APIs",
                        "source": "app_file",
                    }
                ],
            }
            for project in projects
        ]
    }

    validated = validate_quest_analysis_payload(raw, projects, source="fake")

    assert validated.source == "fake"
    assert set(validated.matches_by_project) == {project.id for project in projects}
    assert validated.matches_by_project[projects[0].id][0]["quest"] == "Off the Grid"


def test_quest_analysis_validation_rejects_malformed_output() -> None:
    projects = fake_projects(2)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [{"quest": "Off the Grid", "confidence": 1.2, "evidence": ""}],
            }
        ]
    }

    try:
        validate_quest_analysis_payload(raw, projects)
    except QuestAnalysisError as error:
        assert "confidence" in str(error) or "missed" in str(error)
    else:
        raise AssertionError("malformed quest analysis should fail")


def test_quest_analysis_validation_requires_source_field() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [{"quest": "Off the Grid", "confidence": 0.9, "evidence": "local model"}],
            }
        ]
    }

    try:
        validate_quest_analysis_payload(raw, projects)
    except QuestAnalysisError as error:
        assert "source" in str(error)
    else:
        raise AssertionError("a match without a source must be rejected")


def test_quest_analysis_validation_rejects_prompt_taxonomy_as_evidence() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [
                    {
                        "quest": "Off the Grid",
                        "confidence": 0.0,
                        "evidence": (
                            "Runs entirely on local or open-weight models with no proprietary cloud inference APIs. "
                            "Signals: local transformers model load"
                        ),
                        "source": "readme",
                    }
                ],
            }
        ]
    }

    try:
        validate_quest_analysis_payload(raw, projects)
    except QuestAnalysisError as error:
        assert "confidence" in str(error) or "quest instructions" in str(error)
    else:
        raise AssertionError("prompt taxonomy evidence must be rejected")


def test_quest_analysis_validation_accepts_expanded_track_quests() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [
                    {"quest": "Nemotron", "confidence": 0.8, "evidence": "nvidia parakeet asr", "source": "app_file"},
                    {"quest": "Tiny Titan", "confidence": 0.7, "evidence": "MiniCPM5-1B", "source": "readme"},
                ],
            }
        ]
    }

    validated = validate_quest_analysis_payload(raw, projects, source="fake")

    quests = {match["quest"] for match in validated.matches_by_project[projects[0].id]}
    assert quests == {"Nemotron", "Tiny Titan"}
    assert validated.matches_by_project[projects[0].id][0]["source"] in {"readme", "app_file"}


def test_declared_quest_matches_from_official_space_tags() -> None:
    matches = declared_quest_matches_from_tags(
        [
            "track:wood",
            "sponsor:openai",
            "sponsor:nvidia",
            "achievement:llama",
            "tiny-titan",
            "region:us",
            "unknown:tag",
        ]
    )

    quests = [match["quest"] for match in matches]
    assert quests == ["Thousand Token Wood", "Codex", "Nemotron", "Llama Champion", "Tiny Titan"]
    assert {match["source"] for match in matches} == {"metadata"}
    assert all(match["confidence"] == 1.0 for match in matches)


def test_quest_analysis_validation_canonicalizes_known_label_suffixes() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [
                    {
                        "quest": "Off the Grid (LOCAL-FIRST)",
                        "confidence": 0.9,
                        "evidence": "local gguf model",
                        "source": "app_file",
                    }
                ],
            }
        ]
    }

    validated = validate_quest_analysis_payload(raw, projects, source="fake")

    assert validated.matches_by_project[projects[0].id][0]["quest"] == "Off the Grid"


def test_quest_analysis_validation_expands_known_composite_quest_labels() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [
                    {
                        "quest": "Best MiniCPM Build / Tiny Titan",
                        "confidence": 0.84,
                        "evidence": "MiniCPM5-1B model",
                        "source": "app_file",
                    },
                    {
                        "quest": "Off-Brand / Sharing is Caring",
                        "confidence": 0.72,
                        "evidence": "custom UI exports a card",
                        "source": "readme",
                    },
                ],
            }
        ]
    }

    validated = validate_quest_analysis_payload(raw, projects, source="fake")

    quests = [match["quest"] for match in validated.matches_by_project[projects[0].id]]
    assert quests == ["OpenBMB", "Tiny Titan", "Off-Brand", "Sharing is Caring"]


def test_quest_analysis_validation_accepts_best_prefixed_known_labels() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [
                    {
                        "quest": "Best Well-Tuned",
                        "confidence": 0.84,
                        "evidence": "PEFT adapter",
                        "source": "app_file",
                    }
                ],
            }
        ]
    }

    validated = validate_quest_analysis_payload(raw, projects, source="fake")

    assert validated.matches_by_project[projects[0].id][0]["quest"] == "Well-Tuned"


def test_quest_analysis_validation_accepts_best_use_of_known_labels() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [
                    {
                        "quest": "Best Use of Modal",
                        "confidence": 0.84,
                        "evidence": "modal.App background worker",
                        "source": "app_file",
                    }
                ],
            }
        ]
    }

    validated = validate_quest_analysis_payload(raw, projects, source="fake")

    assert validated.matches_by_project[projects[0].id][0]["quest"] == "Modal"


def test_quest_analysis_validation_accepts_common_compact_aliases() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [
                    {
                        "quest": "Offbrand",
                        "confidence": 0.84,
                        "evidence": "custom frontend",
                        "source": "readme",
                    },
                    {
                        "quest": "Offgrid",
                        "confidence": 0.82,
                        "evidence": "loads weights locally",
                        "source": "app_file",
                    },
                    {
                        "quest": "Llama Champion badge",
                        "confidence": 0.8,
                        "evidence": "llama-cpp-python",
                        "source": "app_file",
                    },
                    {
                        "quest": "Modal-first",
                        "confidence": 0.79,
                        "evidence": "Modal serverless GPUs",
                        "source": "readme",
                    },
                    {
                        "quest": "Nemotron-3 Nano 4B",
                        "confidence": 0.78,
                        "evidence": "Nemotron 3 Nano 4B",
                        "source": "readme",
                    },
                ],
            }
        ]
    }

    validated = validate_quest_analysis_payload(raw, projects, source="fake")

    quests = [match["quest"] for match in validated.matches_by_project[projects[0].id]]
    assert quests == ["Off-Brand", "Off the Grid", "Llama Champion", "Modal", "Nemotron"]


def test_quest_analysis_validation_rejects_unknown_composite_quest_labels() -> None:
    projects = fake_projects(1)
    raw = {
        "projects": [
            {
                "project_id": projects[0].id,
                "matches": [
                    {
                        "quest": "Mystery Award / Tiny Titan",
                        "confidence": 0.84,
                        "evidence": "tiny model",
                        "source": "app_file",
                    }
                ],
            }
        ]
    }

    try:
        validate_quest_analysis_payload(raw, projects, source="fake")
    except QuestAnalysisError as error:
        assert "unknown quest in composite" in str(error)
    else:
        raise AssertionError("unknown composite quest labels must be rejected")


def test_quest_json_extractor_accepts_fenced_object() -> None:
    payload = _extract_json_object('```json\n{"projects":[]}\n```')

    assert payload == {"projects": []}


def test_minicpm_quest_analyzer_attaches_project_id_to_match_payload(monkeypatch) -> None:
    project = fake_projects(1)[0]
    analyzer = MiniCPMQuestAnalyzer()
    monkeypatch.setattr(analyzer, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        analyzer,
        "_generate_json",
        lambda _prompt: {
            "matches": [
                {
                    "quest": "Off the Grid",
                    "confidence": 0.86,
                    "evidence": "local model artifact",
                    "source": "app_file",
                }
            ]
        },
    )

    result = analyzer.analyze([project])

    assert set(result) == {project.id}
    assert result[project.id][0]["quest"] == "Off the Grid"


def test_minicpm_quest_analyzer_uses_metadata_before_model_matches(monkeypatch) -> None:
    project = fake_projects(1)[0]
    project = Project(
        **{
            **project.to_refresh_snapshot_dict(),
            "tags": ["track:wood", "sponsor:openai"],
            "readme_body": "Uses MiniCPM locally.",
            "app_file_source": "model = 'openbmb/MiniCPM5-1B'",
        }
    )
    prompts: list[str] = []
    analyzer = MiniCPMQuestAnalyzer()
    monkeypatch.setattr(analyzer, "_ensure_loaded", lambda: None)

    def fake_generate(prompt: str) -> dict:
        prompts.append(prompt)
        return {
            "matches": [
                {
                    "quest": "Best Use of Codex",
                    "confidence": 0.88,
                    "evidence": "Codex helped development",
                    "source": "readme",
                },
                {
                    "quest": "OpenBMB",
                    "confidence": 0.9,
                    "evidence": "openbmb/MiniCPM5-1B",
                    "source": "app_file",
                },
            ]
        }

    monkeypatch.setattr(analyzer, "_generate_json", fake_generate)

    result = analyzer.analyze([project])

    assert prompts
    assert "- Thousand Token Wood:" not in prompts[0]
    assert "- Backyard AI:" not in prompts[0]
    assert "- Codex:" not in prompts[0]
    assert [match["quest"] for match in result[project.id]] == [
        "Thousand Token Wood",
        "Codex",
        "OpenBMB",
    ]
    assert result[project.id][1]["source"] == "metadata"


def test_minicpm_quest_analyzer_skips_model_when_metadata_covers_every_profile(monkeypatch) -> None:
    project = fake_projects(1)[0]
    project = Project(
        **{
            **project.to_refresh_snapshot_dict(),
            "tags": [
                "achievement:offgrid",
                "achievement:welltuned",
                "achievement:offbrand",
                "achievement:llama",
                "achievement:sharing",
                "achievement:fieldnotes",
                "track:backyard",
                "track:wood",
                "sponsor:openbmb",
                "sponsor:openai",
                "sponsor:nvidia",
                "sponsor:modal",
                "tiny-titan",
                "best-agent",
            ],
            "readme_body": "All declared in metadata.",
            "app_file_source": "",
        }
    )
    analyzer = MiniCPMQuestAnalyzer()

    def fail_load() -> None:
        raise AssertionError("fully declared metadata should not load MiniCPM")

    monkeypatch.setattr(analyzer, "_ensure_loaded", fail_load)

    result = analyzer.analyze([project])

    assert not remaining_project_quest_ids(project)
    assert len(result[project.id]) == 14
    assert {match["source"] for match in result[project.id]} == {"metadata"}


def test_minicpm_quest_analyzer_repairs_invalid_json_with_base_model(monkeypatch) -> None:
    analyzer = MiniCPMQuestAnalyzer()
    monkeypatch.setattr(analyzer, "_ensure_loaded", lambda: None)
    outputs = [
        # truncated output the deterministic quote-escaper cannot fix -> falls through to base-model repair
        '{"matches":[{"quest":"Off-Brand","confidence":0.8,"evidence":"truncated',
        '{"matches":[{"quest":"Off-Brand","confidence":0.8,"evidence":"custom Server title","source":"app_file"}]}',
    ]
    calls: list[bool] = []

    def fake_generate(_system: str, _prompt: str, *, disable_adapter: bool = False) -> str:
        calls.append(disable_adapter)
        return outputs.pop(0)

    monkeypatch.setattr(analyzer, "_generate_text", fake_generate)

    result = analyzer.analyze([fake_projects(1)[0]])

    assert calls == [False, True]
    assert result["build-small-hackathon/project-0"][0]["evidence"] == "custom Server title"


def test_minicpm_quest_analyzer_escapes_inner_quotes_without_repair(monkeypatch) -> None:
    analyzer = MiniCPMQuestAnalyzer()
    monkeypatch.setattr(analyzer, "_ensure_loaded", lambda: None)
    calls: list[bool] = []

    def fake_generate(_system: str, _prompt: str, *, disable_adapter: bool = False) -> str:
        calls.append(disable_adapter)
        return (
            '{"matches":[{"quest":"Off-Brand","confidence":0.8,'
            '"evidence":"app = Server(title="Broken")","source":"app_file"}]}'
        )

    monkeypatch.setattr(analyzer, "_generate_text", fake_generate)

    result = analyzer.analyze([fake_projects(1)[0]])

    assert calls == [False]  # deterministic escape; no base-model repair round-trip
    assert result["build-small-hackathon/project-0"][0]["evidence"] == 'app = Server(title="Broken")'


def test_minicpm_quest_analyzer_tolerates_unparseable_project(monkeypatch) -> None:
    analyzer = MiniCPMQuestAnalyzer()
    monkeypatch.setattr(analyzer, "_ensure_loaded", lambda: None)

    def fail(_prompt: str) -> dict:
        raise QuestAnalysisError("quest analyzer returned invalid JSON")

    monkeypatch.setattr(analyzer, "_generate_json", fail)

    result = analyzer.analyze([fake_projects(1)[0]])

    assert result == {"build-small-hackathon/project-0": []}


def test_minicpm_quest_analyzer_repairs_schema_errors_with_base_model(monkeypatch) -> None:
    project = fake_projects(1)[0]
    analyzer = MiniCPMQuestAnalyzer()
    monkeypatch.setattr(analyzer, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(
        analyzer,
        "_generate_json",
        lambda _prompt: {
            "matches": [
                {"quest": "Off the Grid", "confidence": 0.4, "evidence": "local model", "source": "readme"},
                {"quest": "Off the Grid", "confidence": 0.8, "evidence": "no cloud API", "source": "app_file"},
            ]
        },
    )
    repair_errors: list[str] = []

    def fake_repair(_raw, error: str):
        repair_errors.append(error)
        return {
            "matches": [
                {"quest": "Off the Grid", "confidence": 0.8, "evidence": "no cloud API", "source": "app_file"}
            ]
        }

    monkeypatch.setattr(analyzer, "_repair_schema_json", fake_repair)

    result = analyzer.analyze([project])

    assert "duplicate quest" in repair_errors[0]
    assert result[project.id] == [
        {"quest": "Off the Grid", "confidence": 0.8, "evidence": "no cloud API", "source": "app_file"}
    ]


def test_quest_prompt_uses_raw_readme_and_app_source_segments() -> None:
    project = Project(
        id="build-small-hackathon/two-segment",
        title="Two Segment",
        summary="card summary should not drive quest analysis",
        tags=("gradio", "region:us"),
        models=("openbmb/MiniCPM5-1B",),
        datasets=(),
        likes=1,
        sdk="gradio",
        license="mit",
        created_at="2026-06-01T00:00:00+00:00",
        last_modified="2026-06-08T00:00:00+00:00",
        host="https://two-segment.hf.space",
        url="https://huggingface.co/spaces/build-small-hackathon/two-segment",
        app_file="app.py",
        app_file_embedding_text="compact app signals should not drive quest analysis",
        readme_body="README evidence: field notes and a tiny OpenBMB model.",
        app_file_source="from llama_cpp import Llama\nmodel = 'openbmb/MiniCPM5-1B'",
    )

    prompt = render_project_quest_prompt(project)

    assert "[README]" in prompt
    assert "README evidence: field notes" in prompt
    assert "[APP_FILE] app.py" in prompt
    assert "from llama_cpp import Llama" in prompt
    assert "card summary should not drive quest analysis" not in prompt
    assert "compact app signals should not drive quest analysis" not in prompt
    assert "region:us" not in prompt


def test_quest_analyzer_rejects_non_minicpm_backend(monkeypatch) -> None:
    monkeypatch.setenv("ADVISOR_QUEST_ANALYZER_BACKEND", "rules")

    try:
        create_quest_analyzer(device="local")
    except QuestAnalysisError as error:
        assert "minicpm-transformers" in str(error)
    else:
        raise AssertionError("dashboard refresh should only accept MiniCPM quest analysis")


def fake_index() -> ProjectIndex:
    projects = fake_projects(12)
    embeddings = []
    for index, _project in enumerate(projects):
        vector = [0.0] * 16
        vector[index % 3] = 3.0
        vector[3 + index % 5] = 1.5
        vector[8 + index % 8] = 0.7
        embeddings.append(vector)
    snapshot_generated_at = "2026-06-08T00:00:00+00:00"
    source = "https://example.test/spaces"
    payload = build_index_payload(projects, snapshot_generated_at, source, embeddings)
    return ProjectIndex(
        projects=projects,
        generated_at=snapshot_generated_at,
        source=source,
        index_payload=payload,
    )


def noisy_cluster_label_index() -> ProjectIndex:
    themes = [
        ("dream", ("Dream Lantern", "Dream Atlas"), "dream journal symbolic oracle"),
        ("family", ("Family Ledger", "Care Kinship"), "family care bill coordination"),
        ("garden", ("Garden Notebook", "Seed Exchange"), "garden seed neighborhood plants"),
        ("notice", ("Notice Helper", "Scam Screen"), "notice scam safety verification"),
        ("order", ("Order Desk", "Inventory Voice"), "order inventory audio assistant"),
        ("repair", ("Repair Coach", "Tool Shed"), "repair maintenance workshop"),
    ]
    projects: list[Project] = []
    embeddings = []
    for theme_index, (theme, titles, summary) in enumerate(themes):
        for title in titles:
            projects.append(
                Project(
                    id=f"build-small-hackathon/{title.lower().replace(' ', '-')}",
                    title=title,
                    summary=(
                        f"{summary} for a build-small-hackathon AI project in the US region "
                        "with a Gradio demo."
                    ),
                    tags=("build-small-hackathon", "ai", "gradio", "region:us", theme),
                    models=("tiny-model",),
                    datasets=(),
                    likes=theme_index,
                    sdk="gradio",
                    license="mit",
                    created_at="2026-06-01T00:00:00+00:00",
                    last_modified=f"2026-06-{theme_index + 1:02d}T00:00:00+00:00",
                    host=f"https://{title.lower().replace(' ', '-')}.hf.space",
                    url=f"https://huggingface.co/spaces/build-small-hackathon/{title.lower().replace(' ', '-')}",
                    app_file="app.py",
                    app_file_embedding_text="shared local small model app",
                )
            )
            vector = [0.0] * len(themes)
            vector[theme_index] = 1.0
            embeddings.append(vector)

    snapshot_generated_at = "2026-06-08T00:00:00+00:00"
    source = "https://example.test/spaces"
    payload = build_index_payload(projects, snapshot_generated_at, source, embeddings)
    return ProjectIndex(
        projects=projects,
        generated_at=snapshot_generated_at,
        source=source,
        index_payload=payload,
    )


def fake_projects(count: int) -> list[Project]:
    return [
        Project(
            id=f"build-small-hackathon/project-{index}",
            title=f"Project {index}",
            summary=f"Offline project planner {index}",
            tags=("gradio", "local-first"),
            models=("tiny-model",),
            datasets=(),
            likes=index % 5,
            sdk="gradio",
            license="mit",
            created_at="2026-06-01T00:00:00+00:00",
            last_modified=f"2026-06-{index + 1:02d}T00:00:00+00:00",
            host=f"https://project-{index}.hf.space",
            url=f"https://huggingface.co/spaces/build-small-hackathon/project-{index}",
            app_file="app.py",
            app_file_embedding_text="local inference gradio small model artifact",
        )
        for index in range(count)
    ]
