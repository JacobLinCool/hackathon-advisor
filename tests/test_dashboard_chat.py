from hackathon_advisor.dashboard_chat import DashboardChatEngine
from tests.test_dashboard_repository import analyzed_repository, not_analyzed_repository


class ScriptedRunner:
    """ChatRunner double that replays prepared model outputs and records calls."""

    backend = "scripted"
    model_id = "scripted-test-model"
    supports_thinking = False

    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def stream(self, messages, *, tools=None, max_new_tokens, enable_thinking=False):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "max_new_tokens": max_new_tokens,
                "enable_thinking": enable_thinking,
            }
        )
        output = self.outputs.pop(0)
        pieces = output if isinstance(output, list) else [output]
        for count, piece in enumerate(pieces, start=1):
            yield count, piece


class ThinkingScriptedRunner(ScriptedRunner):
    """ScriptedRunner whose outputs start inside a <think> block (native thinking)."""

    supports_thinking = True


def run_turn(runner, repository, message, history=None):
    engine = DashboardChatEngine(runner, lambda: repository)
    return list(engine.turn_stream(message, history))


def events_of(events, event_type):
    return [event for event in events if event["type"] == event_type]


def test_tool_turn_streams_verified_data_before_prose() -> None:
    runner = ScriptedRunner(
        [
            '<function name="top_projects_by_quests"></function>',
            "Project 9 leads with two quests.",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "who completed the most quests?")

    tool_call = events_of(events, "tool_call")[0]
    assert tool_call["status"] == "valid"
    assert tool_call["name"] == "top_projects_by_quests"

    tool_result = events_of(events, "tool_result")[0]
    assert tool_result["data"]["rows"][0]["id"] == "build-small-hackathon/project-9"
    assert tool_result["map_action"]["type"] == "highlight_projects"

    types = [event["type"] for event in events]
    assert types.index("tool_result") < types.index("token")

    done = events_of(events, "done")[0]
    assert done["response"] == "Project 9 leads with two quests."
    assert done["tool"] == "top_projects_by_quests"
    assert done["history"][-2:] == [
        {"role": "user", "content": "who completed the most quests?"},
        {"role": "assistant", "content": "Project 9 leads with two quests."},
    ]


def test_model_digest_strips_urls_and_ids() -> None:
    runner = ScriptedRunner(
        [
            '<function name="top_projects_by_quests"></function>',
            "Project 9 leads.",
        ]
    )

    run_turn(runner, analyzed_repository(), "quest leaderboard")

    answer_call = runner.calls[1]
    assert answer_call["tools"] is None
    digest = answer_call["messages"][-1]["content"]
    assert "url" not in digest
    assert "huggingface.co" not in digest
    assert 'title: "Project 9"' in digest
    assert "quest_count: 2" in digest


def test_model_digest_trims_long_listings_and_bm25_total() -> None:
    repository = analyzed_repository()
    runner = ScriptedRunner(
        ['<function name="search_projects"><param name="query">project</param></function>', "Found a few."]
    )
    run_turn(runner, repository, "find project planners")
    search_digest = runner.calls[1]["messages"][-1]["content"]
    assert "total:" not in search_digest

    runner = ScriptedRunner(['<function name="list_clusters"></function>', "A few clusters."])
    run_turn(runner, repository, "what clusters exist")
    cluster_digest = runner.calls[1]["messages"][-1]["content"]
    # Only the largest cluster is restatable; the full list lives on the UI cards.
    assert cluster_digest.count("label:") == 1
    assert f"cluster_count: {repository.list_clusters()['cluster_count']}" in cluster_digest


def test_not_analyzed_quests_skip_the_answer_pass() -> None:
    runner = ScriptedRunner(['<function name="top_projects_by_quests"></function>'])

    events = run_turn(runner, not_analyzed_repository(), "who completed the most quests?")

    skipped = events_of(events, "answer_skipped")[0]
    assert skipped["reason"] == "quests_not_analyzed"
    assert events_of(events, "token") == []
    assert len(runner.calls) == 1  # pass 2 never ran
    assert events_of(events, "done")[0]["response"] == skipped["text"]


def test_empty_search_skips_the_answer_pass() -> None:
    runner = ScriptedRunner(
        ['<function name="search_projects"><param name="query">zzzz qqqq</param></function>']
    )

    events = run_turn(runner, analyzed_repository(), "find zzzz qqqq")

    skipped = events_of(events, "answer_skipped")[0]
    assert skipped["reason"] == "no_search_results"
    assert "zzzz qqqq" in skipped["text"]
    assert len(runner.calls) == 1


def test_unknown_cluster_uses_templated_sentence() -> None:
    runner = ScriptedRunner(
        ['<function name="show_cluster"><param name="label">flying castles</param></function>']
    )

    events = run_turn(runner, analyzed_repository(), "show me the flying castles cluster")

    skipped = events_of(events, "answer_skipped")[0]
    assert skipped["reason"] == "unknown_cluster"
    assert "flying castles" in skipped["text"]
    assert events_of(events, "tool_result")[0]["map_action"] is None


def test_show_cluster_emits_filter_map_action() -> None:
    repository = analyzed_repository()
    label = repository.list_clusters()["clusters"][0]["label"]
    runner = ScriptedRunner(
        [
            f'<function name="show_cluster"><param name="label">{label}</param></function>',
            "That cluster groups the local-first planners.",
        ]
    )

    events = run_turn(runner, repository, f"what is in {label}?")

    map_action = events_of(events, "tool_result")[0]["map_action"]
    assert map_action == {"type": "filter_cluster", "label": label}


def test_show_cluster_map_action_carries_canonical_label_for_fuzzy_input() -> None:
    """The UI resolves filter_cluster by exact label match, so the engine must
    forward the repository's canonical label, never the user's fuzzy input."""
    repository = analyzed_repository()
    canonical = repository.list_clusters()["clusters"][0]["label"]
    fuzzy = canonical.split()[0].lower()
    assert fuzzy != canonical
    runner = ScriptedRunner(
        [
            f'<function name="show_cluster"><param name="label">{fuzzy}</param></function>',
            "Here is that cluster.",
        ]
    )

    events = run_turn(runner, repository, f"what is in the {fuzzy} cluster?")

    map_action = events_of(events, "tool_result")[0]["map_action"]
    assert map_action == {"type": "filter_cluster", "label": canonical}


def test_plain_prose_routes_to_dedicated_smalltalk_pass() -> None:
    runner = ScriptedRunner(
        [
            "Hello! Happy to help.",
            "Hi! Ask me what everyone is building.",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "hello there")

    assert events_of(events, "tool_call")[0]["status"] == "none"
    assert events_of(events, "tool_result") == []
    done = events_of(events, "done")[0]
    assert done["tool"] == ""
    assert done["response"] == "Hi! Ask me what everyone is building."
    assert runner.calls[1]["tools"] is None


def test_unmatched_data_question_defaults_to_search_not_smalltalk() -> None:
    """Regression: 'how many voice apps' once slipped into ungrounded small talk."""
    runner = ScriptedRunner(
        [
            "I'm not sure, but I can provide more information if you'd like.",
            "The closest matches are shown on the cards.",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "how many voice apps")

    tool_call = events_of(events, "tool_call")[0]
    assert tool_call["status"] == "defaulted"
    assert tool_call["name"] == "search_projects"
    assert tool_call["arguments"]["query"] == "how many voice apps"


def test_short_followup_stays_on_smalltalk_path() -> None:
    runner = ScriptedRunner(
        [
            "Because it scored well.",
            "I should look that up rather than guess - ask me about the quest leaderboard!",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "are you sure?")

    assert events_of(events, "tool_call")[0]["status"] == "none"
    assert events_of(events, "tool_result") == []


def test_show_project_streams_readme_and_highlights_the_dot() -> None:
    runner = ScriptedRunner(
        [
            '<function name="show_project"><param name="project">Project 4</param></function>',
            "Project 4 is an offline planner; its app file loads gradio.",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "how does Project 4 work?")

    tool_result = events_of(events, "tool_result")[0]
    assert tool_result["tool"] == "show_project"
    assert "README evidence for project 4" in tool_result["data"]["readme_excerpt"]
    assert tool_result["map_action"] == {
        "type": "highlight_projects",
        "ids": ["build-small-hackathon/project-4"],
    }
    digest = runner.calls[1]["messages"][-1]["content"]
    assert "README evidence for project 4" in digest
    assert "import gradio" in digest
    assert events_of(events, "done")[0]["tool"] == "show_project"


def test_show_project_falls_back_to_search_for_unknown_names() -> None:
    runner = ScriptedRunner(
        [
            '<function name="show_project"><param name="project">offline planner thing</param></function>',
            "The closest matches are on the cards.",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "tell me about the offline planner thing")

    tool_result = events_of(events, "tool_result")[0]
    assert tool_result["tool"] == "search_projects"
    assert tool_result["data"]["results"]
    assert events_of(events, "done")[0]["tool"] == "search_projects"


def test_prose_answer_to_data_question_is_routed_by_intent() -> None:
    runner = ScriptedRunner(
        [
            "I do not have access to quest completion data.",
            "Project 9 leads with two quests.",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "who completed the most quests?")

    tool_call = events_of(events, "tool_call")[0]
    assert tool_call["status"] == "defaulted"
    assert tool_call["name"] == "top_projects_by_quests"
    assert events_of(events, "tool_result")[0]["data"]["rows"]
    assert events_of(events, "done")[0]["response"] == "Project 9 leads with two quests."


def test_malformed_call_degrades_through_intent_router() -> None:
    runner = ScriptedRunner(
        [
            '<function name="nonexistent_tool"></function>',
            "Project 9 leads the quest board.",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "who completed the most quests")

    tool_call = events_of(events, "tool_call")[0]
    assert tool_call["status"] == "defaulted"
    assert tool_call["name"] == "top_projects_by_quests"
    assert tool_call["errors"]


def test_stray_function_block_is_cut_from_the_answer() -> None:
    runner = ScriptedRunner(
        [
            '<function name="atlas_overview"></function>',
            ["The field has ten projects. ", '<function name="x">', "</function> extra"],
        ]
    )

    events = run_turn(runner, analyzed_repository(), "overview please")

    done = events_of(events, "done")[0]
    assert done["response"] == "The field has ten projects."
    for token in events_of(events, "token"):
        assert "<function" not in token["text"]


def test_thinking_trace_streams_separately_from_the_tool_call() -> None:
    runner = ThinkingScriptedRunner(
        [
            [
                "The user wants the leaderboard. I could emit ",
                '<function name="x"> here but the right tool is top_projects_by_quests.',
                "</th",
                'ink>\n\n<function name="top_projects_by_quests"></function>',
            ],
            ["Let me restate the digest.</think>\n\nProject 9 leads with two quests."],
        ]
    )

    events = run_turn(runner, analyzed_repository(), "who completed the most quests?")

    thinking_pass1 = [e for e in events if e["type"] == "thinking" and e["pass"] == 1]
    assert "".join(e["text"] for e in thinking_pass1).startswith("The user wants the leaderboard")
    # <function inside the REASONING must not be mistaken for the call itself.
    tool_call = events_of(events, "tool_call")[0]
    assert tool_call["status"] == "valid"
    assert tool_call["name"] == "top_projects_by_quests"

    thinking_pass2 = [e for e in events if e["type"] == "thinking" and e["pass"] == 2]
    assert "".join(e["text"] for e in thinking_pass2) == "Let me restate the digest."
    done = events_of(events, "done")[0]
    assert done["response"] == "Project 9 leads with two quests."
    for token in events_of(events, "token"):
        assert "</think>" not in token["text"]
    # Thinking never leaks into the durable history.
    assert all("Let me restate" not in entry["content"] for entry in done["history"])


def test_thinking_marker_split_across_pieces_is_handled() -> None:
    runner = ThinkingScriptedRunner(
        [
            ["step one ", "step two</t", "hink>", '\n\n<function name="list_quests"></function>'],
            ["ok</think>\n\nThe quests are on the cards."],
        ]
    )

    events = run_turn(runner, analyzed_repository(), "list the quests")

    thinking = "".join(e["text"] for e in events if e["type"] == "thinking" and e["pass"] == 1)
    assert thinking == "step one step two"
    assert events_of(events, "tool_call")[0]["name"] == "list_quests"


def test_truncated_thinking_degrades_gracefully() -> None:
    """A 4096-token cut inside <think> leaves no answer text; the turn must still
    resolve (intent backstop on pass 1, templated sentence on pass 2)."""
    runner = ThinkingScriptedRunner(
        [
            ["I am still reasoning about which tool to"],  # no </think>, no call
            ["and the digest says</think>\n\nProject 9 leads."],
        ]
    )

    events = run_turn(runner, analyzed_repository(), "who completed the most quests?")

    tool_call = events_of(events, "tool_call")[0]
    assert tool_call["status"] == "defaulted"
    assert tool_call["name"] == "top_projects_by_quests"
    assert events_of(events, "done")[0]["response"] == "Project 9 leads."


def test_chat_generations_use_thinking_and_4096_budget() -> None:
    runner = ScriptedRunner(
        [
            '<function name="atlas_overview"></function>',
            "Ten projects in view.",
        ]
    )

    events = run_turn(runner, analyzed_repository(), "overview")

    for call in runner.calls:
        assert call["enable_thinking"] is True
        assert call["max_new_tokens"] >= 4096
    for progress in events_of(events, "model_progress"):
        assert progress["max_tokens"] >= 4096


def test_history_drops_repeated_assistant_answers() -> None:
    """Regression: a greedy 1B echoes any sentence that appears twice in history."""
    runner = ScriptedRunner(
        [
            '<function name="atlas_overview"></function>',
            "Ten projects in view.",
        ]
    )
    looping_history = [
        {"role": "user", "content": "why"},
        {"role": "assistant", "content": "I should look that up."},
        {"role": "user", "content": "are you sure?"},
        {"role": "assistant", "content": "I should look that up."},
        {"role": "user", "content": "really?"},
        {"role": "assistant", "content": "I should look that up."},
    ]

    run_turn(runner, analyzed_repository(), "overview", looping_history)

    pass1_messages = runner.calls[0]["messages"]
    repeated = [m for m in pass1_messages if m.get("content") == "I should look that up."]
    assert len(repeated) == 1


def test_grounded_answer_pass_sees_no_history() -> None:
    """Echoes regression: with history in the prompt, a greedy 1B repeats prior
    answers instead of reading the digest. Facts come from the digest alone."""
    runner = ScriptedRunner(
        [
            '<function name="atlas_overview"></function>',
            "Ten projects in view.",
        ]
    )
    long_history = []
    for index in range(6):
        long_history.append({"role": "user", "content": f"question {index}"})
        long_history.append({"role": "assistant", "content": f"answer {index}"})

    run_turn(runner, analyzed_repository(), "overview", long_history)

    answer_messages = runner.calls[1]["messages"]
    roles = [m["role"] for m in answer_messages]
    assert roles == ["system", "user", "assistant", "tool"]


def test_overview_digest_leads_with_most_liked_projects() -> None:
    runner = ScriptedRunner(
        [
            '<function name="atlas_overview"></function>',
            "Project 9 is the most liked.",
        ]
    )

    run_turn(runner, analyzed_repository(), "what is the coolest project?")

    digest = runner.calls[1]["messages"][-1]["content"]
    assert digest.startswith("most_liked_projects:")
    assert "most_completed_quests:" in digest


def test_history_is_normalized_and_capped() -> None:
    runner = ScriptedRunner(
        [
            '<function name="atlas_overview"></function>',
            "Ten projects in view.",
        ]
    )
    junk_history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "content": "should be dropped"},
        "not even a dict",
        {"role": "user", "content": ""},
    ]

    events = run_turn(runner, analyzed_repository(), "overview", junk_history)

    pass1_messages = runner.calls[0]["messages"]
    roles = [message["role"] for message in pass1_messages]
    assert roles == ["system", "user", "assistant", "user"]
    done = events_of(events, "done")[0]
    assert all(entry["role"] in ("user", "assistant") for entry in done["history"])
    assert len(done["history"]) <= 12
