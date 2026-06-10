import pytest

from hackathon_advisor.dashboard_chat_contracts import (
    CHAT_TOOL_SPECS,
    chat_tool_schemas,
    heuristic_chat_call,
    parse_native_tool_call,
    resolve_chat_tool_call,
    strip_function_blocks,
)
from hackathon_advisor.tool_contracts import ToolContractError


def test_chat_tool_schemas_are_openai_style() -> None:
    schemas = chat_tool_schemas()

    assert len(schemas) == len(CHAT_TOOL_SPECS) == 9
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"


def test_parse_native_tool_call_reads_params() -> None:
    call = parse_native_tool_call(
        '<function name="search_projects"><param name="query">voice agents</param></function>'
    )

    assert call.name == "search_projects"
    assert call.arguments == {"query": "voice agents"}


def test_parse_native_tool_call_handles_cdata_and_surrounding_prose() -> None:
    call = parse_native_tool_call(
        'Let me look that up.\n<function name="show_cluster">'
        "<param name=\"label\"><![CDATA[Voice / Chatbot & ASR]]></param></function> Done."
    )

    assert call.name == "show_cluster"
    assert call.arguments == {"label": "Voice / Chatbot & ASR"}


def test_parse_native_tool_call_without_params() -> None:
    call = parse_native_tool_call('<function name="list_quests"></function>')

    assert call.name == "list_quests"
    assert call.arguments == {}


def test_parse_native_tool_call_rejects_garbage() -> None:
    with pytest.raises(ToolContractError):
        parse_native_tool_call('<function name="x"><param name="q">unclosed</function>')
    with pytest.raises(ToolContractError):
        parse_native_tool_call("just prose, no call")
    with pytest.raises(ToolContractError):
        parse_native_tool_call('<function><param name="q">missing name</param></function>')


def test_resolve_marks_plain_prose_as_none() -> None:
    resolution = resolve_chat_tool_call("Hello! Ask me about the atlas.", fallback_query="hello")

    assert resolution.status == "none"
    assert resolution.call is None
    assert resolution.errors == ()


def test_resolve_accepts_valid_call() -> None:
    resolution = resolve_chat_tool_call(
        '<function name="show_quest"><param name="quest">Tiny Titan</param></function>',
        fallback_query="tiny titan",
    )

    assert resolution.status == "valid"
    assert resolution.call is not None
    assert resolution.call.name == "show_quest"


def test_resolve_defaults_unknown_tool_through_intent_router() -> None:
    resolution = resolve_chat_tool_call(
        '<function name="made_up_tool"></function>',
        fallback_query="who completed the most quests?",
    )

    assert resolution.status == "defaulted"
    assert resolution.call is not None
    assert resolution.call.name == "top_projects_by_quests"
    assert resolution.errors


def test_resolve_defaults_malformed_xml_to_search() -> None:
    resolution = resolve_chat_tool_call(
        '<function name="search_projects"><param name="query">a < b</param></function>',
        fallback_query="projects about voice",
    )

    assert resolution.status == "defaulted"
    assert resolution.call is not None
    assert resolution.call.name == "search_projects"
    assert resolution.call.arguments["query"] == "projects about voice"


def test_resolve_rejects_missing_required_argument() -> None:
    resolution = resolve_chat_tool_call(
        '<function name="show_cluster"></function>',
        fallback_query="show me the voice cluster",
    )

    assert resolution.status == "defaulted"
    assert resolution.errors


def test_heuristic_chat_call_routes_intents() -> None:
    assert heuristic_chat_call("who completed the most quests").name == "top_projects_by_quests"
    assert heuristic_chat_call("what clusters exist").name == "list_clusters"
    assert heuristic_chat_call("list the quests please").name == "list_quests"
    assert heuristic_chat_call("what changed recently").name == "recent_activity"
    assert heuristic_chat_call("what is everyone building").name == "atlas_overview"
    assert heuristic_chat_call("knitting helpers").name == "search_projects"
    assert heuristic_chat_call("").name == "atlas_overview"


def test_data_intent_call_separates_detail_from_listing() -> None:
    from hackathon_advisor.dashboard_chat_contracts import data_intent_call

    detail = data_intent_call("what is in the Dream / Oracle cluster?")
    assert detail is not None and detail.name == "show_cluster"
    assert "Dream / Oracle" in detail.arguments["label"]

    quest_detail = data_intent_call("tell me about the Tiny Titan quest")
    assert quest_detail is not None and quest_detail.name == "show_quest"

    assert data_intent_call("what clusters exist").name == "list_clusters"
    assert data_intent_call("find projects about voice").name == "search_projects"
    assert data_intent_call("how many voice apps").name == "search_projects"
    assert data_intent_call("what is the coolest project?").name == "atlas_overview"
    assert data_intent_call("which project is most liked").name == "atlas_overview"
    assert data_intent_call("hello there") is None
    assert data_intent_call("thanks!") is None


def test_data_intent_call_routes_project_reading() -> None:
    from hackathon_advisor.dashboard_chat_contracts import data_intent_call

    readme = data_intent_call("read the readme of Jawbreaker")
    assert readme is not None and readme.name == "show_project"

    how = data_intent_call("how does Jawbreaker work?")
    assert how is not None and how.name == "show_project"

    about = data_intent_call("tell me about Jawbreaker")
    assert about is not None and about.name == "show_project"
    assert about.arguments["project"] == "tell me about Jawbreaker"

    # cluster/quest detail intents keep their own tools
    assert data_intent_call("tell me about the Tiny Titan quest").name == "show_quest"
    assert data_intent_call("what is in the Dream / Oracle cluster?").name == "show_cluster"


def test_smalltalk_intent_only_matches_greetings_and_followups() -> None:
    from hackathon_advisor.dashboard_chat_contracts import smalltalk_intent

    assert smalltalk_intent("hello!")
    assert smalltalk_intent("hey there")
    assert smalltalk_intent("thanks")
    assert smalltalk_intent("why")
    assert smalltalk_intent("are you sure?")
    assert smalltalk_intent("who are you")
    assert smalltalk_intent("")

    assert not smalltalk_intent("how many voice apps")
    assert not smalltalk_intent("what is the coolest project?")
    assert not smalltalk_intent("voice agents for elderly care")
    assert not smalltalk_intent("knitting helpers")


def test_strip_function_blocks_removes_stray_calls() -> None:
    text = 'Here you go. <function name="x"><param name="q">v</param></function> The map shows it.'

    assert strip_function_blocks(text) == "Here you go.  The map shows it.".strip()
    assert strip_function_blocks("plain prose") == "plain prose"
