import pytest

from hackathon_advisor.tool_contracts import (
    ToolCall,
    ToolContractError,
    parse_xml_tool_call,
    resolve_tool_call,
    tool_schemas,
    validate_tool_call,
)


def test_tool_schemas_are_model_ready() -> None:
    schemas = tool_schemas()

    assert len(schemas) >= 8
    assert schemas[0]["type"] == "function"
    assert {schema["function"]["name"] for schema in schemas} >= {
        "search_projects",
        "find_whitespace",
        "save_idea",
        "score_idea",
        "make_plan",
        "set_goals",
    }
    schema_text = str(schemas)
    assert "set_target" not in schema_text
    assert "side_quests" not in schema_text
    assert "prize" not in schema_text.lower()
    assert "badge" not in schema_text.lower()


def test_parse_and_validate_minicpm_xml_tool_call() -> None:
    call = parse_xml_tool_call('<function name="search_projects">{"query":"lullaby audio"}</function>')

    assert validate_tool_call(call) == ToolCall("search_projects", {"query": "lullaby audio"})


def test_validate_rejects_unknown_tool() -> None:
    with pytest.raises(ToolContractError, match="unknown tool"):
        validate_tool_call(ToolCall("invent_project", {}))


def test_validate_rejects_bad_argument_type() -> None:
    with pytest.raises(ToolContractError, match="search_projects.query must be string"):
        validate_tool_call(ToolCall("search_projects", {"query": 47}))


def test_validate_rejects_extra_arguments() -> None:
    with pytest.raises(ToolContractError, match="unexpected arguments"):
        validate_tool_call(ToolCall("find_whitespace", {"query": "unused"}))


def test_resolve_defaults_to_search_when_output_is_broken() -> None:
    resolution = resolve_tool_call("<function", fallback_query="offline archive")

    assert resolution.status == "defaulted"
    assert resolution.call == ToolCall("search_projects", {"query": "offline archive"})
    assert resolution.errors


def test_resolve_defaults_to_whitespace_without_query() -> None:
    resolution = resolve_tool_call("no function here")

    assert resolution.status == "defaulted"
    assert resolution.call == ToolCall("find_whitespace", {})
