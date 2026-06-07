from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal
from xml.etree import ElementTree


JsonType = Literal["string", "integer", "number", "boolean", "array", "object"]


@dataclass(frozen=True)
class ToolField:
    type: JsonType
    description: str
    required: bool = False
    enum: tuple[str, ...] = ()
    items_type: JsonType | None = None

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.items_type:
            schema["items"] = {"type": self.items_type}
        return schema


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fields: dict[str, ToolField]

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        name: field.to_schema() for name, field in self.fields.items()
                    },
                    "required": [name for name, field in self.fields.items() if field.required],
                },
            },
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ToolResolution:
    status: Literal["valid", "defaulted"]
    call: ToolCall
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "call": self.call.to_dict(),
            "errors": list(self.errors),
        }


class ToolContractError(ValueError):
    pass


TOOL_SPECS: dict[str, ToolSpec] = {
    "list_projects": ToolSpec(
        name="list_projects",
        description="Read prominent Build Small project cards from the offline snapshot.",
        fields={
            "track": ToolField("string", "Optional model, goal, or topic filter."),
            "sort": ToolField("string", "Sort key.", enum=("likes", "recent", "title")),
        },
    ),
    "search_projects": ToolSpec(
        name="search_projects",
        description="Find existing Spaces that echo the user's project idea.",
        fields={"query": ToolField("string", "The user idea or topic to search.", required=True)},
    ),
    "get_project": ToolSpec(
        name="get_project",
        description="Read one project card by full Space id or slug.",
        fields={"id": ToolField("string", "Project id such as build-small-hackathon/lolaby.", required=True)},
    ),
    "find_whitespace": ToolSpec(
        name="find_whitespace",
        description="Return under-explored project regions from the offline index.",
        fields={},
    ),
    "save_idea": ToolSpec(
        name="save_idea",
        description="Write or update the current idea page.",
        fields={
            "title": ToolField("string", "Short idea title.", required=True),
            "pitch": ToolField("string", "One-sentence idea pitch.", required=True),
        },
    ),
    "score_idea": ToolSpec(
        name="score_idea",
        description="Score the current idea against the fixed hackathon rubric.",
        fields={"id": ToolField("string", "Idea id; omit to score the current idea.")},
    ),
    "compare_ideas": ToolSpec(
        name="compare_ideas",
        description="Rank the current idea board and explain tradeoffs.",
        fields={},
    ),
    "make_plan": ToolSpec(
        name="make_plan",
        description="Draft the next build steps for the current idea.",
        fields={"id": ToolField("string", "Idea id; omit to plan the current idea.")},
    ),
    "update_profile": ToolSpec(
        name="update_profile",
        description="Remember a user skill, constraint, preference, or available time.",
        fields={
            "field": ToolField(
                "string",
                "Profile field to update.",
                required=True,
                enum=("skills", "time", "preferences", "constraints"),
            ),
            "value": ToolField("string", "Profile value to remember.", required=True),
        },
    ),
    "set_goals": ToolSpec(
        name="set_goals",
        description="Change the selected goals used to bias ideation and planning.",
        fields={"goals": ToolField("array", "Goal ids to prioritize.", required=True, items_type="string")},
    ),
}


def tool_schemas() -> list[dict[str, Any]]:
    return [spec.to_schema() for spec in TOOL_SPECS.values()]


def parse_xml_tool_call(text: str) -> ToolCall:
    wrapped = f"<root>{text.strip()}</root>"
    try:
        root = ElementTree.fromstring(wrapped)
    except ElementTree.ParseError as error:
        raise ToolContractError(f"invalid XML tool call: {error}") from error

    functions = [node for node in root if node.tag == "function"]
    if len(functions) != 1:
        raise ToolContractError(f"expected exactly one function call, got {len(functions)}")
    node = functions[0]
    name = str(node.attrib.get("name") or "").strip()
    if not name:
        raise ToolContractError("function call is missing a name")
    raw_arguments = (node.text or "").strip() or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise ToolContractError(f"function arguments are not valid JSON: {error.msg}") from error
    if not isinstance(arguments, dict):
        raise ToolContractError("function arguments must be a JSON object")
    return ToolCall(name=name, arguments=arguments)


def validate_tool_call(call: ToolCall, specs: dict[str, ToolSpec] = TOOL_SPECS) -> ToolCall:
    spec = specs.get(call.name)
    if spec is None:
        raise ToolContractError(f"unknown tool: {call.name}")
    allowed = set(spec.fields)
    extra = sorted(set(call.arguments) - allowed)
    if extra:
        raise ToolContractError(f"unexpected arguments for {call.name}: {', '.join(extra)}")
    missing = sorted(name for name, field in spec.fields.items() if field.required and name not in call.arguments)
    if missing:
        raise ToolContractError(f"missing required arguments for {call.name}: {', '.join(missing)}")
    for name, value in call.arguments.items():
        field = spec.fields[name]
        _validate_value(call.name, name, value, field)
    return call


def resolve_tool_call(model_output: str, fallback_query: str = "") -> ToolResolution:
    errors: list[str] = []
    try:
        call = validate_tool_call(parse_xml_tool_call(model_output))
        return ToolResolution(status="valid", call=call, errors=())
    except ToolContractError as error:
        errors.append(str(error))

    query = fallback_query.strip()
    if query:
        call = ToolCall("search_projects", {"query": query})
    else:
        call = ToolCall("find_whitespace", {})
    return ToolResolution(status="defaulted", call=call, errors=tuple(errors))


def _validate_value(tool_name: str, field_name: str, value: Any, field: ToolField) -> None:
    if field.type == "string":
        valid = isinstance(value, str)
    elif field.type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif field.type == "number":
        valid = (isinstance(value, int | float)) and not isinstance(value, bool)
    elif field.type == "boolean":
        valid = isinstance(value, bool)
    elif field.type == "array":
        valid = isinstance(value, list)
    elif field.type == "object":
        valid = isinstance(value, dict)
    else:
        valid = False
    if not valid:
        raise ToolContractError(f"{tool_name}.{field_name} must be {field.type}")
    if field.enum and value not in field.enum:
        raise ToolContractError(f"{tool_name}.{field_name} must be one of: {', '.join(field.enum)}")
    if field.items_type and isinstance(value, list):
        for index, item in enumerate(value):
            _validate_value(tool_name, f"{field_name}[{index}]", item, ToolField(field.items_type, "array item"))
