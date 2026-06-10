"""Tool contracts for the atlas chat: native MiniCPM5 format, chat-specific fallback.

The dashboard chat drives the BASE MiniCPM5-1B model through its native
tool-calling protocol: tool JSON schemas go in via ``apply_chat_template(...,
tools=...)`` and the model answers either with plain prose (no tool needed) or
with one XML call of the form::

    <function name="tool_name"><param name="arg">value</param></function>

That argument encoding (``<param>`` children, CDATA for special characters)
differs from the advisor's JSON-body format in ``tool_contracts.py``, so it gets
its own parser here. Validation reuses ``validate_tool_call`` against the chat
tool specs. The degradation ladder is chat-specific: prose with no function call
is a deliberate "no tool" outcome, while a malformed call degrades through a
keyword intent router and finally to a BM25 search of the raw message — never to
the advisor's ``search_projects``/``find_whitespace`` defaults, which assume the
advisor's idea-board context.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal
from xml.etree import ElementTree

from hackathon_advisor.tool_contracts import (
    ToolCall,
    ToolContractError,
    ToolField,
    ToolSpec,
    validate_tool_call,
)

CHAT_TOOL_SPECS: dict[str, ToolSpec] = {
    "atlas_overview": ToolSpec(
        name="atlas_overview",
        description="Summarize the whole field: project totals, biggest clusters, quest coverage.",
        fields={},
    ),
    "list_clusters": ToolSpec(
        name="list_clusters",
        description="List the project clusters (themes) with sizes and keywords.",
        fields={},
    ),
    "show_cluster": ToolSpec(
        name="show_cluster",
        description="Inspect one cluster by its label and show example projects.",
        fields={
            "label": ToolField("string", "Cluster label, such as Voice / Chatbot.", required=True)
        },
    ),
    "list_quests": ToolSpec(
        name="list_quests",
        description="List the hackathon quests with how many projects completed each.",
        fields={},
    ),
    "show_quest": ToolSpec(
        name="show_quest",
        description="Inspect one quest: its description, coverage, and example projects.",
        fields={
            "quest": ToolField(
                "string", "Quest name, such as Off the Grid or Tiny Titan.", required=True
            )
        },
    ),
    "show_project": ToolSpec(
        name="show_project",
        description="Read one project's README and main app file by project name.",
        fields={"project": ToolField("string", "Project name, id, or slug.", required=True)},
    ),
    "top_projects_by_quests": ToolSpec(
        name="top_projects_by_quests",
        description="Rank projects by how many quests they completed (the quest leaderboard).",
        fields={},
    ),
    "search_projects": ToolSpec(
        name="search_projects",
        description="Full-text search across all projects on the map.",
        fields={
            "query": ToolField("string", "Topic, model, or idea to search for.", required=True)
        },
    ),
    "recent_activity": ToolSpec(
        name="recent_activity",
        description="Show the most recently updated projects.",
        fields={},
    ),
}


@dataclass(frozen=True)
class ChatToolResolution:
    """Outcome of reading one pass-1 model output.

    ``none`` means the model deliberately answered without a tool (chit-chat);
    ``call`` is None only in that case.
    """

    status: Literal["valid", "defaulted", "none"]
    call: ToolCall | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "call": self.call.to_dict() if self.call else None,
            "errors": list(self.errors),
        }


def chat_tool_schemas() -> list[dict[str, Any]]:
    return [spec.to_schema() for spec in CHAT_TOOL_SPECS.values()]


_FUNCTION_BLOCK_RE = re.compile(r"<function\b.*?</function>", re.DOTALL)
_FUNCTION_OPEN_RE = re.compile(r"<function\b")


def parse_native_tool_call(text: str) -> ToolCall:
    """Extract and parse the first native-format function call in ``text``.

    The native template lets the model wrap a call in prose, so surrounding text
    is ignored; only the ``<function ...>...</function>`` block is parsed.
    """
    block = _FUNCTION_BLOCK_RE.search(text or "")
    if block is None:
        raise ToolContractError("no <function> call found in model output")
    try:
        node = ElementTree.fromstring(block.group(0))
    except ElementTree.ParseError as error:
        raise ToolContractError(f"invalid native tool call XML: {error}") from error
    name = str(node.attrib.get("name") or "").strip()
    if not name:
        raise ToolContractError("function call is missing a name")
    arguments: dict[str, Any] = {}
    for child in node:
        if child.tag != "param":
            raise ToolContractError(f"unexpected element <{child.tag}> in function call")
        param_name = str(child.attrib.get("name") or "").strip()
        if not param_name:
            raise ToolContractError("param is missing a name")
        arguments[param_name] = _element_text(child).strip()
    return ToolCall(name=name, arguments=arguments)


def resolve_chat_tool_call(model_output: str, fallback_query: str = "") -> ChatToolResolution:
    """Validate one pass-1 output, or degrade: intent router, then BM25 search."""
    text = str(model_output or "")
    if _FUNCTION_OPEN_RE.search(text) is None:
        return ChatToolResolution(status="none", call=None, errors=())

    errors: list[str] = []
    try:
        call = validate_tool_call(parse_native_tool_call(text), specs=CHAT_TOOL_SPECS)
        return ChatToolResolution(status="valid", call=call, errors=())
    except ToolContractError as error:
        errors.append(str(error))

    call = heuristic_chat_call(fallback_query)
    return ChatToolResolution(status="defaulted", call=call, errors=tuple(errors))


def data_intent_call(message: str) -> ToolCall | None:
    """Map a message with a CLEAR data intent to a tool call; None means no clear intent.

    Used as the accuracy backstop when the model answers a data-shaped question in plain
    prose: an explicit intent routes to the matching tool, anything else (greetings,
    meta questions) stays conversational."""
    lower = " ".join(str(message or "").casefold().split())
    cleaned = " ".join(str(message or "").split())
    detail_intent = _mentions(
        lower, ("what is in", "what's in", "inside", "show me the", "tell me about", "about the")
    )
    if _mentions(
        lower, ("leaderboard", "most quest", "who completed", "top project", "top team", "winning")
    ):
        return ToolCall("top_projects_by_quests", {})
    if _mentions(lower, ("cluster", "theme", "group", "region")):
        if detail_intent:
            # cluster_detail() fuzzy-resolves a label embedded in the question.
            return ToolCall("show_cluster", {"label": cleaned})
        return ToolCall("list_clusters", {})
    if _mentions(lower, ("quest", "badge", "challenge")):
        if detail_intent:
            return ToolCall("show_quest", {"quest": cleaned})
        return ToolCall("list_quests", {})
    if _mentions(lower, ("recent", "latest", "newest", "just updated", "activity")):
        return ToolCall("recent_activity", {})
    if _mentions(
        lower,
        (
            "overview",
            "everyone building",
            "everyone doing",
            "whole field",
            "summary of the",
            "most liked",
            "most popular",
            "coolest",
            "best project",
            "favorite project",
        ),
    ):
        return ToolCall("atlas_overview", {})
    if (
        _mentions(
            lower,
            ("readme", "app file", "source code", "how does", "how is", "what does", "built with"),
        )
        or detail_intent
    ):
        # project_detail() spots a title embedded in the question; the engine falls
        # back to BM25 search when no project matches.
        return ToolCall("show_project", {"project": cleaned})
    if _mentions(
        lower,
        (
            "find ",
            "search",
            "looking for",
            "projects about",
            "projects on",
            "show me",
            "anything about",
            "who is building",
            "how many",
            "number of",
            "count of",
            "is there a",
            "are there any",
        ),
    ):
        return ToolCall("search_projects", {"query": " ".join(str(message).split())})
    return None


_SMALLTALK_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "yo",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "cool",
    "nice",
    "bye",
    "goodbye",
    "why",
    "really",
    "are you sure",
    "who are you",
    "what are you",
    "what can you do",
    "how do you work",
    "help",
)


def smalltalk_intent(message: str) -> bool:
    """True only for greetings, meta questions, and short follow-ups.

    The chat is a data-exploration surface, so the safe default for anything
    substantive is a tool (BM25 search) — letting an unmatched question fall
    through to ungrounded small talk is how the model ends up inventing facts."""
    lower = " ".join(str(message or "").casefold().split()).rstrip(".!?")
    if not lower:
        return True
    # Pattern-table only: an unknown two-word phrase like "knitting helpers" is a
    # search, and an unmatched search honestly answers "no match" — never invents.
    return any(
        lower == pattern or lower.startswith(f"{pattern} ") for pattern in _SMALLTALK_PATTERNS
    )


def heuristic_chat_call(message: str) -> ToolCall:
    """Keyword intent router used when the model's tool call cannot be salvaged."""
    intent = data_intent_call(message)
    if intent is not None:
        return intent
    cleaned = " ".join(str(message or "").split())
    if cleaned:
        return ToolCall("search_projects", {"query": cleaned})
    return ToolCall("atlas_overview", {})


def strip_function_blocks(text: str) -> str:
    """Remove any stray function-call XML a pass-2 generation might emit."""
    return _FUNCTION_BLOCK_RE.sub("", str(text or "")).strip()


def _mentions(lower_text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in lower_text for phrase in phrases)


def _element_text(node: ElementTree.Element) -> str:
    return "".join(node.itertext())
