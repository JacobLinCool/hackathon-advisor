"""The atlas chat engine: a two-pass, tool-grounded conversation over the idea map.

Flow per turn (the native MiniCPM5 tool protocol, run on the BASE model):

1. *Pass 1* — the model sees the chat history plus the tool schemas (injected by the
   chat template via ``tools=``) and either calls one tool or answers plain prose.
2. The call is validated and degraded through the chat-specific ladder
   (``resolve_chat_tool_call``), then executed against a fresh
   :class:`~hackathon_advisor.dashboard_repository.DashboardRepository` snapshot.
3. The full verified result streams to the UI *first* (``tool_result`` + optional
   ``map_action``) so cards and the map always carry the real numbers.
4. *Pass 2* — a compact digest (urls/ids/scores stripped, so the model cannot
   misquote what it never saw) goes back as a ``role:"tool"`` message and the model
   writes a short grounded answer at temperature 0 with NO tools injected.
   Empty results skip pass 2 entirely: a 1B narrating absent data is where it
   hallucinates, so those turns get a deterministic templated sentence instead.

The engine is UI- and app-agnostic: it depends only on a ChatRunner and a
repository factory, and every yielded event is a JSON-serializable dict.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import re
from typing import Any

from hackathon_advisor._text import clean
from hackathon_advisor.aliases import normalize_text
from hackathon_advisor.dashboard_chat_contracts import (
    ChatToolResolution,
    chat_tool_schemas,
    data_intent_call,
    resolve_chat_tool_call,
    smalltalk_intent,
    strip_function_blocks,
)
from hackathon_advisor.dashboard_repository import DashboardRepository
from hackathon_advisor.model_runtime import ChatRunner
from hackathon_advisor.tool_contracts import ToolCall

# One generous budget for every chat generation: with thinking enabled the model
# reasons inside <think>...</think> before the tool call / answer, and the trace
# alone can run long. The model stops at EOS well before the cap on normal turns.
MAX_CHAT_GENERATION_TOKENS = 4096
MAX_HISTORY_MESSAGES = 12  # six user/assistant turns
MAX_ANSWER_HISTORY_MESSAGES = 4  # two turns of context for the prose passes
MAX_HISTORY_MESSAGE_CHARS = 600

CHAT_PLANNING_PROMPT = (
    "You are the Atlas Guide for the Build Small hackathon idea map. "
    "You cannot see the atlas directly: the ONLY way to answer a question about projects, "
    "clusters, quests, teams, or recent activity is to call one of the provided tools, which "
    "read the live atlas. For any such question respond with exactly one tool call and no "
    'other text, for example: <function name="search_projects"><param name="query">voice'
    "</param></function>. Reply in plain prose only for greetings or questions about yourself."
)

CHAT_ANSWER_PROMPT = (
    "You are the Atlas Guide for the Build Small hackathon idea map. "
    "Write a short conversational answer to the user's question using ONLY the facts in the "
    "tool response. Quote counts and names exactly as given. Do not invent projects, numbers, "
    "or links. Do not enumerate every item: summarize, naming at most three examples. "
    "Two to four sentences, no lists, no markdown."
)

CHAT_SMALLTALK_PROMPT = (
    "You are the Atlas Guide for the Build Small hackathon idea map. "
    "Reply briefly and warmly. You have NO project data in this conversation: never state "
    "project names, counts, likes, or rankings, and do not defend earlier numbers — if asked "
    "about data, say you should look it up and suggest asking what everyone is building, "
    "which projects completed the most quests, or what clusters exist. One or two sentences."
)

# Keys stripped from the model-facing digest. The UI renders links and ids from the
# verified payload; the model only needs labels, titles, and counts.
_DIGEST_DROPPED_KEYS = frozenset({"url", "id", "score", "host", "quest_ids"})

# A trailing fragment of "<function" left by a max_new_tokens cut mid-marker.
_PARTIAL_TAG_RE = re.compile(r"<[a-z]{0,8}$")

THINK_END_MARKER = "</think>"


class _ThinkSplitter:
    """Incrementally split a thinking-mode stream into (kind, text) chunks.

    With ``enable_thinking`` the chat template ends the prompt with ``<think>\\n``,
    so the generation is reasoning text up to ``</think>`` followed by the real
    content. The marker can arrive split across stream pieces, so a small tail
    buffer is kept until it can no longer be a marker prefix. When the runner does
    not think (rules backend) every piece passes straight through as answer text."""

    def __init__(self, active: bool) -> None:
        self._thinking = bool(active)
        self._buffer = ""

    def feed(self, piece: str) -> list[tuple[str, str]]:
        if not self._thinking:
            return [("answer", piece)] if piece else []
        self._buffer += piece
        marker = self._buffer.find(THINK_END_MARKER)
        if marker >= 0:
            thought = self._buffer[:marker]
            rest = self._buffer[marker + len(THINK_END_MARKER) :].lstrip("\n")
            self._buffer = ""
            self._thinking = False
            chunks: list[tuple[str, str]] = []
            if thought:
                chunks.append(("thinking", thought))
            if rest:
                chunks.append(("answer", rest))
            return chunks
        keep = _marker_prefix_length(self._buffer)
        flush, self._buffer = (
            self._buffer[: len(self._buffer) - keep],
            self._buffer[len(self._buffer) - keep :],
        )
        return [("thinking", flush)] if flush else []

    def finish(self) -> list[tuple[str, str]]:
        """Flush the tail when the stream ends mid-thought (max_new_tokens cut)."""
        if self._thinking and self._buffer:
            tail, self._buffer = self._buffer, ""
            return [("thinking", tail)]
        return []


def _marker_prefix_length(text: str) -> int:
    for length in range(min(len(text), len(THINK_END_MARKER) - 1), 0, -1):
        if THINK_END_MARKER.startswith(text[-length:]):
            return length
    return 0


class DashboardChatEngine:
    def __init__(
        self,
        runner: ChatRunner,
        repository_factory: Callable[[], DashboardRepository],
    ) -> None:
        self.runner = runner
        self.repository_factory = repository_factory

    def turn_stream(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        history = _normalize_history(history)
        normalized, corrections = normalize_text(message)
        yield {
            "type": "start",
            "normalized_text": normalized,
            "corrections": [correction.to_dict() for correction in corrections],
        }
        repository = self.repository_factory()

        yield {"type": "stage", "stage": "planning", "label": "Reading the atlas"}
        resolution, raw_output = yield from self._pick_tool(normalized, history)
        if resolution.status == "none":
            # Accuracy backstop: when the model answers in prose (declining its tools),
            # route any substantive question to a tool — a matched intent first, BM25
            # search otherwise. Only greetings/meta/short follow-ups may stay on the
            # ungrounded small-talk path; this is a data surface, and letting a question
            # like "how many voice apps" through is how invented facts reach the user.
            intent = data_intent_call(normalized)
            if intent is None and not smalltalk_intent(normalized):
                intent = ToolCall("search_projects", {"query": normalized})
            if intent is not None:
                resolution = ChatToolResolution(
                    status="defaulted",
                    call=intent,
                    errors=("model answered without a tool; routed by intent",),
                )
        yield {
            "type": "tool_call",
            "name": resolution.call.name if resolution.call else "",
            "arguments": resolution.call.arguments if resolution.call else {},
            "status": resolution.status,
            "errors": list(resolution.errors),
        }

        if resolution.status == "none":
            response = yield from self._smalltalk(normalized, history, raw_output)
            yield self._done(normalized, history, response, tool="", data={}, map_action=None)
            return

        call = resolution.call
        assert call is not None
        yield {
            "type": "stage",
            "stage": "running_tool",
            "tool": call.name,
            "label": f"Calling {call.name}",
        }
        # _execute may swap the tool (show_project falls back to search when no
        # project matches), so the executed name drives rendering from here on.
        tool_name, data, map_action, empty_reason = self._execute(call, repository)
        yield {"type": "tool_result", "tool": tool_name, "data": data, "map_action": map_action}

        if empty_reason:
            response = _templated_sentence(call, data, empty_reason)
            yield {"type": "answer_skipped", "reason": empty_reason, "text": response}
        else:
            yield {"type": "stage", "stage": "writing", "label": "Writing the answer"}
            executed = ToolCall(tool_name, call.arguments)
            response = yield from self._grounded_answer(normalized, history, executed, data)
            if not response:
                response = _templated_sentence(call, data, "empty_answer")
                yield {"type": "answer_skipped", "reason": "empty_answer", "text": response}

        yield self._done(
            normalized, history, response, tool=tool_name, data=data, map_action=map_action
        )

    def _pick_tool(
        self,
        message: str,
        history: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        messages = [
            {"role": "system", "content": CHAT_PLANNING_PROMPT},
            *history,
            {"role": "user", "content": message},
        ]
        splitter = _ThinkSplitter(getattr(self.runner, "supports_thinking", False))
        answer_pieces: list[str] = []
        for count, piece in self.runner.stream(
            messages,
            tools=chat_tool_schemas(),
            max_new_tokens=MAX_CHAT_GENERATION_TOKENS,
            enable_thinking=True,
        ):
            for kind, text in splitter.feed(piece):
                if kind == "thinking":
                    yield {"type": "thinking", "pass": 1, "text": text}
                else:
                    answer_pieces.append(text)
            yield {
                "type": "model_progress",
                "pass": 1,
                "tokens": count,
                "max_tokens": MAX_CHAT_GENERATION_TOKENS,
            }
        for _kind, text in splitter.finish():
            yield {"type": "thinking", "pass": 1, "text": text}
        # Only the post-thinking text may be parsed: the reasoning trace legitimately
        # talks about <function ...> syntax without being a call.
        raw_output = "".join(answer_pieces).strip()
        return resolve_chat_tool_call(raw_output, fallback_query=message), raw_output

    def _smalltalk(
        self,
        message: str,
        history: list[dict[str, Any]],
        raw_output: str,
    ) -> Iterator[dict[str, Any]]:
        """Dedicated no-tools generation: the pass-1 output is tuned for tool
        selection, not for a satisfying greeting, so chit-chat gets its own pass."""
        yield {"type": "stage", "stage": "writing", "label": "Writing the answer"}
        messages = [
            {"role": "system", "content": CHAT_SMALLTALK_PROMPT},
            *_answer_history(history),
            {"role": "user", "content": message},
        ]
        response = yield from self._stream_prose(messages, MAX_CHAT_GENERATION_TOKENS)
        if not response:
            response = strip_function_blocks(raw_output) or (
                "Hello! Ask me what everyone is building, which projects completed the most "
                "quests, or what clusters exist."
            )
            yield {"type": "answer_skipped", "reason": "empty_answer", "text": response}
        return response

    def _grounded_answer(
        self,
        message: str,
        history: list[dict[str, Any]],
        call: ToolCall,
        data: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        digest = render_digest(_digest_for_model(call.name, data))
        # NO history here: every fact the answer needs is in the digest, and a greedy
        # 1B echoes similar-sounding lines from prior turns over the digest in front
        # of it. Conversation context only matters for pass-1's tool choice.
        messages = [
            {"role": "system", "content": CHAT_ANSWER_PROMPT},
            {"role": "user", "content": message},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": call.name, "arguments": call.arguments}],
            },
            {"role": "tool", "content": digest},
        ]
        return (yield from self._stream_prose(messages, MAX_CHAT_GENERATION_TOKENS))

    def _stream_prose(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> Iterator[dict[str, Any]]:
        """Stream a no-tools generation as thinking + token events; returns the prose.

        The reasoning trace streams as ``thinking`` events; only the post-think text
        becomes the answer. If a stray ``<function`` shows up in the answer the stream
        stops early; the ``done`` response carries the stripped text, which the UI
        treats as authoritative."""
        splitter = _ThinkSplitter(getattr(self.runner, "supports_thinking", False))
        pieces: list[str] = []
        stream = self.runner.stream(messages, max_new_tokens=max_new_tokens, enable_thinking=True)
        stray_function = False
        try:
            for count, piece in stream:
                for kind, text in splitter.feed(piece):
                    if kind == "thinking":
                        yield {"type": "thinking", "pass": 2, "text": text}
                        continue
                    pieces.append(text)
                    if "<function" in "".join(pieces[-4:]):
                        stray_function = True
                        break
                    yield {"type": "token", "text": text}
                if stray_function:
                    break
                yield {
                    "type": "model_progress",
                    "pass": 2,
                    "tokens": count,
                    "max_tokens": max_new_tokens,
                }
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        for _kind, text in splitter.finish():
            yield {"type": "thinking", "pass": 2, "text": text}
        text = "".join(pieces)
        marker = text.find("<function")
        if marker >= 0:
            text = text[:marker]
        # A generation cut at max_new_tokens can end mid-marker ("<fun"); drop any
        # trailing partial tag so it never reaches the authoritative response.
        text = _PARTIAL_TAG_RE.sub("", text)
        return clean(strip_function_blocks(text))

    def _execute(
        self,
        call: ToolCall,
        repository: DashboardRepository,
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None, str]:
        """Run one validated tool; returns (executed tool, data, map action, empty reason)."""
        name = call.name
        if name == "atlas_overview":
            data = repository.overview()
            return name, data, {"type": "clear_filters"}, ""
        if name == "list_clusters":
            data = repository.list_clusters()
            return name, data, None, "" if data["clusters"] else "no_clusters"
        if name == "show_cluster":
            label = clean(call.arguments.get("label"))
            detail = repository.cluster_detail(label)
            if detail is None:
                return name, {"requested_label": label}, None, "unknown_cluster"
            return name, detail, {"type": "filter_cluster", "label": detail["label"]}, ""
        if name == "list_quests":
            data = repository.list_quests()
            if data["status"] != "analyzed":
                return name, data, None, "quests_not_analyzed"
            return name, data, None, ""
        if name == "show_quest":
            quest = clean(call.arguments.get("quest"))
            detail = repository.quest_detail(quest)
            if detail is None:
                return name, {"requested_quest": quest}, None, "unknown_quest"
            if detail["status"] != "analyzed":
                return name, detail, None, "quests_not_analyzed"
            map_action = {"type": "filter_quest", "quest": detail["id"]}
            if detail["project_count"] == 0:
                return name, detail, map_action, "quest_no_projects"
            return name, detail, map_action, ""
        if name == "show_project":
            requested = clean(call.arguments.get("project"))
            detail = repository.project_detail(requested)
            if detail is None:
                # Half-remembered names still get useful cards: fall back to search.
                return self._search(repository, requested)
            return name, detail, {"type": "highlight_projects", "ids": [detail["id"]]}, ""
        if name == "top_projects_by_quests":
            data = repository.top_by_quests()
            if data["status"] != "analyzed":
                return name, data, None, "quests_not_analyzed"
            if not data["rows"]:
                return name, data, None, "no_leaderboard_rows"
            ids = [row["id"] for row in data["rows"]]
            return name, data, {"type": "highlight_projects", "ids": ids}, ""
        if name == "search_projects":
            return self._search(repository, clean(call.arguments.get("query")))
        if name == "recent_activity":
            data = repository.recent_activity()
            if not data["projects"]:
                return name, data, None, "no_projects"
            ids = [project["id"] for project in data["projects"]]
            return name, data, {"type": "highlight_projects", "ids": ids}, ""
        # Unreachable for validated calls; degrade to a safe overview.
        return "atlas_overview", repository.overview(), None, ""

    def _search(
        self,
        repository: DashboardRepository,
        query: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None, str]:
        data = repository.search(query)
        if not data["results"]:
            return "search_projects", data, None, "no_search_results"
        ids = [result["id"] for result in data["results"]]
        return (
            "search_projects",
            data,
            {"type": "highlight_projects", "ids": ids, "query": query},
            "",
        )

    def _done(
        self,
        message: str,
        history: list[dict[str, Any]],
        response: str,
        *,
        tool: str,
        data: dict[str, Any],
        map_action: dict[str, Any] | None,
    ) -> dict[str, Any]:
        new_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return {
            "type": "done",
            "response": response,
            "tool": tool,
            "data": data,
            "map_action": map_action,
            "history": _normalize_history(new_history),
        }


def _normalize_history(history: Any) -> list[dict[str, Any]]:
    """Keep only well-formed prior prose turns, clipped, deduplicated, and capped.

    Tool digests are deliberately dropped from history: stale counts must never
    leak into a later answer — every turn re-reads a fresh repository snapshot.
    Repeated assistant sentences are collapsed too: a greedy 1B that sees the
    same line twice in history will echo it a third time regardless of the
    digest in front of it."""
    if not isinstance(history, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = clean(item.get("content"))
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content[:MAX_HISTORY_MESSAGE_CHARS]})
    return _dedupe_assistant_echoes(cleaned)[-MAX_HISTORY_MESSAGES:]


def _dedupe_assistant_echoes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive identical assistant answers, keeping the NEWEST turn.

    Walks backwards so the latest user/assistant pair always survives; the older
    repeats (and the user turns that elicited them) are dropped."""
    deduped_reversed: list[dict[str, Any]] = []
    previous_assistant = None
    skip_next_user = False
    for item in reversed(messages):
        if item["role"] == "assistant":
            if item["content"] == previous_assistant:
                skip_next_user = True
                continue
            previous_assistant = item["content"]
            deduped_reversed.append(item)
        else:
            if skip_next_user:
                skip_next_user = False
                continue
            deduped_reversed.append(item)
    return list(reversed(deduped_reversed))


def _answer_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The short tail of history given to answer generations.

    Facts come from the digest, not from history; the prose passes only need
    enough context for follow-ups, and a longer tail mostly adds echo bait."""
    return history[-MAX_ANSWER_HISTORY_MESSAGES:]


def _digest_for_model(tool: str, data: dict[str, Any]) -> Any:
    """Compact the verified payload into what the model may safely restate.

    Beyond stripping urls/ids/scores, long listings are trimmed per tool: a 1B asked
    to repeat ten labels starts blending them, so it only sees the few it may name.
    The UI renders the FULL verified payload independently."""
    trimmed: dict[str, Any] = dict(data)
    if tool == "atlas_overview":
        # Self-describing keys, most-liked first: with three lists in one digest a 1B
        # answering "what's the coolest project" otherwise grabs the wrong column.
        trimmed = {
            "most_liked_projects": data.get("most_liked"),
            "project_count": data.get("project_count"),
            "cluster_count": data.get("cluster_count"),
            "largest_clusters": data.get("top_clusters"),
            "most_completed_quests": data.get("top_quests"),
            "quest_status": data.get("quest_status"),
        }
    if tool == "list_clusters":
        # Ten compound labels is past what a 1B can restate without blending them;
        # it gets the count and the largest cluster, the cards carry the full list.
        clusters = data.get("clusters") or []
        trimmed = {
            "cluster_count": data.get("cluster_count"),
            "largest_cluster": clusters[0] if clusters else None,
            "note": "the full cluster list is already shown to the user as cards",
        }
    if tool == "list_quests":
        quests = data.get("quests") or []
        trimmed = {
            "status": data.get("status"),
            "quest_count": len(quests),
            "most_completed_quest": quests[0] if quests else None,
            "note": "the full quest list is already shown to the user as cards",
        }
    if tool == "show_cluster":
        trimmed["examples"] = (data.get("examples") or [])[:3]
    if tool == "show_quest":
        trimmed["examples"] = (data.get("examples") or [])[:3]
    if tool == "search_projects":
        # BM25 "total" counts any term overlap; quoting it as "N projects about X"
        # would mislead, so the model only sees the close matches themselves.
        trimmed.pop("total", None)
    return _strip_digest_keys(trimmed)


def _strip_digest_keys(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            key: _strip_digest_keys(value)
            for key, value in data.items()
            if key not in _DIGEST_DROPPED_KEYS
        }
    if isinstance(data, list):
        return [_strip_digest_keys(item) for item in data]
    return data


def render_digest(data: Any, indent: int = 0) -> str:
    """Render the digest as plain ``key: value`` lines instead of JSON.

    A 1B model copying labels out of nested JSON starts blending adjacent strings;
    one fact per line keeps its quotes literal."""
    return "\n".join(_digest_lines(data, indent))


def _digest_lines(value: Any, indent: int) -> list[str]:
    pad = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_digest_lines(item, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_digest_value(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                flat = ", ".join(
                    f"{key}: {_digest_value(entry)}"
                    for key, entry in item.items()
                    if not isinstance(entry, (dict, list))
                )
                lines.append(f"{pad}- {flat}")
            else:
                lines.append(f"{pad}- {_digest_value(item)}")
        return lines
    return [f"{pad}{_digest_value(value)}"]


def _digest_value(value: Any) -> Any:
    # Quote strings so compound labels like "Dream / Oracle" keep hard copy
    # boundaries — a greedy 1B blends adjacent unquoted multi-word labels.
    if isinstance(value, str):
        return f'"{value}"'
    return value


def _templated_sentence(call: ToolCall, data: dict[str, Any], reason: str) -> str:
    """Deterministic sentences for the turns where the model must not improvise."""
    if reason == "quests_not_analyzed":
        return (
            "Quest analysis has not run for this snapshot yet, so quest coverage is empty. "
            "Refresh the map to classify the field, or ask about clusters and projects instead."
        )
    if reason == "unknown_cluster":
        requested = clean(data.get("requested_label")) or "that name"
        return (
            f"I could not find a cluster matching {requested} in the current snapshot. "
            "Ask me to list the clusters to see the live labels."
        )
    if reason == "unknown_quest":
        requested = clean(data.get("requested_quest")) or "that name"
        return (
            f"I could not match {requested} to a hackathon quest. "
            "Ask me to list the quests to see the official names."
        )
    if reason == "quest_no_projects":
        label = clean(data.get("label"))
        if label:
            return f"No project in the current snapshot has completed {label} yet."
        return "No project in the current snapshot has completed that quest yet."
    if reason == "no_leaderboard_rows":
        return (
            "Quest analysis ran, but no project in the current snapshot has completed a "
            "quest yet — the leaderboard is empty."
        )
    if reason == "no_search_results":
        query = clean(data.get("query")) or "that"
        return (
            f"The atlas has no match for {query}. "
            "That can be good news for originality — try a broader term to double-check."
        )
    if reason == "no_clusters" or reason == "no_projects":
        return "The current snapshot has no data for that yet. Try refreshing the map."
    if reason == "empty_answer":
        return "The verified results are on the cards below."
    return "The verified results are on the cards below."
