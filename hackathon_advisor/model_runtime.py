from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol

from hackathon_advisor.tools import idea_from_text
from hackathon_advisor.tool_contracts import ToolResolution, resolve_tool_call, tool_schemas


DEFAULT_MODEL_ID = "openbmb/MiniCPM5-1B"
DEFAULT_BACKEND = "rules"


class ToolPlanner(Protocol):
    backend: str
    model_id: str

    def plan(self, message: str, state: dict[str, Any]) -> ToolResolution:
        ...


@dataclass(frozen=True)
class RuntimeStatus:
    backend: str
    model_id: str
    loaded: bool
    tool_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "loaded": self.loaded,
            "tool_count": self.tool_count,
        }


class RuleBasedPlanner:
    backend = "rules"
    model_id = "deterministic-tool-router"

    def plan(self, message: str, state: dict[str, Any]) -> ToolResolution:
        text = " ".join(message.strip().split())
        lower = text.lower()
        if not text:
            output = '<function name="list_projects">{"sort":"likes"}</function>'
        elif any(term in lower for term in ("compare", "choose", "rank")):
            output = '<function name="compare_ideas">{}</function>'
        elif any(term in lower for term in ("plan", "roadmap", "next step", "milestone")):
            output = '<function name="make_plan">{}</function>'
        elif any(term in lower for term in ("whitespace", "original", "new", "bolder", "unwritten", "gap")):
            output = '<function name="find_whitespace">{}</function>'
        elif any(term in lower for term in ("search", "similar", "already", "existing", "overlap", "echo")):
            output = f'<function name="search_projects">{{"query":{_json_string(text)}}}</function>'
        else:
            title, pitch = idea_from_text(text)
            output = (
                f'<function name="save_idea">'
                f'{{"title":{_json_string(title)},"pitch":{_json_string(pitch)}}}'
                f"</function>"
            )
        return resolve_tool_call(output, fallback_query=text)


class MiniCPMTransformersPlanner:
    backend = "minicpm-transformers"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._tokenizer = None
        self._model = None

    def plan(self, message: str, state: dict[str, Any]) -> ToolResolution:
        self._ensure_loaded()
        prompt = render_context(message, state)
        output = self._generate_tool_call(prompt)
        return resolve_tool_call(output, fallback_query=message)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "ADVISOR_MODEL_BACKEND=minicpm-transformers requires optional model dependencies. "
                "Install the model extra before enabling it."
            ) from error
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if hasattr(torch, "inference_mode"):
            self._inference_mode = torch.inference_mode

    def _generate_tool_call(self, prompt: str) -> str:
        assert self._tokenizer is not None
        assert self._model is not None
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas(),
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        generated = self._model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
        )
        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


def create_tool_planner() -> ToolPlanner:
    backend = os.environ.get("ADVISOR_MODEL_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend in ("", "rules"):
        return RuleBasedPlanner()
    if backend in ("minicpm", "minicpm-transformers"):
        return MiniCPMTransformersPlanner(os.environ.get("ADVISOR_MODEL_ID", DEFAULT_MODEL_ID))
    raise RuntimeError(f"Unsupported ADVISOR_MODEL_BACKEND={backend!r}")


def runtime_status(planner: ToolPlanner) -> RuntimeStatus:
    return RuntimeStatus(
        backend=planner.backend,
        model_id=planner.model_id,
        loaded=not isinstance(planner, MiniCPMTransformersPlanner) or planner._model is not None,
        tool_count=len(tool_schemas()),
    )


def render_context(message: str, state: dict[str, Any]) -> str:
    ideas = state.get("ideas") or []
    trace = state.get("trace") or []
    idea_lines = [
        f"- {idea.get('title', 'Untitled')}: {idea.get('pitch', '')}"
        for idea in ideas[-3:]
    ]
    trace_lines = [
        f"- {event.get('input', '')} -> {event.get('verdict', '')} {event.get('overall', '')}"
        for event in trace[-3:]
    ]
    return "\n".join(
        [
            "Choose exactly one tool call for the next advisor action.",
            "Return only <function name=\"tool_name\">{...json...}</function>.",
            f"User message: {message}",
            "Idea board:",
            *(idea_lines or ["- empty"]),
            "Recent trace:",
            *(trace_lines or ["- empty"]),
        ]
    )


def system_prompt() -> str:
    return (
        "You are The Unwritten Almanac's originality and build-plan advisor. "
        "Use tools to inspect existing projects, find whitespace, save ideas, score ideas, and make plans. "
        "Emit exactly one XML tool call."
    )


def _json_string(value: str) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _title(text: str) -> str:
    title = text[:64].strip(" .") or "Unwritten Page"
    if any(char.isupper() or char.isdigit() for char in title):
        return title[0].upper() + title[1:]
    return title.capitalize()
