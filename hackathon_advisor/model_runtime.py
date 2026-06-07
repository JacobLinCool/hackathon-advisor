from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
import re
from typing import Any, Protocol

from hackathon_advisor.tools import idea_from_text
from hackathon_advisor.tool_contracts import ToolResolution, resolve_tool_call, tool_schemas


DEFAULT_MODEL_ID = "openbmb/MiniCPM5-1B"
DEFAULT_ADAPTER_ID = "build-small-hackathon/hackathon-advisor-minicpm5-lora"
DEFAULT_BACKEND = "rules"


class ToolPlanner(Protocol):
    backend: str
    model_id: str
    adapter_id: str

    def plan(self, message: str, state: dict[str, Any]) -> ToolResolution:
        ...


@dataclass(frozen=True)
class RuntimeStatus:
    backend: str
    model_id: str
    adapter_id: str
    loaded: bool
    tool_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "adapter_id": self.adapter_id,
            "loaded": self.loaded,
            "tool_count": self.tool_count,
        }


class RuleBasedPlanner:
    backend = "rules"
    model_id = "deterministic-tool-router"
    adapter_id = ""

    def plan(self, message: str, state: dict[str, Any]) -> ToolResolution:
        text = " ".join(message.strip().split())
        lower = text.lower()
        project_id = _project_reference_id(text)
        if not text:
            output = '<function name="list_projects">{"sort":"likes"}</function>'
        elif _wants_project_list(lower):
            output = '<function name="list_projects">{"sort":"likes"}</function>'
        elif project_id:
            output = f'<function name="get_project">{{"id":{_json_string(project_id)}}}</function>'
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

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, adapter_id: str = "") -> None:
        self.model_id = model_id.strip() or DEFAULT_MODEL_ID
        self.adapter_id = adapter_id.strip()
        self._tokenizer = None
        self._model = None
        self._inference_mode = None

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
            if self.adapter_id:
                from peft import PeftConfig, PeftModel
        except ImportError as error:
            raise RuntimeError(
                "ADVISOR_MODEL_BACKEND=minicpm-transformers requires torch, transformers, accelerate, "
                "and peft when ADVISOR_ADAPTER_ID is set. Install runtime requirements before enabling it."
            ) from error
        base_model_id = self.model_id
        tokenizer_id = self.adapter_id or base_model_id
        if self.adapter_id:
            adapter_config = PeftConfig.from_pretrained(self.adapter_id)
            base_model_id = str(adapter_config.base_model_name_or_path or base_model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if self.adapter_id:
            model = PeftModel.from_pretrained(model, self.adapter_id)
        model.eval()
        self._model = model
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
        ).to(next(self._model.parameters()).device)
        _strip_unused_generation_inputs(inputs)
        context = self._inference_mode() if self._inference_mode is not None else nullcontext()
        with context:
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
        return MiniCPMTransformersPlanner(
            os.environ.get("ADVISOR_MODEL_ID", DEFAULT_MODEL_ID),
            os.environ.get("ADVISOR_ADAPTER_ID", ""),
        )
    raise RuntimeError(f"Unsupported ADVISOR_MODEL_BACKEND={backend!r}")


def runtime_status(planner: ToolPlanner) -> RuntimeStatus:
    return RuntimeStatus(
        backend=planner.backend,
        model_id=planner.model_id,
        adapter_id=planner.adapter_id,
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


def _strip_unused_generation_inputs(inputs: dict[str, Any]) -> None:
    inputs.pop("token_type_ids", None)


def _json_string(value: str) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _wants_project_list(lower_text: str) -> bool:
    exact_phrases = (
        "projects",
        "spaces",
        "current map",
        "project map",
    )
    command_prefixes = (
        "list projects",
        "list spaces",
        "show projects",
        "show spaces",
        "show current map",
        "show project map",
        "open current map",
        "browse projects",
        "browse spaces",
    )
    return lower_text in exact_phrases or any(lower_text.startswith(prefix) for prefix in command_prefixes)


def _project_reference_id(text: str) -> str:
    prefixes = (
        "read project ",
        "open project ",
        "show project ",
        "read space ",
        "open space ",
        "show space ",
    )
    lower = text.lower()
    raw = ""
    for prefix in prefixes:
        if lower.startswith(prefix):
            raw = text[len(prefix) :].strip()
            break
    if not raw:
        return ""
    raw = re.sub(r"^https?://huggingface\.co/spaces/", "", raw, flags=re.IGNORECASE)
    return raw.split()[0].strip(".,;:!?\"'")


def _title(text: str) -> str:
    title = text[:64].strip(" .") or "Unwritten Page"
    if any(char.isupper() or char.isdigit() for char in title):
        return title[0].upper() + title[1:]
    return title.capitalize()
