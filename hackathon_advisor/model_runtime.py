from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import os
import re
import threading
from typing import Any, Protocol

from hackathon_advisor.tools import idea_from_text
from hackathon_advisor.tool_contracts import ToolResolution, resolve_tool_call, tool_schemas
from hackathon_advisor.zerogpu import zero_gpu_enabled

_logger = logging.getLogger("hackathon_advisor")


DEFAULT_MODEL_ID = "openbmb/MiniCPM5-1B"
DEFAULT_ADAPTER_ID = "build-small-hackathon/hackathon-advisor-minicpm5-lora"
DEFAULT_BACKEND = "rules"
MAX_TOOL_CALL_TOKENS = 180


class ToolPlanner(Protocol):
    backend: str
    model_id: str
    adapter_id: str
    adapter_revision: str

    def plan(self, message: str, state: dict[str, Any]) -> ToolResolution:
        ...

    def plan_iter(self, message: str, state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield {"type": "model_progress", "tokens": int} events while planning, then a
        final {"type": "resolved", "resolution": ToolResolution} event."""
        ...


@dataclass(frozen=True)
class RuntimeStatus:
    backend: str
    model_id: str
    adapter_id: str
    adapter_revision: str
    loaded: bool
    tool_count: int
    device: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "loaded": self.loaded,
            "tool_count": self.tool_count,
            "device": self.device,
        }


class RuleBasedPlanner:
    backend = "rules"
    model_id = "deterministic-tool-router"
    adapter_id = ""
    adapter_revision = ""

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
        elif _has_command_term(lower, ("compare", "choose", "rank")):
            output = '<function name="compare_ideas">{}</function>'
        elif _has_command_term(lower, ("plan", "roadmap", "next step", "milestone")):
            output = '<function name="make_plan">{}</function>'
        elif _has_command_term(lower, ("whitespace", "original", "new", "bolder", "unwritten", "gap")):
            output = '<function name="find_whitespace">{}</function>'
        elif _has_command_term(lower, ("search", "similar", "already", "existing", "overlap", "echo")):
            output = f'<function name="search_projects">{{"query":{_json_string(text)}}}</function>'
        else:
            title, pitch = idea_from_text(text)
            output = (
                f'<function name="save_idea">'
                f'{{"title":{_json_string(title)},"pitch":{_json_string(pitch)}}}'
                f"</function>"
            )
        return resolve_tool_call(output, fallback_query=text)

    def plan_iter(self, message: str, state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield {"type": "resolved", "resolution": self.plan(message, state)}


class MiniCPMTransformersPlanner:
    backend = "minicpm-transformers"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_id: str = "",
        adapter_revision: str = "",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id.strip() or DEFAULT_MODEL_ID
        self.adapter_id = adapter_id.strip()
        self.adapter_revision = adapter_revision.strip()
        self.device = (device or "auto").strip().lower() or "auto"
        self.resolved_device = ""
        self._tokenizer = None
        self._model = None
        self._inference_mode = None

    def plan(self, message: str, state: dict[str, Any]) -> ToolResolution:
        resolution: ToolResolution | None = None
        for event in self.plan_iter(message, state):
            if event.get("type") == "resolved":
                resolution = event["resolution"]
        assert resolution is not None
        return resolution

    def plan_iter(self, message: str, state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self._ensure_loaded()
        prompt = render_context(message, state)
        pieces: list[str] = []
        for tokens, piece in self._stream_tool_call(prompt):
            pieces.append(piece)
            yield {"type": "model_progress", "tokens": tokens, "max_tokens": MAX_TOOL_CALL_TOKENS}
        output = _normalize_xml_tool_output("".join(pieces).strip())
        yield {"type": "resolved", "resolution": resolve_tool_call(output, fallback_query=message)}

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
        adapter_kwargs = {"revision": self.adapter_revision} if self.adapter_revision else {}
        if self.adapter_id:
            adapter_config = PeftConfig.from_pretrained(self.adapter_id, **adapter_kwargs)
            base_model_id = str(adapter_config.base_model_name_or_path or base_model_id)

        target = _resolve_torch_device(self.device, torch)
        self.resolved_device = target

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=True,
            **(adapter_kwargs if self.adapter_id else {}),
        )
        model = self._load_model_on_device(
            AutoModelForCausalLM, base_model_id, target, torch
        )
        if self.adapter_id:
            model = PeftModel.from_pretrained(model, self.adapter_id, **adapter_kwargs)
            if target not in ("auto", "cpu"):
                model = model.to(target)
        model.eval()
        _disable_sampling_generation_defaults(model)
        self._model = model
        if hasattr(torch, "inference_mode"):
            self._inference_mode = torch.inference_mode
        _logger.info(
            "MiniCPM loaded | requested_device=%s resolved_device=%s adapter=%s",
            self.device,
            self.resolved_device,
            self.adapter_id or "(none)",
        )

    def _load_model_on_device(self, model_cls: Any, base_model_id: str, target: str, torch: Any) -> Any:
        if target == "auto":
            return model_cls.from_pretrained(
                base_model_id, dtype="auto", device_map="auto", trust_remote_code=True
            )
        if target == "cpu":
            return model_cls.from_pretrained(
                base_model_id, dtype=torch.float32, device_map={"": "cpu"}, trust_remote_code=True
            )
        # mps / cuda: load on CPU first (no accelerate dispatch), then move to the device.
        if target == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            model = model_cls.from_pretrained(
                base_model_id, dtype=torch.float16, trust_remote_code=True
            )
            return model.to(target)
        except Exception as error:  # noqa: BLE001 - keep the turn runnable on CPU
            if target == "mps":
                _logger.warning("MPS load failed (%r); falling back to CPU float32.", error)
                self.resolved_device = "cpu"
                return model_cls.from_pretrained(
                    base_model_id, dtype=torch.float32, device_map={"": "cpu"}, trust_remote_code=True
                )
            raise

    def _prepare_inputs(self, prompt: str) -> Any:
        assert self._tokenizer is not None
        assert self._model is not None
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(next(self._model.parameters()).device)
        _strip_unused_generation_inputs(inputs)
        return inputs

    def _stream_tool_call(self, prompt: str) -> Iterator[tuple[int, str]]:
        from transformers import TextIteratorStreamer

        assert self._tokenizer is not None
        assert self._model is not None
        inputs = self._prepare_inputs(prompt)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": MAX_TOOL_CALL_TOKENS,
            "do_sample": False,
            "streamer": streamer,
        }
        errors: list[BaseException] = []

        def _run() -> None:
            context = self._inference_mode() if self._inference_mode is not None else nullcontext()
            try:
                with context:
                    self._model.generate(**generation_kwargs)
            except BaseException as error:  # surfaced after the streamer drains
                errors.append(error)
                # generate() never reached its end sentinel, so wake the consumer instead of
                # letting it block forever, then re-raise from the main thread below.
                streamer.end()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        tokens = 0
        for piece in streamer:
            if not piece:
                continue
            tokens += 1
            yield tokens, piece
        worker.join()
        if errors:
            raise errors[0]


def _device_available(device: str, torch: Any) -> bool:
    try:
        if device == "cuda":
            return bool(torch.cuda.is_available())
        if device == "mps":
            backend = getattr(torch.backends, "mps", None)
            return bool(backend is not None and backend.is_available())
    except Exception:  # pragma: no cover - device dependent
        return False
    return False


def _best_local_device(torch: Any) -> str:
    # Avoid touching CUDA inside a ZeroGPU main process — there is no local GPU there, and
    # probing it can disturb the ZeroGPU allocator.
    if not zero_gpu_enabled() and _device_available("cuda", torch):
        return "cuda"
    if _device_available("mps", torch):
        return "mps"
    return "cpu"


def _resolve_torch_device(preference: str, torch: Any) -> str:
    """Map a configured device preference to a concrete torch device.

    "auto" stays "auto" (accelerate device_map handles ZeroGPU/CUDA/CPU placement). "local"
    picks the best on-machine accelerator: CUDA -> MPS (Apple Silicon) -> CPU. An explicit
    cuda/mps that is unavailable degrades to the best available local device."""
    pref = (preference or "auto").strip().lower()
    if pref == "auto":
        return "auto"
    if pref == "cpu":
        return "cpu"
    if pref in ("cuda", "mps"):
        return pref if _device_available(pref, torch) else _best_local_device(torch)
    return _best_local_device(torch)


def create_tool_planner(device: str = "auto") -> ToolPlanner:
    backend = os.environ.get("ADVISOR_MODEL_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend in ("", "rules"):
        return RuleBasedPlanner()
    if backend in ("minicpm", "minicpm-transformers"):
        return MiniCPMTransformersPlanner(
            os.environ.get("ADVISOR_MODEL_ID", DEFAULT_MODEL_ID),
            os.environ.get("ADVISOR_ADAPTER_ID", ""),
            os.environ.get("ADVISOR_ADAPTER_REVISION", ""),
            device=device,
        )
    raise RuntimeError(f"Unsupported ADVISOR_MODEL_BACKEND={backend!r}")


def runtime_status(planner: ToolPlanner) -> RuntimeStatus:
    device = getattr(planner, "resolved_device", "") or getattr(planner, "device", "")
    return RuntimeStatus(
        backend=planner.backend,
        model_id=planner.model_id,
        adapter_id=planner.adapter_id,
        adapter_revision=planner.adapter_revision,
        loaded=not isinstance(planner, MiniCPMTransformersPlanner) or planner._model is not None,
        tool_count=len(tool_schemas()),
        device=str(device),
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
            f"Available tools: {', '.join(spec['function']['name'] for spec in tool_schemas())}.",
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


def _disable_sampling_generation_defaults(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None


def _normalize_xml_tool_output(output: str) -> str:
    stripped = output.strip()
    if stripped.startswith('name="'):
        stripped = f"<function {stripped}"
    if stripped.startswith("<function ") and not stripped.endswith("</function>"):
        stripped = f"{stripped}</function>"
    return stripped


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


def _has_command_term(lower_text: str, terms: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lower_text)
        for term in terms
    )


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
