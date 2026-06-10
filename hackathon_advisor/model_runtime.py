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
DEFAULT_ADAPTER_REVISION = "25de69bcde397e1bcdd852923b56a42f10222650"
DEFAULT_BACKEND = "minicpm-transformers"
MAX_TOOL_CALL_TOKENS = 180
MINICPM_DEMO_TEMPERATURE = 0.9
MINICPM_DEMO_TOP_P = 0.95

# One lock for every MiniCPM generation in this process. The atlas chat borrows the
# advisor's loaded model and toggles its LoRA off via PeftModel.disable_adapter(),
# which mutates shared model state — so adapter toggling and generate() must never
# interleave across threads. The lock is held for the FULL lifetime of the streaming
# worker thread (acquired before it starts, released after it joins).
_GENERATION_LOCK = threading.Lock()


def generation_lock() -> threading.Lock:
    return _GENERATION_LOCK


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
        elif _matches_command(lower, ("compare", "compare ideas", "choose", "rank", "rank ideas")):
            output = '<function name="compare_ideas">{}</function>'
        elif _matches_command(
            lower,
            (
                "plan",
                "make a plan",
                "make a build plan",
                "draft a plan",
                "draft a build plan",
                "build plan",
                "roadmap",
                "next step",
                "milestone",
            ),
        ):
            output = '<function name="make_plan">{}</function>'
        elif _matches_command(
            lower,
            (
                "gap",
                "find gap",
                "find a gap",
                "find whitespace",
                "write bolder",
                "bolder",
                "unwritten",
                "make it more original",
                "new direction",
            ),
        ):
            output = '<function name="find_whitespace">{}</function>'
        elif _matches_command(
            lower,
            (
                "search",
                "search for",
                "find similar",
                "similar",
                "is this already",
                "already built",
                "check overlap",
                "overlap",
                "show echoes",
                "echo",
            ),
        ):
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
        self._load_lock = threading.Lock()

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
        # Double-checked: the advisor and the atlas chat share this planner, so two
        # cold-start requests could otherwise both run the full from_pretrained load
        # (a ~2x transient memory spike). _GENERATION_LOCK starts too late to help.
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            self._load()

    def _load(self) -> None:
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
        model = _load_minicpm_causal_lm(AutoModelForCausalLM, base_model_id, target, torch)
        if self.adapter_id:
            model = PeftModel.from_pretrained(model, self.adapter_id, **adapter_kwargs)
            if target not in ("auto", "cpu"):
                model = model.to(target)
        model.eval()
        self._model = model
        if hasattr(torch, "inference_mode"):
            self._inference_mode = torch.inference_mode
        _logger.info(
            "MiniCPM loaded | requested_device=%s resolved_device=%s adapter=%s",
            self.device,
            self.resolved_device,
            self.adapter_id or "(none)",
        )

    def _prepare_inputs(self, prompt: str) -> Any:
        assert self._tokenizer is not None
        assert self._model is not None
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ]
        return _minicpm_chat_inputs(
            self._tokenizer,
            messages,
            enable_thinking=False,
            device=next(self._model.parameters()).device,
        )

    def _stream_tool_call(self, prompt: str) -> Iterator[tuple[int, str]]:
        assert self._tokenizer is not None
        assert self._model is not None
        inputs = self._prepare_inputs(prompt)
        yield from _stream_minicpm_generation(
            self._model,
            self._tokenizer,
            inputs,
            max_new_tokens=MAX_TOOL_CALL_TOKENS,
            temperature=0.0,
            inference_mode=self._inference_mode,
        )

    def ensure_loaded(self) -> None:
        """Public lazy-load trigger so a borrower (the atlas chat) can share the model."""
        self._ensure_loaded()

    def base_model_context(self):
        """Context manager that exposes the BASE weights of the loaded model.

        With a LoRA adapter attached this is PeftModel.disable_adapter(); without one
        the model already is the base, so a nullcontext suffices. Callers must hold
        generation_lock() around the entered context (see _stream_minicpm_generation)."""
        if self.adapter_id and self._model is not None and hasattr(self._model, "disable_adapter"):
            return self._model.disable_adapter()
        return nullcontext()


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


def _load_minicpm_causal_lm(model_cls: Any, model_id: str, target: str, torch: Any) -> Any:
    if target == "auto":
        return model_cls.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    if target == "cuda":
        return model_cls.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")
    if target == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return model_cls.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to("mps")
    return model_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to("cpu")


def _minicpm_chat_inputs(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
    device: Any,
) -> Any:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    _strip_unused_generation_inputs(inputs)
    return inputs


def _minicpm_chat_inputs_with_tools(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    enable_thinking: bool,
    device: Any,
) -> Any:
    """Chat inputs with the native tools= injection (atlas chat pass 1).

    Kept separate from _minicpm_chat_inputs so the advisor's exact template call —
    asserted verbatim in tests — stays untouched."""
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    _strip_unused_generation_inputs(inputs)
    return inputs


def _minicpm_generation_kwargs(
    inputs: dict[str, Any],
    *,
    max_new_tokens: int,
    temperature: float = MINICPM_DEMO_TEMPERATURE,
    top_p: float = MINICPM_DEMO_TOP_P,
    streamer: Any | None = None,
) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
    }
    if streamer is not None:
        generation_kwargs["streamer"] = streamer
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
    else:
        generation_kwargs.update(do_sample=False)
    return generation_kwargs


def _stream_minicpm_generation(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    inference_mode: Any | None = None,
    model_context: Any | None = None,
) -> Iterator[tuple[int, str]]:
    """Stream one MiniCPM generation as (token_count, text_piece) tuples.

    Shared by the advisor tool-call pass and both atlas-chat passes. generate() runs in
    a daemon thread feeding a TextIteratorStreamer; the process-wide generation lock is
    held from before the worker starts until after it joins, so an adapter toggle
    (``model_context`` — e.g. PeftModel.disable_adapter()) can never interleave with a
    concurrent adapter-on generation. The ``finally`` also covers a consumer that
    abandons the generator mid-stream."""
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = _minicpm_generation_kwargs(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        streamer=streamer,
    )
    errors: list[BaseException] = []

    def _run() -> None:
        context = inference_mode() if inference_mode is not None else nullcontext()
        weights = model_context() if model_context is not None else nullcontext()
        try:
            with weights, context:
                model.generate(**generation_kwargs)
        except BaseException as error:  # surfaced after the streamer drains
            errors.append(error)
            # generate() never reached its end sentinel, so wake the consumer instead of
            # letting it block forever, then re-raise from the main thread below.
            streamer.end()

    worker = threading.Thread(target=_run, daemon=True)
    with _GENERATION_LOCK:
        worker.start()
        try:
            tokens = 0
            for piece in streamer:
                if not piece:
                    continue
                tokens += 1
                yield tokens, piece
        finally:
            worker.join()
    if errors:
        raise errors[0]


class ChatRunner(Protocol):
    """Streams atlas-chat generations over the (shared) base model."""

    backend: str
    model_id: str

    supports_thinking: bool

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int,
        enable_thinking: bool = False,
    ) -> Iterator[tuple[int, str]]:
        ...


class MiniCPMChatRunner:
    """Atlas-chat generations on the advisor's MiniCPM instance with the LoRA disabled.

    Borrows the advisor planner's model and tokenizer (never loads its own copy) and
    runs every generation under base_model_context() + the shared generation lock, so
    the chat speaks with the BASE MiniCPM5-1B voice while the advisor keeps its adapter."""

    backend = "minicpm-transformers"
    # With enable_thinking the template ends the prompt with "<think>\n", so the
    # stream is reasoning text up to "</think>" followed by the actual content.
    supports_thinking = True

    def __init__(self, planner: MiniCPMTransformersPlanner) -> None:
        self._planner = planner

    @property
    def model_id(self) -> str:
        return self._planner.model_id

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int,
        enable_thinking: bool = False,
    ) -> Iterator[tuple[int, str]]:
        planner = self._planner
        planner.ensure_loaded()
        assert planner._model is not None and planner._tokenizer is not None
        device = next(planner._model.parameters()).device
        if tools:
            inputs = _minicpm_chat_inputs_with_tools(
                planner._tokenizer,
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
                device=device,
            )
        else:
            inputs = _minicpm_chat_inputs(
                planner._tokenizer,
                messages,
                enable_thinking=enable_thinking,
                device=device,
            )
        yield from _stream_minicpm_generation(
            planner._model,
            planner._tokenizer,
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            inference_mode=planner._inference_mode,
            model_context=planner.base_model_context,
        )


class RuleBasedChatRunner:
    """Deterministic ChatRunner for the rules backend (tests, weight-free UI work).

    Pass 1 (tools given) emits a native-format call chosen by the keyword intent
    router; pass 2 emits a fixed grounded sentence — the UI's verified cards carry
    the actual data either way."""

    backend = "rules"
    model_id = "deterministic-chat-router"
    supports_thinking = False

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int,
        enable_thinking: bool = False,
    ) -> Iterator[tuple[int, str]]:
        from xml.sax.saxutils import escape

        from hackathon_advisor.dashboard_chat_contracts import heuristic_chat_call

        if tools:
            message = _last_user_content(messages)
            call = heuristic_chat_call(message)
            params = "".join(
                f'<param name="{name}">{escape(str(value))}</param>'
                for name, value in call.arguments.items()
            )
            yield 1, f'<function name="{call.name}">{params}</function>'
            return
        yield 1, "Here is what the atlas snapshot shows; the cards below are the verified data."


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def create_chat_runner(planner: ToolPlanner) -> ChatRunner:
    """Build the atlas ChatRunner for an advisor planner; never loads a second model."""
    if isinstance(planner, MiniCPMTransformersPlanner):
        return MiniCPMChatRunner(planner)
    return RuleBasedChatRunner()


def create_tool_planner(device: str = "auto") -> ToolPlanner:
    backend = os.environ.get("ADVISOR_MODEL_BACKEND", "").strip().lower() or DEFAULT_BACKEND
    if backend == "rules":
        return RuleBasedPlanner()
    if backend in ("minicpm", "minicpm-transformers"):
        return MiniCPMTransformersPlanner(
            os.environ.get("ADVISOR_MODEL_ID", DEFAULT_MODEL_ID),
            os.environ.get("ADVISOR_ADAPTER_ID", DEFAULT_ADAPTER_ID),
            os.environ.get("ADVISOR_ADAPTER_REVISION", DEFAULT_ADAPTER_REVISION),
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


def _matches_command(lower_text: str, phrases: tuple[str, ...]) -> bool:
    return lower_text in phrases or any(lower_text.startswith(f"{phrase} ") for phrase in phrases)


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
