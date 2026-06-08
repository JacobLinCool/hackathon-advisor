from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from typing import Any, Protocol

from hackathon_advisor.data import Project, normalize_project_tags
from hackathon_advisor.model_runtime import (
    DEFAULT_MODEL_ID,
    _minicpm_generation_kwargs,
    _load_minicpm_causal_lm,
    _minicpm_chat_inputs,
    _resolve_torch_device,
)
from hackathon_advisor.quest_taxonomy import (
    QUEST_SYSTEM_PROMPT,
    QUESTS,
    build_app_segment,
    build_readme_segment,
    canonical_quest_ids,
    normalize_match,
    render_quest_prompt,
)


MAX_QUEST_TOKENS = 1024
DEFAULT_QUEST_ADAPTER_ID = "artifacts/quest-lora"
DEFAULT_QUEST_ADAPTER_REVISION = ""


class QuestAnalysisError(RuntimeError):
    pass


class QuestAnalyzer(Protocol):
    source: str

    def analyze(self, projects: Sequence[Project]) -> dict[str, list[dict[str, Any]]]:
        ...


@dataclass(frozen=True)
class ValidatedQuestAnalysis:
    matches_by_project: dict[str, list[dict[str, Any]]]
    source: str


class MiniCPMQuestAnalyzer:
    source = "minicpm-json-quest-analyzer"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "auto",
        adapter_id: str = DEFAULT_QUEST_ADAPTER_ID,
        adapter_revision: str = DEFAULT_QUEST_ADAPTER_REVISION,
    ) -> None:
        self.model_id = model_id.strip() or DEFAULT_MODEL_ID
        self.device = (device or "auto").strip().lower() or "auto"
        self.adapter_id = adapter_id.strip()
        self.adapter_revision = adapter_revision.strip()
        self.resolved_device = ""
        self._tokenizer = None
        self._model = None

    def analyze(self, projects: Sequence[Project]) -> dict[str, list[dict[str, Any]]]:
        self._ensure_loaded()
        matches: dict[str, list[dict[str, Any]]] = {}
        for project in projects:
            try:
                raw = self._generate_json(render_project_quest_prompt(project))
                validated = self._validate_or_repair_project(project, raw).matches_by_project
            except QuestAnalysisError as error:
                raise QuestAnalysisError(f"{project.id}: {error}") from error
            matches.update(validated)
        return matches

    def _validate_or_repair_project(self, project: Project, raw: Mapping[str, Any]) -> ValidatedQuestAnalysis:
        try:
            return _validate_single_project_payload(project, raw)
        except QuestAnalysisError as error:
            repaired = self._repair_schema_json(raw, str(error))
            try:
                return _validate_single_project_payload(project, repaired)
            except QuestAnalysisError as repair_error:
                raise QuestAnalysisError(f"{error}; MiniCPM schema repair failed: {repair_error}") from repair_error

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if self.adapter_id:
                from peft import PeftConfig, PeftModel
        except ImportError as error:
            raise QuestAnalysisError(
                "MiniCPM quest analysis requires torch and transformers (and peft when "
                "ADVISOR_QUEST_ADAPTER_ID is set). Install runtime requirements before enabling dashboard refresh."
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

    def _generate_json(self, prompt: str) -> dict[str, Any]:
        text = self._generate_text(QUEST_SYSTEM_PROMPT, prompt)
        try:
            parsed = _extract_json_object(text)
        except QuestAnalysisError as error:
            repaired = self._repair_invalid_json(text)
            try:
                parsed = _extract_json_object(repaired)
            except QuestAnalysisError as repair_error:
                preview = " ".join(text.split())[:280]
                repair_preview = " ".join(repaired.split())[:280]
                raise QuestAnalysisError(
                    f"{error}: {preview}; MiniCPM JSON repair failed: {repair_error}: {repair_preview}"
                ) from repair_error
        if not isinstance(parsed, dict):
            raise QuestAnalysisError("quest analyzer did not return a JSON object")
        return parsed

    def _generate_text(self, system_prompt: str, user_prompt: str, *, disable_adapter: bool = False) -> str:
        import torch

        assert self._tokenizer is not None
        assert self._model is not None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = _minicpm_chat_inputs(
            self._tokenizer,
            messages,
            enable_thinking=False,
            device=next(self._model.parameters()).device,
        )
        generation_kwargs = _minicpm_generation_kwargs(
            inputs,
            max_new_tokens=MAX_QUEST_TOKENS,
            temperature=0.0,  # strict JSON wants deterministic greedy decoding
        )
        generation_kwargs["eos_token_id"] = self._chat_eos_token_id()
        adapter_context = _disabled_adapter(self._model) if disable_adapter else nullcontext()
        with adapter_context, torch.inference_mode():
            output = self._model.generate(**generation_kwargs)
        generated = output[:, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def _repair_invalid_json(self, invalid_output: str) -> str:
        repair_system = "You repair JSON. Return exactly one valid JSON object and nothing else."
        repair_prompt = "\n".join(
            [
                "Rewrite this invalid JSON as valid compact JSON.",
                "Every match object must contain exactly these keys: quest, confidence, evidence, source.",
                "Keep the same matches, quest names, confidence values, and source values.",
                "If an evidence value contains unescaped double quote characters, escape them or paraphrase the evidence.",
                "Never omit source. If a source key was damaged by malformed JSON, infer readme or app_file from the damaged object.",
                "Drop any match whose quest is not valid, or whose source cannot be inferred as readme or app_file.",
                f"Valid quests: {', '.join(QUESTS)}.",
                'Valid sources: readme, app_file.',
                "Do not copy any text from these repair instructions into evidence.",
                "",
                "Invalid JSON:",
                invalid_output,
            ]
        )
        return self._generate_text(repair_system, repair_prompt, disable_adapter=True)

    def _repair_schema_json(self, parsed_output: Mapping[str, Any], validation_error: str) -> dict[str, Any]:
        repair_system = "You repair JSON schemas. Return exactly one valid JSON object and nothing else."
        repair_prompt = "\n".join(
            [
                "The following quest-classification JSON parsed, but failed schema validation.",
                f"Validation error: {validation_error}",
                "",
                "Rewrite it to satisfy this schema exactly:",
                '{"matches":[{"quest":"...","confidence":0.1,"evidence":"...","source":"readme"}]}',
                "",
                "Rules:",
                f"- quest must be one of: {', '.join(QUESTS)}.",
                "- source must be readme or app_file.",
                "- Every match object must include source; never omit it.",
                "- If source is missing but evidence clearly came from code, use app_file; if it clearly came from prose, use readme.",
                "- confidence must be greater than 0 and no more than 1.",
                "- Keep at most one match per quest; keep the strongest and clearest evidence.",
                "- Remove matches with empty evidence or evidence copied from the quest instructions.",
                "- Do not copy any text from these repair instructions into evidence.",
                "- Do not add new quests that are not already intended by the input.",
                "",
                "Input JSON:",
                json.dumps(parsed_output, ensure_ascii=False, separators=(",", ":")),
            ]
        )
        repaired = self._generate_text(repair_system, repair_prompt, disable_adapter=True)
        parsed = _extract_json_object(repaired)
        if not isinstance(parsed, dict):
            raise QuestAnalysisError("MiniCPM schema repair did not return a JSON object")
        return parsed

    def _chat_eos_token_id(self) -> int:
        assert self._tokenizer is not None
        token_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(token_id, int) or token_id < 0:
            raise QuestAnalysisError("MiniCPM tokenizer is missing the <|im_end|> chat terminator")
        return token_id


def create_quest_analyzer(device: str = "auto") -> QuestAnalyzer:
    backend = os.environ.get("ADVISOR_QUEST_ANALYZER_BACKEND", "").strip().lower()
    if not backend:
        backend = os.environ.get("ADVISOR_MODEL_BACKEND", "").strip().lower()
    if backend in {"minicpm", "minicpm-transformers"}:
        return MiniCPMQuestAnalyzer(
            os.environ.get("ADVISOR_QUEST_MODEL_ID", os.environ.get("ADVISOR_MODEL_ID", DEFAULT_MODEL_ID)),
            device=device,
            adapter_id=os.environ.get("ADVISOR_QUEST_ADAPTER_ID", DEFAULT_QUEST_ADAPTER_ID),
            adapter_revision=os.environ.get("ADVISOR_QUEST_ADAPTER_REVISION", DEFAULT_QUEST_ADAPTER_REVISION),
        )
    raise QuestAnalysisError(
        "Dashboard refresh requires ADVISOR_QUEST_ANALYZER_BACKEND=minicpm-transformers. "
        f"Got {backend or 'unset'}."
    )


def validate_quest_analysis_payload(
    payload: Mapping[str, Any],
    projects: Sequence[Project],
    *,
    source: str = "validated-json",
) -> ValidatedQuestAnalysis:
    rows = payload.get("projects")
    if not isinstance(rows, list):
        raise QuestAnalysisError("quest analysis JSON must contain a projects list")
    expected_ids = [project.id for project in projects]
    expected = set(expected_ids)
    seen: set[str] = set()
    matches_by_project: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise QuestAnalysisError("quest project rows must be objects")
        project_id = str(row.get("project_id") or "")
        if project_id not in expected:
            raise QuestAnalysisError(f"quest analysis returned an unknown project id: {project_id}")
        if project_id in seen:
            raise QuestAnalysisError(f"quest analysis returned a duplicate project id: {project_id}")
        seen.add(project_id)
        matches_by_project[project_id] = _validate_project_matches(row.get("matches"), project_id)
    missing = [project_id for project_id in expected_ids if project_id not in seen]
    if missing:
        raise QuestAnalysisError(f"quest analysis missed {len(missing)} projects")
    return ValidatedQuestAnalysis(matches_by_project=matches_by_project, source=source)


def validate_matches_by_project(
    matches_by_project: Mapping[str, Sequence[Mapping[str, Any]]],
    projects: Sequence[Project],
    *,
    source: str,
) -> ValidatedQuestAnalysis:
    payload = {
        "projects": [
            {
                "project_id": project.id,
                "matches": list(matches_by_project.get(project.id, [])),
            }
            for project in projects
        ]
    }
    return validate_quest_analysis_payload(payload, projects, source=source)


def _validate_single_project_payload(project: Project, raw: Mapping[str, Any]) -> ValidatedQuestAnalysis:
    return validate_quest_analysis_payload(
        {
            "projects": [
                {
                    "project_id": project.id,
                    "matches": raw.get("matches"),
                }
            ]
        },
        [project],
    )


def render_project_quest_prompt(project: Project) -> str:
    """Render the strict two-segment quest prompt from a snapshot Project.

    Refresh snapshots carry the same raw README body and main app source segments
    used for quest LoRA training.
    """
    return render_quest_prompt(
        title=project.title,
        sdk=project.sdk,
        declared_models=project.models,
        tags=normalize_project_tags(project.tags),
        readme_segment=build_readme_segment(project.readme_body),
        app_file_name=project.app_file,
        app_file_segment=build_app_segment(project.app_file_source, project.app_file_embedding_text),
    )


def _validate_project_matches(raw_matches: Any, project_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw_matches, list):
        raise QuestAnalysisError(f"quest matches for {project_id} must be a list")
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_match in raw_matches:
        if not isinstance(raw_match, dict):
            raise QuestAnalysisError(f"quest matches for {project_id} must be objects")
        try:
            quest_ids = canonical_quest_ids(raw_match.get("quest"))
        except ValueError as error:
            raise QuestAnalysisError(f"quest match for {project_id}: {error}") from error
        for quest_id in quest_ids:
            try:
                match = normalize_match({**raw_match, "quest": quest_id})
            except ValueError as error:
                raise QuestAnalysisError(f"quest match for {project_id}: {error}") from error
            if match["quest"] in seen:
                raise QuestAnalysisError(f"duplicate quest for {project_id}: {match['quest']}")
            seen.add(match["quest"])
            matches.append(match)
    return matches


def _extract_json_object(text: str) -> Any:
    text = _strip_json_fence(text.strip())
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, offset = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if text[index + offset :].strip():
            continue
        return value
    raise QuestAnalysisError("quest analyzer returned invalid JSON")


def _disabled_adapter(model: Any) -> Any:
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()
    return nullcontext()


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return text
    opener = lines[0].strip().lower()
    if opener not in {"```", "```json"}:
        return text
    return "\n".join(lines[1:-1]).strip()
