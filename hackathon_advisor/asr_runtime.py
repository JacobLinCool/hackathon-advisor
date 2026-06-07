from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


DEFAULT_ASR_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_ASR_BACKEND = "nemo-asr"
DEFAULT_ASR_SAMPLE_RATE = 16_000
DEFAULT_WHISPER_MODEL_ID = "openai/whisper-small.en"
WHISPER_BACKEND = "whisper-transformers"

_logger = logging.getLogger("hackathon_advisor")


@dataclass(frozen=True)
class AsrTranscript:
    transcript: str
    model_id: str
    backend: str
    sample_rate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "model_id": self.model_id,
            "backend": self.backend,
            "sample_rate": self.sample_rate,
        }


@dataclass(frozen=True)
class AsrStatus:
    backend: str
    model_id: str
    loaded: bool
    sample_rate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "loaded": self.loaded,
            "sample_rate": self.sample_rate,
        }


class NemotronAsrTranscriber:
    """Nemotron voice input. Its declared identity (status, model id) is the deployed Space
    backend — NVIDIA NeMo ASR. When NeMo is not installed (e.g. local development on a Mac,
    where NeMo does not install cleanly), transcription transparently falls back to a local
    Whisper model through transformers so voice still works; the returned transcript reports
    whichever engine actually ran."""

    backend = DEFAULT_ASR_BACKEND

    def __init__(
        self,
        model_id: str = DEFAULT_ASR_MODEL_ID,
        sample_rate: int = DEFAULT_ASR_SAMPLE_RATE,
        whisper_model_id: str = DEFAULT_WHISPER_MODEL_ID,
    ) -> None:
        self.model_id = model_id.strip() or DEFAULT_ASR_MODEL_ID
        self.sample_rate = sample_rate
        self.whisper_model_id = whisper_model_id.strip() or DEFAULT_WHISPER_MODEL_ID
        self._engine: tuple[str, Any] | None = None
        self._active_backend = ""
        self._active_model_id = ""

    def status(self) -> AsrStatus:
        return AsrStatus(
            backend=self._active_backend or self.backend,
            model_id=self._active_model_id or self.model_id,
            loaded=self._engine is not None,
            sample_rate=self.sample_rate,
        )

    def transcribe(self, audio_path: Path) -> AsrTranscript:
        source = Path(audio_path)
        if not source.is_file():
            raise RuntimeError("Voice note was not saved before transcription.")
        self._ensure_loaded()
        kind, engine = self._engine  # type: ignore[misc]
        with tempfile.TemporaryDirectory(prefix="advisor-asr-") as directory:
            wav_path = Path(directory) / "voice.wav"
            normalize_audio_for_asr(source, wav_path, self.sample_rate)
            if kind == "nemo":
                outputs = engine.transcribe([str(wav_path)], batch_size=1)
                transcript = extract_transcript(outputs).strip()
            else:
                transcript = _whisper_transcribe(engine, wav_path, self.sample_rate).strip()
        if not transcript:
            raise RuntimeError(f"{self._active_backend or self.backend} returned an empty transcript.")
        return AsrTranscript(
            transcript=transcript,
            model_id=self._active_model_id or self.model_id,
            backend=self._active_backend or self.backend,
            sample_rate=self.sample_rate,
        )

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        preference = os.environ.get("ADVISOR_ASR_BACKEND", "auto").strip().lower()
        if preference in ("whisper", WHISPER_BACKEND):
            self._load_whisper()
            return
        try:
            self._load_nemo()
            return
        except RuntimeError:
            if preference in ("nemo", "nemo-asr", "nemotron"):
                raise  # explicit Nemotron request: do not silently fall back
            _logger.warning("NeMo ASR unavailable; falling back to local Whisper (%s).", self.whisper_model_id)
            self._load_whisper()

    def _load_nemo(self) -> None:
        try:
            import torch
            import nemo.collections.asr as nemo_asr
        except ImportError as error:
            raise RuntimeError(
                "Nemotron voice input requires NVIDIA NeMo ASR. Install `nemo_toolkit[asr]` "
                "before enabling voice transcription."
            ) from error
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)
        device = os.environ.get("ADVISOR_ASR_DEVICE", "").strip() or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._engine = ("nemo", model)
        self._active_backend = self.backend
        self._active_model_id = self.model_id

    def _load_whisper(self) -> None:
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as error:
            raise RuntimeError(
                "Local voice fallback requires transformers and torch. Install runtime "
                "requirements before enabling voice transcription."
            ) from error
        device = _resolve_asr_device(torch)
        if device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        processor = WhisperProcessor.from_pretrained(self.whisper_model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_id)
        model.to(device)
        model.eval()
        self._engine = ("whisper", (processor, model))
        self._active_backend = WHISPER_BACKEND
        self._active_model_id = self.whisper_model_id
        _logger.info("Whisper ASR loaded | model=%s device=%s", self.whisper_model_id, device)


def create_asr_transcriber() -> NemotronAsrTranscriber:
    sample_rate = int(os.environ.get("ADVISOR_ASR_SAMPLE_RATE", str(DEFAULT_ASR_SAMPLE_RATE)))
    if sample_rate <= 0:
        raise RuntimeError("ADVISOR_ASR_SAMPLE_RATE must be a positive integer.")
    return NemotronAsrTranscriber(
        model_id=os.environ.get("ADVISOR_ASR_MODEL_ID", DEFAULT_ASR_MODEL_ID),
        sample_rate=sample_rate,
        whisper_model_id=os.environ.get("ADVISOR_ASR_WHISPER_MODEL", DEFAULT_WHISPER_MODEL_ID),
    )


def _resolve_asr_device(torch: Any) -> str:
    forced = os.environ.get("ADVISOR_ASR_DEVICE", "").strip().lower()
    if forced:
        return forced
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - device dependent
        pass
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - device dependent
        pass
    return "cpu"


def _whisper_transcribe(engine: tuple[Any, Any], wav_path: Path, sample_rate: int) -> str:
    import torch

    processor, model = engine
    audio = _read_wav_mono_float32(wav_path)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    features = inputs.input_features.to(model.device)
    with torch.inference_mode():
        generated = model.generate(features, max_new_tokens=128)
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def _read_wav_mono_float32(wav_path: Path) -> Any:
    import wave

    import numpy as np

    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def normalize_audio_for_asr(source: Path, target: Path, sample_rate: int = DEFAULT_ASR_SAMPLE_RATE) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Voice transcription requires ffmpeg to normalize audio.")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(target),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffmpeg could not read this audio file."
        raise RuntimeError(message)


def extract_transcript(outputs: Any) -> str:
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict):
        return str(outputs.get("text") or outputs.get("transcript") or "")
    if isinstance(outputs, (list, tuple)):
        if not outputs:
            return ""
        return extract_transcript(outputs[0])
    text = getattr(outputs, "text", None)
    if text is not None:
        return str(text)
    transcript = getattr(outputs, "transcript", None)
    if transcript is not None:
        return str(transcript)
    return str(outputs)
