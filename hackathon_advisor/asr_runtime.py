from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


DEFAULT_ASR_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_ASR_BACKEND = "nemo-asr"
DEFAULT_ASR_SAMPLE_RATE = 16_000


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
    backend = DEFAULT_ASR_BACKEND

    def __init__(
        self,
        model_id: str = DEFAULT_ASR_MODEL_ID,
        sample_rate: int = DEFAULT_ASR_SAMPLE_RATE,
    ) -> None:
        self.model_id = model_id.strip() or DEFAULT_ASR_MODEL_ID
        self.sample_rate = sample_rate
        self._model = None

    def status(self) -> AsrStatus:
        return AsrStatus(
            backend=self.backend,
            model_id=self.model_id,
            loaded=self._model is not None,
            sample_rate=self.sample_rate,
        )

    def transcribe(self, audio_path: Path) -> AsrTranscript:
        source = Path(audio_path)
        if not source.is_file():
            raise RuntimeError("Voice note was not saved before transcription.")
        self._ensure_loaded()
        with tempfile.TemporaryDirectory(prefix="advisor-asr-") as directory:
            wav_path = Path(directory) / "voice.wav"
            normalize_audio_for_asr(source, wav_path, self.sample_rate)
            outputs = self._model.transcribe([str(wav_path)], batch_size=1)
        transcript = extract_transcript(outputs).strip()
        if not transcript:
            raise RuntimeError("Nemotron ASR returned an empty transcript.")
        return AsrTranscript(
            transcript=transcript,
            model_id=self.model_id,
            backend=self.backend,
            sample_rate=self.sample_rate,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import nemo.collections.asr as nemo_asr
        except ImportError as error:
            raise RuntimeError(
                "Nemotron voice input requires NVIDIA NeMo ASR. Install `nemo_toolkit[asr]` "
                "before enabling voice transcription."
            ) from error
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)
        device = os.environ.get("ADVISOR_ASR_DEVICE", "").strip()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        self._model = model


def create_asr_transcriber() -> NemotronAsrTranscriber:
    sample_rate = int(os.environ.get("ADVISOR_ASR_SAMPLE_RATE", str(DEFAULT_ASR_SAMPLE_RATE)))
    if sample_rate <= 0:
        raise RuntimeError("ADVISOR_ASR_SAMPLE_RATE must be a positive integer.")
    return NemotronAsrTranscriber(
        model_id=os.environ.get("ADVISOR_ASR_MODEL_ID", DEFAULT_ASR_MODEL_ID),
        sample_rate=sample_rate,
    )


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
