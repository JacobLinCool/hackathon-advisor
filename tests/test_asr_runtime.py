from dataclasses import dataclass
import builtins

from hackathon_advisor.asr_runtime import (
    DEFAULT_ASR_MODEL_ID,
    NemotronAsrTranscriber,
    extract_transcript,
)


@dataclass
class Hypothesis:
    text: str


def test_nemotron_transcriber_status_is_lazy() -> None:
    transcriber = NemotronAsrTranscriber()

    status = transcriber.status().to_dict()

    assert status["backend"] == "nemo-asr"
    assert status["model_id"] == DEFAULT_ASR_MODEL_ID
    assert status["loaded"] is False
    assert status["sample_rate"] == 16_000


def test_nemotron_transcriber_requires_nemo_asr(monkeypatch) -> None:
    real_import = builtins.__import__

    def block_nemo_import(name, *args, **kwargs):
        if name == "nemo.collections.asr":
            raise ImportError("nemo unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_nemo_import)
    transcriber = NemotronAsrTranscriber()

    try:
        transcriber._ensure_loaded()
    except RuntimeError as error:
        message = str(error)
        assert "NVIDIA NeMo ASR" in message
    else:
        raise AssertionError("missing NeMo should fail before loading another backend")


def test_extract_transcript_accepts_nemo_output_shapes() -> None:
    assert extract_transcript(["A spoken idea."]) == "A spoken idea."
    assert extract_transcript([{"text": "A mapped archive."}]) == "A mapped archive."
    assert extract_transcript([Hypothesis("A private timeline.")]) == "A private timeline."
