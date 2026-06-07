from dataclasses import dataclass

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


def test_extract_transcript_accepts_nemo_output_shapes() -> None:
    assert extract_transcript(["A spoken idea."]) == "A spoken idea."
    assert extract_transcript([{"text": "A mapped archive."}]) == "A mapped archive."
    assert extract_transcript([Hypothesis("A private timeline.")]) == "A private timeline."
