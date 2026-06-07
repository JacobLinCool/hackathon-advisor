from pathlib import Path
import sys
from types import ModuleType

from hackathon_advisor.data import DEFAULT_EMBEDDING_MODEL_FILE, DEFAULT_EMBEDDING_MODEL_REPO
from hackathon_advisor.llama_embedding import DEFAULT_N_CTX, LlamaCppEmbedder, create_llama_cpp_embedder


def test_llama_embedder_uses_q8_defaults_and_full_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "embedding.gguf"
    model_path.write_bytes(b"gguf")
    captured: dict = {}

    hub = ModuleType("huggingface_hub")

    def fake_hf_hub_download(repo_id: str, filename: str, repo_type: str) -> str:
        captured["download"] = {
            "repo_id": repo_id,
            "filename": filename,
            "repo_type": repo_type,
        }
        return str(model_path)

    hub.hf_hub_download = fake_hf_hub_download
    llama_cpp = ModuleType("llama_cpp")
    llama_cpp.LLAMA_POOLING_TYPE_MEAN = 1

    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            captured["llama_kwargs"] = kwargs

        def embed(self, text: str, normalize: bool) -> list[float]:
            captured["embed"] = {"text": text, "normalize": normalize}
            return [1.0, 0.0]

    llama_cpp.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_cpp)

    vector = LlamaCppEmbedder().embed("private archive")

    assert vector == [1.0, 0.0]
    assert captured["download"] == {
        "repo_id": DEFAULT_EMBEDDING_MODEL_REPO,
        "filename": DEFAULT_EMBEDDING_MODEL_FILE,
        "repo_type": "model",
    }
    assert captured["llama_kwargs"]["n_ctx"] == DEFAULT_N_CTX
    assert captured["llama_kwargs"]["n_batch"] == DEFAULT_N_CTX
    assert captured["llama_kwargs"]["n_ubatch"] == DEFAULT_N_CTX
    assert captured["embed"] == {"text": "private archive", "normalize": True}


def test_create_llama_embedder_accepts_explicit_batch(monkeypatch) -> None:
    monkeypatch.setenv("ADVISOR_EMBEDDING_BATCH", "256")

    embedder = create_llama_cpp_embedder({"dimensions": 768})

    assert embedder.n_batch == 256
