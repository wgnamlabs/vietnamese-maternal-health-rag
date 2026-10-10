from dataclasses import dataclass


@dataclass
class EmbedderConfig:
    model_name: str
    batch_size: int = 32
    max_seq_length: int = 2048
    device: str = "cuda"
    trust_remote_code: bool = False
    note: str = ""
    auto_batch: bool = False
    min_batch_size: int = 1


# Config được chốt từ benchmark DEV và xác nhận trên TEST.
SELECTED_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SELECTED_EMBEDDER_CONFIG = EmbedderConfig(
    model_name=SELECTED_EMBEDDING_MODEL,
    batch_size=32,
    max_seq_length=2048,
    # Runtime mode: one query embedding at a time on CPU to reserve VRAM
    # for the GPU reranker and Ollama generation model.
    device="cpu",
    trust_remote_code=False,
    note="Selected embedding model; runtime query encoding runs on CPU.",
    auto_batch=True,
    min_batch_size=2,
)

# Shortlist được giữ lại để tái lập hoặc mở experiment benchmark mới.
MODELS_TO_COMPARE = [
    EmbedderConfig(
        model_name="BAAI/bge-m3",
        batch_size=32,
        max_seq_length=2048,
        device="cuda",
        trust_remote_code=False,
        note="Multilingual lightweight benchmark candidate.",
    ),
    EmbedderConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        batch_size=4,
        max_seq_length=2048,
        device="cuda",
        trust_remote_code=False,
        note="Lightweight Qwen benchmark candidate; query uses prompt_name='query'.",
        auto_batch=False,
        min_batch_size=2,
    ),
    SELECTED_EMBEDDER_CONFIG,
]
