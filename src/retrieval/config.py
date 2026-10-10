"""Configuration for online retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from indexing.embedding import EmbedderConfig, SELECTED_EMBEDDER_CONFIG
from indexing.vector_store import QdrantStoreConfig


@dataclass
class RetrievalConfig:
    """Central configuration for the selected retrieval baseline."""

    qdrant: QdrantStoreConfig = field(default_factory=QdrantStoreConfig)
    embedder: EmbedderConfig = field(
        default_factory=lambda: EmbedderConfig(
            **SELECTED_EMBEDDER_CONFIG.__dict__
        )
    )
    bm25_path: Path = Path("data/vector_db/bm25_index.pkl")
    dense_top_k: int = 15
    sparse_top_k: int = 15
    # Frozen after DEV/TEST retrieval tuning:
    # Dense/BM25 top-15 -> RRF candidate pool top-50 -> reranker -> final top-5.
    candidate_pool_top_k: int = 30
    final_top_k: int = 5
    rrf_k: int = 60
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    # Embedding query encoding runs on CPU; reserve GPU for reranker + Ollama.
    reranker_device: str = "cuda"
    reranker_batch_size: int = 16

    def __post_init__(self) -> None:
        for name in (
            "dense_top_k",
            "sparse_top_k",
            "candidate_pool_top_k",
            "final_top_k",
            "rrf_k",
            "reranker_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} phải > 0")
        if self.final_top_k > self.candidate_pool_top_k:
            raise ValueError(
                "final_top_k không được lớn hơn candidate_pool_top_k."
            )


def default_retrieval_config(project_root: Optional[Path] = None) -> RetrievalConfig:
    """Create production retrieval config rooted at the project directory."""
    root = project_root or Path.cwd()
    return RetrievalConfig(
        qdrant=QdrantStoreConfig(),
        bm25_path=root / "data" / "vector_db" / "bm25_index.pkl",
    )
