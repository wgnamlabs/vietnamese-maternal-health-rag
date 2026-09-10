"""Configuration for online retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from indexing.embedding import EmbedderConfig
from indexing.vector_store import QdrantStoreConfig


@dataclass
class RetrievalConfig:
    """Cấu hình tập trung cho retrieval và reranking.

    Các giá trị mặc định được lấy từ index manifest và protocol evaluation.
    Có thể truyền các config indexing đã tồn tại để không tạo lại logic load.
    """

    qdrant: QdrantStoreConfig = field(default_factory=QdrantStoreConfig)
    embedder: EmbedderConfig = field(
        default_factory=lambda: EmbedderConfig(
            model_name="Qwen/Qwen3-Embedding-4B",
            batch_size=16,
            max_seq_length=2048,
            device="cuda",
            trust_remote_code=False,
            auto_batch=True,
            min_batch_size=2,
        )
    )
    bm25_path: Path = Path("data/vector_db/bm25_index.pkl")
    dense_top_k: int = 15
    sparse_top_k: int = 15
    candidate_pool_top_k: int = 15
    final_top_k: int = 5
    rrf_k: int = 60
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
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
    """Tạo config production mặc định từ project root."""
    root = project_root or Path.cwd()
    return RetrievalConfig(
        qdrant=QdrantStoreConfig(
            collection_name="phu_san_chunks",
            host="localhost",
            port=6333,
            vector_size=2560,
            distance="cosine",
        ),
        bm25_path=root / "data" / "vector_db" / "bm25_index.pkl",
    )
