"""Reciprocal Rank Fusion and hybrid retrieval."""

from __future__ import annotations

import time
from typing import Sequence

from .base import BaseRetriever, RetrievedChunk
from .config import RetrievalConfig


def reciprocal_rank_fusion(
    dense_results: Sequence[RetrievedChunk],
    sparse_results: Sequence[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse two ranked lists with RRF and deduplicate by chunk ID.

    Rank được đánh số từ 1. Metadata/text của lần xuất hiện đầu tiên được giữ.
    """
    if k <= 0:
        raise ValueError("k phải > 0")

    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for results in (dense_results, sparse_results):
        for rank, chunk in enumerate(results, start=1):
            chunk_id = str(chunk.chunk_id)
            if chunk_id not in by_id:
                by_id[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    score=0.0,
                    source_type="hybrid",
                    metadata=dict(chunk.metadata),
                )
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused = []
    for chunk_id, chunk in by_id.items():
        chunk.score = scores[chunk_id]
        fused.append(chunk)
    fused.sort(key=lambda item: (-item.score, item.chunk_id))
    return fused


class HybridRetriever(BaseRetriever):
    """Kết hợp dense và sparse retriever bằng RRF."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        config: RetrievalConfig,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.config = config

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve candidate pool từ dense+sparse rồi fuse bằng RRF."""
        if top_k <= 0:
            raise ValueError("top_k phải > 0")
        dense = self.dense_retriever.retrieve(
            query,
            max(top_k, self.config.dense_top_k),
        )
        sparse = self.sparse_retriever.retrieve(
            query,
            max(top_k, self.config.sparse_top_k),
        )
        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion(dense, sparse, k=self.config.rrf_k)[:top_k]
        fusion_seconds = time.perf_counter() - fusion_start
        dense_timing = dense[0].metadata.get("_timing", {}) if dense else {}
        sparse_timing = sparse[0].metadata.get("_timing", {}) if sparse else {}
        for result in fused:
            result.metadata["_timing"] = {
                **dense_timing,
                **sparse_timing,
                "rrf_seconds": fusion_seconds,
            }
        return fused
