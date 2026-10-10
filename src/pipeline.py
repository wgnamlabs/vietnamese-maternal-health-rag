# -*- coding: utf-8 -*-
"""End-to-end local pipeline for the maternal-health RAG assistant."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import torch

from retrieval.config import RetrievalConfig, default_retrieval_config
from retrieval.dense_retriever import DenseRetriever
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.sparse_retriever import SparseRetriever


@dataclass(frozen=True)
class FinalResponse:
    """User-facing answer."""

    answer: str
    refused: bool
    top_score: float
    chunks: list[str]


class RAGPipeline:
    """Run pure retrieval and reranking (Generation v2 is planned).
    
    Note: The RLock makes retrieval and reranking strictly sequential.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.retrieval_config: RetrievalConfig = default_retrieval_config(project_root)
        self.dense: DenseRetriever | None = None
        self.sparse: SparseRetriever | None = None
        self.hybrid: HybridRetriever | None = None
        self.reranker: Reranker | None = None
        self._runtime_lock = RLock()

    def _vram_log(self, stage: str) -> None:
        """Write best-effort local GPU lifecycle diagnostics."""
        payload = {"stage": stage, "timestamp": time.time()}
        if torch.cuda.is_available():
            payload.update(
                {
                    "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                    "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                }
            )
        path = self.project_root / "logs" / "vram_lifecycle.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _ensure_retrievers(self) -> HybridRetriever:
        """Initialize the retrieval stack once and retain its resident embedder."""
        if self.dense is None or self.sparse is None or self.hybrid is None:
            self.dense = DenseRetriever(self.retrieval_config)
            self.sparse = SparseRetriever(self.retrieval_config)
            self.hybrid = HybridRetriever(self.dense, self.sparse, self.retrieval_config)
        return self.hybrid

    def warm_up(self) -> None:
        """Load resident retrieval and reranking models before serving."""
        with self._runtime_lock:
            self._ensure_retrievers()
            if self.dense is None:
                raise RuntimeError("Dense retriever không được khởi tạo khi warm-up.")
            self._vram_log("before_embedding_warmup")
            self.dense.embedder.encode(["chăm sóc thai kỳ"], is_query=True)
            self._vram_log("after_embedding_warmup")
            self._ensure_reranker()
            
            # Dummy rerank để làm nóng CUDA
            from retrieval.base import RetrievedChunk
            dummy_candidates = [RetrievedChunk(chunk_id="dummy", text="dummy text", score=1.0, source_type="dense")]
            self.reranker.rerank("dummy query", dummy_candidates, 1)
            
            self._vram_log("after_reranker_warmup")

    def _ensure_reranker(self) -> Reranker:
        """Initialize the reranker once and keep it resident."""
        if self.reranker is None:
            self.reranker = Reranker(self.retrieval_config)
        return self.reranker

    @staticmethod
    def _log_timing(stage: str, seconds: float) -> None:
        """Print one concise, comparable runtime timing entry."""
        print(f"[Timing] {stage}={seconds:.3f}s")



    def answer_query(self, raw_query: str) -> FinalResponse:
        """Answer one query through local hybrid retrieval and generation."""
        total_start = time.perf_counter()
        if not raw_query or not raw_query.strip():
            raise ValueError("query không được rỗng")

        lock_start = time.perf_counter()
        with self._runtime_lock:
            self._log_timing("runtime_lock_wait", time.perf_counter() - lock_start)
            hybrid = self._ensure_retrievers()
            retrieval_start = time.perf_counter()
            self._vram_log("before_embedding_request")
            candidates = hybrid.retrieve(raw_query, self.retrieval_config.candidate_pool_top_k)
            self._vram_log("after_embedding_request")
            self._log_timing("hybrid_retrieval", time.perf_counter() - retrieval_start)

            reranker_start = time.perf_counter()
            ranked = self._ensure_reranker().rerank(raw_query, candidates, self.retrieval_config.final_top_k)
            self._log_timing("reranker_total", time.perf_counter() - reranker_start)

        self._log_timing("end_to_end", time.perf_counter() - total_start)
        
        # Lưu ý: Cổng kiểm duyệt OOD đã hoàn tất (threshold 0.3).
        # answer_text dưới đây chỉ là chuỗi tạm thời (placeholder) trả về kết quả
        # rerank cho đến khi phase Generation v2 được tích hợp.
        answer_text = "Không có đủ thông tin trong tài liệu hiện có để trả lời câu hỏi này."
        refused = True
        top_score = 0.0
        chunks = []

        if ranked:
            top_score = ranked[0].score
            chunks = [c.chunk_id for c in ranked]
            if top_score < self.retrieval_config.ood_threshold:
                refused = True
                answer_text = "Không có đủ thông tin trong tài liệu hiện có để trả lời câu hỏi này."
            else:
                refused = False
                answer_text = f"[Placeholder cho Generation v2] Đã tìm thấy {len(ranked)} tài liệu phù hợp."

        return FinalResponse(
            answer=answer_text,
            refused=refused,
            top_score=top_score,
            chunks=chunks
        )
