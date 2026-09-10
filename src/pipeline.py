"""End-to-end local pipeline for the maternal-health RAG assistant."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import torch

from generation.config import GenerationConfig
from generation.generator import AnswerGenerator, GenerationResult
from generation.prompt_builder import build_prompt
from query.classifier import QueryClassifier
from query.rewriter import QueryRewriter
from query.templates import get_template
from retrieval.config import RetrievalConfig, default_retrieval_config
from retrieval.dense_retriever import DenseRetriever
from retrieval.fusion import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.sparse_retriever import SparseRetriever


@dataclass(frozen=True)
class FinalResponse:
    """User-facing answer plus citations and auditable metadata."""

    answer: str
    citations: list[dict]
    disclaimer: str
    method: str
    metadata: dict


class RAGPipeline:
    """Run safety routing, retrieval, reranking, and local generation.

    Retrieval models are initialized lazily after a query is classified as
    in-scope. This keeps deterministic emergency and out-of-scope responses
    independent of Qdrant, BM25, GPU, and Ollama availability.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.retrieval_config: RetrievalConfig = default_retrieval_config(project_root)
        self.classifier = QueryClassifier(project_root / "src" / "query" / "rules")
        self.rewriter = QueryRewriter()
        self.generator = AnswerGenerator(
            GenerationConfig(),
            project_root / "evaluation" / "results" / "phase5_generation" / "generation_eval",
        )
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
        path = (
            self.project_root
            / "evaluation"
            / "results"
            / "phase5_generation"
            / "vram_lifecycle.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _ensure_retrievers(self) -> HybridRetriever:
        """Initialize the local retrieval stack on demand."""
        if self.dense is None or self.sparse is None or self.hybrid is None:
            self.dense = DenseRetriever(self.retrieval_config)
            self.sparse = SparseRetriever(self.retrieval_config)
            self.hybrid = HybridRetriever(self.dense, self.sparse, self.retrieval_config)
        return self.hybrid

    def _release_retrieval_models(self) -> None:
        """Release embedding/retrieval objects before reranking or generation."""
        if self.dense is not None:
            self._vram_log("before_embedding_unload")
            self.dense.embedder.unload()
            self._vram_log("after_embedding_unload")
        self.dense = None
        self.sparse = None
        self.hybrid = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_reranker(self) -> None:
        """Release the cross-encoder before local answer generation."""
        if self.reranker is not None:
            self._vram_log("before_reranker_unload")
            self.reranker.unload()
            self._vram_log("after_reranker_unload")
        self.reranker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def answer_query(self, raw_query: str) -> FinalResponse:
        """Answer one query through the local RAG workflow."""
        classification = self.classifier.classify(raw_query)
        if classification.label == "UNRESOLVED":
            return FinalResponse(
                answer=(
                    "Tôi chưa thể xác định chắc chắn yêu cầu của bạn từ câu hỏi hiện tại. "
                    "Vui lòng diễn đạt rõ hơn hoặc liên hệ nhân viên y tế để được hỗ trợ phù hợp."
                ),
                citations=[],
                disclaimer="Thông tin chỉ mang tính tham khảo.",
                method="unresolved",
                metadata={"classification": classification.__dict__},
            )
        if classification.label != "IN_SCOPE_INFO":
            return FinalResponse(
                answer=get_template(classification.label),
                citations=[],
                disclaimer="Thông tin chỉ mang tính tham khảo.",
                method="template",
                metadata={"classification": classification.__dict__},
            )

        # The model lifecycle is stateful to fit 8 GB VRAM. Serialize
        # in-scope requests so one request cannot unload another's model.
        with self._runtime_lock:
            rewritten = self.rewriter.rewrite(raw_query, classification.label)
            hybrid = self._ensure_retrievers()
            self._vram_log("before_embedding_request")
            candidates = hybrid.retrieve(
                rewritten.rewritten_query,
                self.retrieval_config.candidate_pool_top_k,
            )
            self._release_retrieval_models()

            self._vram_log("before_reranker_load")
            self.reranker = Reranker(self.retrieval_config)
            self._vram_log("after_reranker_load")
            ranked = self.reranker.rerank(
                raw_query,
                candidates,
                self.retrieval_config.final_top_k,
            )
            self._release_reranker()

            prompt_payload = build_prompt(raw_query, rewritten.rewritten_query, ranked)
            self._vram_log("before_generation")
            result: GenerationResult = self.generator.generate(
                prompt_payload,
                ranked[0].score if ranked else 0.0,
            )

        if result.method == "score_gate":
            return FinalResponse(
                answer=result.output.answer,
                citations=[],
                disclaimer="Thông tin chỉ mang tính tham khảo.",
                method="score_gated",
                metadata={
                    "classification": classification.__dict__,
                    "top1_score": ranked[0].score if ranked else None,
                    "insufficient_evidence": True,
                },
            )

        return FinalResponse(
            answer=result.output.answer,
            citations=[citation.model_dump() for citation in result.output.citations],
            disclaimer="Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y tế trực tiếp.",
            method="llm_generated",
            metadata={
                "classification": classification.__dict__,
                "reference_map": prompt_payload.reference_map,
                "top1_score": ranked[0].score if ranked else None,
                "insufficient_evidence": result.output.insufficient_evidence,
                "confidence": result.output.confidence,
            },
        )
