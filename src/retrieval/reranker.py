"""Cross-encoder reranking for retrieval candidates."""

from __future__ import annotations

import gc
import time

import torch
from sentence_transformers import CrossEncoder

from .base import RetrievedChunk
from .config import RetrievalConfig


def _gpu_memory(tag: str) -> None:
    """Log PyTorch CUDA allocation for lifecycle auditing."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[Reranker] {tag}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


class Reranker:
    """Rerank candidate chunks with a multilingual cross-encoder."""

    def __init__(
        self,
        config: RetrievalConfig,
        model: CrossEncoder | None = None,
    ) -> None:
        self.config = config
        device = config.reranker_device
        if device == "cuda" and not torch.cuda.is_available():
            print("[Reranker] CUDA không khả dụng, chuyển sang CPU.")
            device = "cpu"
        self.device = device
        load_start = time.perf_counter()
        kwargs = {}
        if device == "cuda":
            kwargs["automodel_args"] = {"torch_dtype": torch.float16}
        self.model = model or CrossEncoder(
            config.reranker_model_name,
            device=device,
            max_length=1024,
            **kwargs
        )
        self.load_seconds = time.perf_counter() - load_start
        print(
            f"[Reranker] device={self.device}; load={self.load_seconds:.3f}s; "
            "resident=true"
        )
        if self.device == "cuda":
            _gpu_memory("after_load")
        else:
            print(
                "[Reranker] CPU mode; GPU memory shown by the embedder remains "
                "allocated to the resident embedding model."
            )

    def unload(self) -> None:
        """Release model, Python references and CUDA allocator memory."""
        _gpu_memory("before_unload")
        model = self.model
        self.model = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _gpu_memory("after_unload")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank candidate pool and overwrite scores with cross-encoder scores."""
        if not query or not query.strip():
            raise ValueError("query không được rỗng")
        if top_k <= 0:
            raise ValueError("top_k phải > 0")
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        inference_start = time.perf_counter()
        scores = self.model.predict(
            pairs,
            batch_size=self.config.reranker_batch_size,
            show_progress_bar=False,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        print(
            f"[Reranker] device={self.device}; candidates={len(chunks)}; "
            f"inference={inference_seconds:.3f}s"
        )
        scored = []
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
            scored.append(chunk)
        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        for result in scored:
            result.metadata["_timing"] = {
                **result.metadata.get("_timing", {}),
                "reranker_load_seconds": self.load_seconds,
                "reranker_inference_seconds": inference_seconds,
            }
        return scored[:top_k]
