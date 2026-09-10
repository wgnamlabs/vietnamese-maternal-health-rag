"""BM25 sparse retrieval using the tokenizer frozen in the artifact."""

from __future__ import annotations

import pickle
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import BaseRetriever, RetrievedChunk
from .config import RetrievalConfig


class SparseRetriever(BaseRetriever):
    """Retrieve chunks from the serialized BM25 index."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self.bm25, self.chunks, self.chunk_ids, self.tokenizer_name = (
            self._load_index(config.bm25_path)
        )
        self.tokenize = self._build_tokenizer(self.tokenizer_name)

    @staticmethod
    def _load_index(path: Path) -> tuple[Any, list[dict], list[str], str]:
        """Load and validate BM25 artifact plus stable chunk mapping."""
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy BM25 index: {path}")
        with path.open("rb") as file:
            data = pickle.load(file)
        if not isinstance(data, dict):
            raise ValueError("BM25 artifact phải là dictionary.")

        required = {"bm25", "chunks", "chunk_ids", "tokenizer"}
        missing = required.difference(data)
        if missing:
            raise ValueError(f"BM25 artifact thiếu field: {sorted(missing)}")
        chunks = data["chunks"]
        chunk_ids = [str(value) for value in data["chunk_ids"]]
        if len(chunks) != len(chunk_ids):
            raise ValueError("BM25 chunks và chunk_ids không cùng độ dài.")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("BM25 artifact có chunk_id trùng.")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or "text" not in chunk:
                raise ValueError(f"BM25 chunk #{index} thiếu text.")
            metadata = chunk.get("metadata") or {}
            if str(metadata.get("chunk_id", "")) != chunk_ids[index]:
                raise ValueError(
                    f"BM25 mapping không nhất quán tại index {index}: "
                    "metadata.chunk_id khác chunk_ids[index]."
                )
        return data["bm25"], chunks, chunk_ids, str(data["tokenizer"])

    @staticmethod
    def _build_tokenizer(name: str) -> Callable[[str], list[str]]:
        """Build đúng tokenizer được ghi trong BM25 artifact."""
        if name == "underthesea":
            try:
                from underthesea import word_tokenize
            except ImportError as exc:
                raise ImportError(
                    "BM25 index được build bằng underthesea nhưng package chưa có."
                ) from exc

            def tokenize(text: str) -> list[str]:
                normalized = unicodedata.normalize("NFC", str(text)).lower()
                return word_tokenize(normalized, format="text").split()

            return tokenize

        if name == "pyvi":
            try:
                from pyvi import ViTokenizer
            except ImportError as exc:
                raise ImportError(
                    "BM25 index được build bằng pyvi nhưng package chưa có."
                ) from exc

            def tokenize(text: str) -> list[str]:
                normalized = unicodedata.normalize("NFC", str(text)).lower()
                return ViTokenizer.tokenize(normalized).split()

            return tokenize

        if name == "whitespace":
            return lambda text: unicodedata.normalize(
                "NFC", str(text)
            ).lower().split()

        raise ValueError(f"Tokenizer BM25 không được hỗ trợ: {name}")

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Word-segment query và trả về top-k BM25 chunks."""
        if not query or not query.strip():
            raise ValueError("query không được rỗng")
        if top_k <= 0:
            raise ValueError("top_k phải > 0")

        tokenize_start = time.perf_counter()
        tokens = self.tokenize(query)
        tokenize_seconds = time.perf_counter() - tokenize_start
        bm25_start = time.perf_counter()
        scores = np.asarray(self.bm25.get_scores(tokens))
        bm25_seconds = time.perf_counter() - bm25_start
        order = np.argsort(-scores, kind="stable")[:top_k]
        results = []
        for index in order.tolist():
            chunk = self.chunks[index]
            results.append(
                RetrievedChunk(
                    chunk_id=self.chunk_ids[index],
                    text=str(chunk["text"]),
                    score=float(scores[index]),
                    source_type="sparse",
                    metadata=dict(chunk.get("metadata") or {}),
                )
            )
        for result in results:
            result.metadata["_timing"] = {
                "tokenize_seconds": tokenize_seconds,
                "bm25_seconds": bm25_seconds,
            }
        return results
