"""Dense retrieval backed by the existing embedder and Qdrant store."""

from __future__ import annotations

import time

from indexing.embedding import SentenceTransformerEmbedder
from indexing.vector_store import QdrantVectorStore

from .base import BaseRetriever, RetrievedChunk
from .config import RetrievalConfig


class DenseRetriever(BaseRetriever):
    """Encode query bằng query instruction rồi search Qdrant."""

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: SentenceTransformerEmbedder | None = None,
        store: QdrantVectorStore | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder or SentenceTransformerEmbedder(config.embedder)
        self.store = store or QdrantVectorStore(config.qdrant)
        self.store.load()

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve top-k chunk bằng dense cosine similarity."""
        if not query or not query.strip():
            raise ValueError("query không được rỗng")
        if top_k <= 0:
            raise ValueError("top_k phải > 0")

        embedding_start = time.perf_counter()
        query_vector = self.embedder.encode([query], is_query=True)[0]
        embedding_seconds = time.perf_counter() - embedding_start

        qdrant_start = time.perf_counter()
        raw_results = self.store.search(query_vector, top_k=top_k)
        qdrant_seconds = time.perf_counter() - qdrant_start
        results = [
            RetrievedChunk(
                chunk_id=str(item["chunk_id"]),
                text=str(item.get("text", "")),
                score=float(item["score"]),
                source_type="dense",
                metadata=dict(item.get("metadata") or {}),
            )
            for item in raw_results
        ]
        for result in results:
            result.metadata["_timing"] = {
                "embedding_seconds": embedding_seconds,
                "qdrant_seconds": qdrant_seconds,
            }
        return results
