"""Interfaces and result types for retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


@dataclass
class RetrievedChunk:
    """Một chunk được trả về bởi một retriever.

    Attributes:
        chunk_id: ID ổn định của chunk trong Qdrant/BM25.
        text: Nội dung chunk.
        score: Điểm xếp hạng của retriever hiện tại.
        source_type: Loại điểm sinh ra kết quả.
        metadata: Metadata gốc của chunk.
    """

    chunk_id: str
    text: str
    score: float
    source_type: Literal["dense", "sparse", "hybrid"]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract interface chung cho dense, sparse và hybrid retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Retrieve các chunk phù hợp nhất với query."""
        raise NotImplementedError
