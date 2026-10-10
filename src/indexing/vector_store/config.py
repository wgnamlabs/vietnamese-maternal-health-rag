from dataclasses import dataclass
from typing import Optional


# Must match the collection used by the final build command.
SELECTED_COLLECTION_NAME = "chunks"
SELECTED_VECTOR_SIZE = 1024


@dataclass
class QdrantStoreConfig:
    """Qdrant configuration for the selected production retrieval index.

    Frozen benchmark selection:
        Chunker  : hierarchical
        Embedder : Qwen/Qwen3-Embedding-0.6B
        Distance : cosine
        Dim      : 1024
    """

    collection_name: str = SELECTED_COLLECTION_NAME
    host: str = "localhost"
    port: int = 6333
    vector_size: int = SELECTED_VECTOR_SIZE
    distance: str = "cosine"

    # path != None uses embedded/local-file Qdrant.
    path: Optional[str] = None

    # memory=True is only suitable for short-lived tests.
    memory: bool = False

    def __post_init__(self) -> None:
        if self.memory and self.path:
            raise ValueError("Chỉ chọn một trong hai: memory=True hoặc path=...")

        if self.vector_size <= 0:
            raise ValueError("vector_size phải > 0")

        self.distance = self.distance.lower().strip()
        if self.distance not in {"cosine", "dot", "euclidean"}:
            raise ValueError(
                "distance phải là một trong: cosine | dot | euclidean"
            )

        if self.port <= 0:
            raise ValueError("port phải > 0")

    @property
    def mode(self) -> str:
        if self.memory:
            return "memory"
        if self.path:
            return "local"
        return "server"

    @property
    def dashboard_url(self) -> Optional[str]:
        if self.mode != "server":
            return None
        return f"http://{self.host}:{self.port}/dashboard"
