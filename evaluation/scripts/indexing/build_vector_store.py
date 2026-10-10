"""Build the frozen Qdrant dense index and BM25 index from benchmark chunks.

Frozen DEV selection, confirmed on TEST:
    Chunker  : hierarchical
    Embedder : Qwen/Qwen3-Embedding-0.6B
    Distance : cosine
    Dim      : 1024
    Dense K  : 15 candidate chunks

The script resolves a model-specific chunk-cache directory so it cannot build
this selected index from chunks belonging to another benchmark candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np


# Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indexing.chunking.base import Chunk
from indexing.chunking.config import SELECTED_CHUNKER
from indexing.embedding import (
    MODELS_TO_COMPARE,
    EmbedderConfig,
    SELECTED_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from indexing.vector_store import (
    QdrantStoreConfig,
    QdrantVectorStore,
    SELECTED_COLLECTION_NAME,
    SELECTED_VECTOR_SIZE,
)


# ---------------------------------------------------------------------
# Frozen selection from DEV, confirmed on TEST
# ---------------------------------------------------------------------

DEFAULT_CHUNKER = SELECTED_CHUNKER
DEFAULT_EMB_MODEL = SELECTED_EMBEDDING_MODEL
DEFAULT_EXPECTED_DIM = SELECTED_VECTOR_SIZE
DEFAULT_COLLECTION = SELECTED_COLLECTION_NAME
DEFAULT_DISTANCE = "cosine"
DEFAULT_TOP_K = 15


def _is_notebook_cloud() -> bool:
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "COLAB_RELEASE_TAG" in os.environ
    )


def _resolve_qdrant_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    return "local" if _is_notebook_cloud() else "server"



def _selected_embedder_config(
    model_name: str,
    device: str,
) -> EmbedderConfig:
    """Dùng đúng config model đã dùng trong embedding benchmark."""
    for cfg in MODELS_TO_COMPARE:
        if cfg.model_name == model_name:
            return replace(cfg, device=device)

    # Cho phép experiment model khác, nhưng default/final vẫn là Qwen3-4B.
    print(
        f"[WARN] Model '{model_name}' không nằm trong MODELS_TO_COMPARE; "
        "dùng EmbedderConfig generic. Kết quả này không còn là cấu hình "
        "đã benchmark/freeze."
    )
    return EmbedderConfig(
        model_name=model_name,
        batch_size=16,
        max_seq_length=2048,
        device=device,
        trust_remote_code=False,
        auto_batch=True,
        min_batch_size=2,
    )


def _load_chunks(
    chunk_file: Path,
) -> Tuple[List[Chunk], list]:
    with chunk_file.open("r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError(
            f"{chunk_file} phải là JSON list không rỗng."
        )

    chunks: List[Chunk] = []
    chunk_ids = []

    for i, item in enumerate(raw_chunks):
        if not isinstance(item, dict):
            raise ValueError(f"Chunk #{i} không phải object.")
        if "text" not in item or "metadata" not in item:
            raise ValueError(
                f"Chunk #{i} thiếu 'text' hoặc 'metadata'."
            )

        metadata = item["metadata"] or {}
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError(
                f"Chunk #{i} thiếu metadata.chunk_id. "
                "Hãy dùng output chunking final đã QA."
            )

        chunks.append(
            Chunk(
                text=item["text"],
                metadata=metadata,
            )
        )
        chunk_ids.append(str(chunk_id))

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Chunk cache có chunk_id trùng; dừng build để tránh Qdrant overwrite."
        )

    return chunks, raw_chunks


def _validate_embeddings(
    embeddings: np.ndarray,
    n_chunks: int,
    expected_dim: int | None,
) -> int:
    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding phải là matrix 2-D, nhận {embeddings.shape}"
        )

    if embeddings.shape[0] != n_chunks:
        raise ValueError(
            f"Embeddings rows={embeddings.shape[0]} "
            f"khác chunks={n_chunks}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings có NaN/Inf.")

    actual_dim = int(embeddings.shape[1])

    if expected_dim is not None and actual_dim != expected_dim:
        raise RuntimeError(
            f"Embedding dimension thực tế={actual_dim}, "
            f"nhưng expected={expected_dim}. "
            "Không tạo Qdrant collection vì config/model có thể đã thay đổi."
        )

    return actual_dim


def _build_vn_tokenizer():
    """Tokenizer thống nhất để serialize BM25 corpus."""
    try:
        from underthesea import word_tokenize

        print("[BM25] tokenizer=underthesea")

        def tokenize(text: str):
            return word_tokenize(
                str(text).lower(),
                format="text",
            ).split()

        return tokenize, "underthesea"

    except ImportError:
        pass

    try:
        from pyvi import ViTokenizer

        print("[BM25] tokenizer=pyvi (fallback)")

        def tokenize(text: str):
            return ViTokenizer.tokenize(
                str(text).lower()
            ).split()

        return tokenize, "pyvi"

    except ImportError:
        pass

    print("[BM25] tokenizer=whitespace (fallback)")

    def tokenize(text: str):
        return str(text).lower().split()

    return tokenize, "whitespace"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_stores(
    base_path: Path,
    chunker_name: str = DEFAULT_CHUNKER,
    emb_model: str = DEFAULT_EMB_MODEL,
    expected_dim: int | None = DEFAULT_EXPECTED_DIM,
    recreate: bool = False,
    device: str = "cuda",
    qdrant_mode: str = "auto",
    host: str = "localhost",
    port: int = 6333,
    local_path: Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> None:

    print("=" * 72)
    print("BUILD FINAL VECTOR STORE")
    print("=" * 72)
    print(f"Chunker        : {chunker_name}")
    print(f"Embedding      : {emb_model}")
    print(f"Distance       : {DEFAULT_DISTANCE}")
    print(f"Recommended K  : {DEFAULT_TOP_K}")
    print()

    if (
        chunker_name != DEFAULT_CHUNKER
        or emb_model != DEFAULT_EMB_MODEL
    ):
        print(
            "[WARN] Bạn đang build cấu hình khác cấu hình đã chốt trên DEV:\n"
            f"       selected={DEFAULT_CHUNKER} + {DEFAULT_EMB_MODEL}\n"
            f"       current ={chunker_name} + {emb_model}\n"
        )

    # ---------------------------------------------------------------
    # Step 1: exact model-specific final chunk cache
    # ---------------------------------------------------------------
    chunk_file = (
        base_path
        / "evaluation"
        / "results"
        / "indexing"
        / "chunks_cache"
        / f"{chunker_name}.json"
    )

    if not chunk_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {chunk_file}.\n"
            "Hãy tạo chunk cache trước khi build vector store."
        )

    print(f"[Step 1] Load chunks: {chunk_file}")
    chunks, _ = _load_chunks(chunk_file)
    print(f"         chunks={len(chunks)}")

    # ---------------------------------------------------------------
    # Step 2: document embeddings
    # IMPORTANT: is_query=False exactly as benchmark
    # ---------------------------------------------------------------
    emb_config = _selected_embedder_config(
        emb_model,
        device=device,
    )

    print("[Step 2] Embedding chunks")
    print(
        f"         batch_size={emb_config.batch_size}, "
        f"auto_batch={emb_config.auto_batch}, "
        f"max_seq_length={emb_config.max_seq_length}"
    )

    embedder = SentenceTransformerEmbedder(emb_config)
    try:
        embedder.load()
        embeddings = embedder.encode(
            [c.text for c in chunks],
            is_query=False,
        )
    finally:
        embedder.unload()

    actual_dim = _validate_embeddings(
        embeddings,
        n_chunks=len(chunks),
        expected_dim=expected_dim,
    )

    print(
        f"         shape={embeddings.shape}, dim={actual_dim}"
    )

    # ---------------------------------------------------------------
    # Step 3: Qdrant
    # ---------------------------------------------------------------
    resolved_mode = _resolve_qdrant_mode(qdrant_mode)

    qdrant_path = None
    memory = False

    if resolved_mode == "local":
        qdrant_path = local_path or (
            base_path
            / "data"
            / "vector_db"
            / "qdrant_local"
        )
        qdrant_path.mkdir(parents=True, exist_ok=True)
    elif resolved_mode == "memory":
        memory = True
    elif resolved_mode != "server":
        raise ValueError(
            "qdrant_mode phải là auto|server|local|memory"
        )

    qdrant_config = QdrantStoreConfig(
        collection_name=collection_name,
        host=host,
        port=port,
        vector_size=actual_dim,
        distance=DEFAULT_DISTANCE,
        path=str(qdrant_path) if qdrant_path else None,
        memory=memory,
    )

    print(
        f"[Step 3] Qdrant mode={qdrant_config.mode}, "
        f"collection={collection_name}"
    )

    store = QdrantVectorStore(qdrant_config)
    store.load()
    store.create_collection(recreate=recreate)
    store.add(
        chunks,
        embeddings,
        chunker_type=chunker_name,
    )
    store.persist()

    qdrant_info = store.collection_info()
    print(
        f"         points={qdrant_info['points_count']}, "
        f"dim={qdrant_info['vector_size']}"
    )

    # ---------------------------------------------------------------
    # Step 4: BM25 over the SAME hierarchical chunks
    # ---------------------------------------------------------------
    print("[Step 4] Build BM25 index")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "Thiếu rank-bm25. Cài bằng: pip install rank-bm25"
        ) from exc

    tokenize, tokenizer_name = _build_vn_tokenizer()
    tokenized_corpus = [
        tokenize(c.text)
        for c in chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    vector_db_dir = base_path / "data" / "vector_db"
    vector_db_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = vector_db_dir / "bm25_index.pkl"

    bm25_data = {
        "bm25": bm25,
        "chunks": [
            {
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
        "chunk_ids": [
            c.metadata["chunk_id"]
            for c in chunks
        ],
        "chunker_type": chunker_name,
        "emb_model": emb_model,
        "tokenizer": tokenizer_name,
        "recommended_top_k": DEFAULT_TOP_K,
    }

    with bm25_path.open("wb") as f:
        pickle.dump(bm25_data, f)

    print(f"         saved={bm25_path}")

    # ---------------------------------------------------------------
    # Step 5: reproducibility manifest
    # ---------------------------------------------------------------
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_on_dev": (
            chunker_name == DEFAULT_CHUNKER
            and emb_model == DEFAULT_EMB_MODEL
        ),
        "chunker": chunker_name,
        "embedding_model": emb_model,
        "embedding_config": asdict(emb_config),
        "embedding_dimension": actual_dim,
        "distance": DEFAULT_DISTANCE,
        "recommended_dense_top_k": DEFAULT_TOP_K,
        "chunk_count": len(chunks),
        "chunk_cache": str(chunk_file.relative_to(base_path)),
        "chunk_cache_sha256": _sha256(chunk_file),
        "qdrant": qdrant_info,
        "bm25": {
            "path": str(bm25_path.relative_to(base_path)),
            "tokenizer": tokenizer_name,
        },
        "benchmark_test": {
            "evidence_recall_at_10": 0.930,
            "evidence_complete_at_10": 0.930,
            "ndcg_at_10": 0.734,
            "mrr": 0.706,
            "ood_auroc": 0.995,
            "evidence_recall_at_15": 0.972,
        },
    }

    manifest_path = vector_db_dir / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[Step 5] Manifest: {manifest_path}")

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"Qdrant collection : {collection_name}\n"
        f"Qdrant mode       : {qdrant_config.mode}\n"
        f"Vector dimension  : {actual_dim}\n"
        f"Chunker           : {chunker_name}\n"
        f"Embedding         : {emb_model}\n"
        f"BM25              : {bm25_path}\n"
        f"Manifest          : {manifest_path}"
    )

    if qdrant_config.dashboard_url:
        print(
            f"Dashboard         : "
            f"{qdrant_config.dashboard_url}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen Qdrant + BM25 index. "
            "Default = hierarchical + Qwen/Qwen3-Embedding-0.6B."
        )
    )

    parser.add_argument(
        "--chunker",
        choices=["semantic", "hierarchical"],
        default=DEFAULT_CHUNKER,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMB_MODEL,
        help="Hugging Face embedding model.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=DEFAULT_EXPECTED_DIM,
        help=(
            "Chỉ dùng để validate output dimension. "
            "Selected Qwen3-Embedding-0.6B dimension = 1024."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Xóa collection cũ trước khi build. "
            "NÊN dùng khi chuyển từ index/model cũ."
        ),
    )

    parser.add_argument(
        "--qdrant-mode",
        choices=["auto", "server", "local", "memory"],
        default="auto",
        help=(
            "auto: Kaggle/Colab=local, môi trường khác=server."
        ),
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
    )
    parser.add_argument(
        "--host",
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=None,
        help="Path cho Qdrant local-file mode.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_stores(
        base_path=PROJECT_ROOT,
        chunker_name=args.chunker,
        emb_model=args.model,
        expected_dim=args.expected_dim,
        recreate=args.recreate,
        device=args.device,
        qdrant_mode=args.qdrant_mode,
        host=args.host,
        port=args.port,
        local_path=args.local_path,
        collection_name=args.collection,
    )
