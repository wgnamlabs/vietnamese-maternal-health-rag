"""Evaluate dense, sparse, hybrid and reranked retrieval on the master benchmark.

Protocol:
    DEV is the default split and is used for configuration selection.
    TEST must be requested explicitly only after the final configuration is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (  # noqa: E402
    DEFAULT_EVIDENCE_MATCH,
    build_gold_relevance,
    complete_evidence_at_k,
    evidence_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    ood_auroc,
    reciprocal_rank,
)
from retrieval import (  # noqa: E402
    DenseRetriever,
    HybridRetriever,
    Reranker,
    RetrievedChunk,
    RetrievalConfig,
    SparseRetriever,
)
from retrieval.config import default_retrieval_config  # noqa: E402

K_VALUES = [3, 5, 10, 15]
PIPELINE_NAMES = ["dense", "sparse", "hybrid", "hybrid_rerank"]


def load_eval_questions(path: Path, split: str) -> list[dict[str, Any]]:
    """Load and validate one benchmark split."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    selected = [item for item in data if item.get("split") == split]
    if not selected:
        raise ValueError(f"Không có câu hỏi cho split={split!r}")
    ids = [item["id"] for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("eval_questions có ID trùng.")
    return selected


def load_bm25_chunks(path: Path) -> list[dict[str, Any]]:
    """Load chunk records used to build BM25 for metric gold mapping."""
    import pickle

    with path.open("rb") as file:
        data = pickle.load(file)
    chunks = data["chunks"]
    if len(chunks) != len(data["chunk_ids"]):
        raise ValueError("BM25 chunks và chunk_ids không cùng độ dài.")
    return chunks


def build_gold_maps(
    questions: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map every answerable question's evidence to frozen hierarchical chunks."""
    maps: dict[str, dict[str, Any]] = {}
    unmapped = []
    for item in questions:
        if not item.get("answerable", False):
            continue
        gold_map = build_gold_relevance(item, chunks)
        maps[item["id"]] = gold_map
        for evidence_id in gold_map["unmapped_evidence"]:
            unmapped.append((item["id"], evidence_id))
    if unmapped:
        raise RuntimeError(
            "Có gold evidence không map được sang BM25 chunks; "
            f"ví dụ={unmapped[:3]}"
        )
    return maps


def ranking_indices(
    results: Sequence[RetrievedChunk],
    chunk_id_to_index: dict[str, int],
) -> list[int]:
    """Convert retrieved stable chunk IDs to metric ranking indices."""
    missing = [chunk.chunk_id for chunk in results if chunk.chunk_id not in chunk_id_to_index]
    if missing:
        raise RuntimeError(f"Retriever trả chunk_id không có trong BM25 mapping: {missing[:3]}")
    return [chunk_id_to_index[chunk.chunk_id] for chunk in results]


def safe_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if not math.isnan(float(value))]
    return float(np.mean(values)) if values else float("nan")


def metric_row(
    ranking: Sequence[int],
    gold_map: dict[str, Any],
) -> dict[str, float]:
    """Calculate the existing retrieval metrics for one question."""
    row = {"mrr": reciprocal_rank(ranking, gold_map)}
    for k in K_VALUES:
        row[f"recall@{k}"] = evidence_recall_at_k(ranking, gold_map, k)
        row[f"complete@{k}"] = complete_evidence_at_k(ranking, gold_map, k)
        row[f"ndcg@{k}"] = ndcg_at_k(ranking, gold_map, k)
    return row


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Aggregate metric rows over answerable questions."""
    if not rows:
        return {}
    output: dict[str, float] = {"n_questions": len(rows)}
    for metric in ["mrr"] + [
        f"{name}@{k}"
        for name in ("recall", "complete", "ndcg")
        for k in K_VALUES
    ]:
        output[metric] = safe_mean(row[metric] for row in rows)
    return output


def group_questions(
    questions: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return overall and diagnostic slices used for retrieval tuning."""
    answerable = [item for item in questions if item.get("answerable", False)]
    groups: dict[str, list[dict[str, Any]]] = {"overall": answerable}
    slices = {
        "hard_multi": lambda item: (
            item.get("benchmark_group") == "hard_multi"
            or item.get("requires_multiple_evidence", False)
        ),
        "table": lambda item: item.get("requires_table", False),
        "numeric": lambda item: item.get("question_type") == "numeric",
        "patient_style": lambda item: item.get("question_style") == "patient_style",
        "multi_evidence": lambda item: item.get("requires_multiple_evidence", False),
    }
    for name, predicate in slices.items():
        selected = [item for item in answerable if predicate(item)]
        if selected:
            groups[name] = selected
    source_tags = sorted(
        {str(item.get("source_doc_tag")) for item in answerable if item.get("source_doc_tag")}
    )
    for source_tag in source_tags:
        groups[f"document:{source_tag}"] = [
            item for item in answerable if item.get("source_doc_tag") == source_tag
        ]
    return groups


def extract_timing(results: Sequence[RetrievedChunk]) -> dict[str, float]:
    """Extract stage timings attached by retrieval components."""
    if not results:
        return {}
    return {
        key: float(value)
        for key, value in results[0].metadata.get("_timing", {}).items()
    }


def serialize_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
    """Serialize a retrieval result for per-query debugging JSON."""
    metadata = {
        key: value
        for key, value in chunk.metadata.items()
        if key != "_timing"
    }
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "score": float(chunk.score),
        "source_type": chunk.source_type,
        "metadata": metadata,
    }


def print_markdown_table(
    summary: dict[str, Any],
    groups: Sequence[str],
    pipelines: Sequence[str],
) -> None:
    """Print copyable Recall/Complete/MRR/nDCG tables."""
    print("\n## Retrieval evaluation")
    print("\n### Overall and query slices")
    print(
        "| Pipeline | Group | N | K | Evidence Recall | Evidence Complete | MRR | nDCG |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for pipeline in pipelines:
        for group in groups:
            metrics = summary[pipeline]["groups"].get(group, {})
            if not metrics:
                continue
            for k in K_VALUES:
                print(
                    f"| {pipeline} | {group} | {int(metrics['n_questions'])} | {k} | "
                    f"{metrics[f'recall@{k}']:.4f} | "
                    f"{metrics[f'complete@{k}']:.4f} | "
                    f"{metrics['mrr']:.4f} | "
                    f"{metrics[f'ndcg@{k}']:.4f} |"
                )

    print("\n### Runtime")
    print("| Pipeline | N | Total runtime (s) | Mean/query (ms) |")
    print("|---|---:|---:|---:|")
    for pipeline in pipelines:
        runtime = summary[pipeline]["runtime"]
        print(
            f"| {pipeline} | {runtime['n_queries']} | "
            f"{runtime['total_seconds']:.3f} | {runtime['mean_ms']:.2f} |"
        )


def run_evaluation(
    questions_path: Path,
    output_dir: Path,
    eval_split: str,
    config: RetrievalConfig,
    selected_pipeline: str = "all",
) -> dict[str, Any]:
    """Run all retrieval configurations or only the selected final pipeline."""
    questions = load_eval_questions(questions_path, eval_split)
    sparse = SparseRetriever(config)
    chunks = sparse.chunks
    chunk_id_to_index = {
        str(chunk["metadata"]["chunk_id"]): index
        for index, chunk in enumerate(chunks)
    }
    gold_maps = build_gold_maps(questions, chunks)

    dense = DenseRetriever(config)
    hybrid = HybridRetriever(dense, sparse, config)

    per_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary: dict[str, Any] = {
        "eval_split": eval_split,
        "n_questions": len(questions),
        "n_in_domain": sum(1 for item in questions if item.get("answerable", False)),
        "n_ood": sum(1 for item in questions if not item.get("answerable", False)),
        "k_values": K_VALUES,
        "candidate_pool_top_k": config.candidate_pool_top_k,
        "final_top_k": config.final_top_k,
        "rrf_k": config.rrf_k,
        "reranker_model": config.reranker_model_name,
        "evidence_match": {
            "min_coverage": DEFAULT_EVIDENCE_MATCH.min_coverage,
            "min_contiguous_tokens": DEFAULT_EVIDENCE_MATCH.min_contiguous_tokens,
        },
    }

    candidate_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    pipeline_specs = (
        ("dense", "sparse", "hybrid")
        if selected_pipeline == "all"
        else ("hybrid",)
    )

    for pipeline_name in pipeline_specs:
        start = time.perf_counter()
        top1_id_scores: list[float] = []
        top1_ood_scores: list[float] = []
        all_rows: list[dict[str, Any]] = []

        for item in questions:
            query_start = time.perf_counter()
            if pipeline_name == "dense":
                results = dense.retrieve(item["question"], config.candidate_pool_top_k)
            elif pipeline_name == "sparse":
                results = sparse.retrieve(item["question"], config.candidate_pool_top_k)
            else:
                results = hybrid.retrieve(item["question"], config.candidate_pool_top_k)
            elapsed = time.perf_counter() - query_start
            ranking = ranking_indices(results, chunk_id_to_index)
            top1_score = float(results[0].score) if results else float("nan")

            row: dict[str, Any] = {
                "question_id": item["id"],
                "question": item["question"],
                "pipeline": pipeline_name,
                "eval_split": eval_split,
                "answerable": bool(item["answerable"]),
                "benchmark_group": item.get("benchmark_group", ""),
                "difficulty": item.get("difficulty", ""),
                "question_type": item.get("question_type", ""),
                "requires_table": bool(item.get("requires_table", False)),
                "requires_multiple_evidence": bool(
                    item.get("requires_multiple_evidence", False)
                ),
                "latency_ms": elapsed * 1000.0,
                "stage_timing_seconds": extract_timing(results),
                "top1_score": top1_score,
                "results": [serialize_chunk(chunk) for chunk in results],
                "ranking_chunk_ids": [chunk.chunk_id for chunk in results],
            }
            if item.get("answerable", False):
                row.update(metric_row(ranking, gold_maps[item["id"]]))
                top1_id_scores.append(top1_score)
            else:
                top1_ood_scores.append(top1_score)
            all_rows.append(row)
            candidate_rows[pipeline_name][item["id"]] = {
                "row": row,
                "results": results,
            }

        runtime = time.perf_counter() - start
        answerable_rows = [row for row in all_rows if row["answerable"]]
        groups = {}
        for group_name, group_items in group_questions(questions).items():
            ids = {item["id"] for item in group_items}
            groups[group_name] = aggregate(
                [row for row in answerable_rows if row["question_id"] in ids]
            )

        summary[pipeline_name] = {
            "groups": groups,
            "runtime": {
                "n_queries": len(all_rows),
                "total_seconds": runtime,
                "mean_ms": runtime / len(all_rows) * 1000.0,
                "mean_query_latency_ms": safe_mean(
                    row["latency_ms"] for row in all_rows
                ),
            },
            "ood": {
                "mean_top1_in_domain": safe_mean(top1_id_scores),
                "mean_top1_ood": safe_mean(top1_ood_scores),
                "auroc": ood_auroc(top1_id_scores, top1_ood_scores),
            },
        }
        per_query[pipeline_name] = all_rows

    reranker_load_start = time.perf_counter()
    reranker = Reranker(config)
    reranker_load_seconds = time.perf_counter() - reranker_load_start
    rerank_start = time.perf_counter()
    rerank_rows: list[dict[str, Any]] = []
    rerank_id_scores: list[float] = []
    rerank_ood_scores: list[float] = []
    for item in questions:
        query_start = time.perf_counter()
        candidates = candidate_rows["hybrid"][item["id"]]["results"]
        results = reranker.rerank(
            item["question"], candidates, config.candidate_pool_top_k
        )
        elapsed = time.perf_counter() - query_start
        ranking = ranking_indices(results, chunk_id_to_index)
        top1_score = float(results[0].score) if results else float("nan")
        hybrid_row = candidate_rows["hybrid"][item["id"]]["row"]
        row = {
            **hybrid_row,
            "pipeline": "hybrid_rerank",
            "latency_ms": hybrid_row["latency_ms"] + elapsed * 1000.0,
            "stage_timing_seconds": {
                **hybrid_row.get("stage_timing_seconds", {}),
                **extract_timing(results),
            },
            "reranker_inference_latency_ms": elapsed * 1000.0,
            "top1_score": top1_score,
            "results": [serialize_chunk(chunk) for chunk in results],
            "ranking_chunk_ids": [chunk.chunk_id for chunk in results],
        }
        if item.get("answerable", False):
            row.update(metric_row(ranking, gold_maps[item["id"]]))
            rerank_id_scores.append(top1_score)
        else:
            rerank_ood_scores.append(top1_score)
        rerank_rows.append(row)
    rerank_runtime = time.perf_counter() - rerank_start
    rerank_answerable_rows = [row for row in rerank_rows if row["answerable"]]
    rerank_groups = {}
    for group_name, group_items in group_questions(questions).items():
        ids = {item["id"] for item in group_items}
        rerank_groups[group_name] = aggregate(
            [row for row in rerank_answerable_rows if row["question_id"] in ids]
        )
    summary["hybrid_rerank"] = {
        "groups": rerank_groups,
        "runtime": {
            "n_queries": len(rerank_rows),
            "total_seconds": safe_mean(
                row["latency_ms"] for row in rerank_rows
            )
            / 1000.0
            * len(rerank_rows),
            "mean_ms": safe_mean(row["latency_ms"] for row in rerank_rows),
            "mean_query_latency_ms": safe_mean(row["latency_ms"] for row in rerank_rows),
            "candidate_generation_total_seconds": summary["hybrid"]["runtime"][
                "total_seconds"
            ],
            "reranker_inference_total_seconds": rerank_runtime,
            "reranker_inference_mean_ms": rerank_runtime
            / len(rerank_rows)
            * 1000.0,
            "model_load_seconds": reranker_load_seconds,
        },
        "stage_timing": {
            "model_load_seconds": reranker_load_seconds,
            "inference_total_seconds": rerank_runtime,
        },
        "ood": {
            "mean_top1_in_domain": safe_mean(rerank_id_scores),
            "mean_top1_ood": safe_mean(rerank_ood_scores),
            "auroc": ood_auroc(rerank_id_scores, rerank_ood_scores),
        },
    }
    per_query["hybrid_rerank"] = rerank_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"retrieval_per_query_{eval_split}.json"
    detail_path.write_text(
        json.dumps(
            {
                "eval_split": eval_split,
                "questions_path": str(questions_path),
                "pipelines": per_query,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path = output_dir / f"retrieval_summary_{eval_split}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    display_pipelines = (
        PIPELINE_NAMES if selected_pipeline == "all" else ["hybrid_rerank"]
    )
    reference_groups = summary[display_pipelines[0]]["groups"]
    preferred_groups = [
        "overall", "hard_multi", "table", "numeric", "patient_style", "multi_evidence",
    ]
    groups = [name for name in preferred_groups if name in reference_groups]
    groups.extend(
        name for name in sorted(reference_groups)
        if name.startswith("document:") and name not in groups
    )
    print_markdown_table(summary, groups, display_pipelines)
    print("\n### OOD AUROC")
    print("| Pipeline | OOD AUROC |")
    print("|---|---:|")
    for pipeline in display_pipelines:
        print(f"| {pipeline} | {summary[pipeline]['ood']['auroc']:.4f} |")
    print(f"\nChi tiết per-query: {detail_path}")
    print(f"Summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate dense/sparse/hybrid/reranked retrieval; default DEV."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Chỉ dùng test sau khi đã chốt cấu hình trên DEV.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "datasets" / "eval_questions.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "retrieval" / "retrieval_eval",
    )
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument(
        "--qdrant-collection",
        default=None,
        help="Tên collection Qdrant; dùng 'chunks' cho vector DB vừa build.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Đặt cùng device cho embedder và reranker (tương thích CLI cũ).",
    )
    parser.add_argument(
        "--embedder-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device cho Qwen3 embedding.",
    )
    parser.add_argument(
        "--reranker-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device cho cross-encoder reranker.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["all", "hybrid_rerank"],
        default="all",
        help=(
            "all: benchmark bốn pipeline; hybrid_rerank: chỉ profile "
            "pipeline đã chốt (Hybrid RRF Top-15 + rerank + Final K=5)."
        ),
    )
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--candidate-pool-top-k", type=int, default=None)
    parser.add_argument("--rrf-k", type=int, default=None)
    parser.add_argument("--final-top-k", type=int, default=None)
    parser.add_argument("--reranker-batch-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = default_retrieval_config(PROJECT_ROOT)
    config.qdrant.host = args.qdrant_host
    config.qdrant.port = args.qdrant_port
    if args.qdrant_collection is not None:
        config.qdrant.collection_name = args.qdrant_collection
    if args.device is not None:
        config.embedder.device = args.device
        config.reranker_device = args.device
    else:
        config.embedder.device = args.embedder_device
        config.reranker_device = args.reranker_device

    overrides = {
        "dense_top_k": args.dense_top_k,
        "sparse_top_k": args.sparse_top_k,
        "candidate_pool_top_k": args.candidate_pool_top_k,
        "rrf_k": args.rrf_k,
        "final_top_k": args.final_top_k,
        "reranker_batch_size": args.reranker_batch_size,
    }
    for name, value in overrides.items():
        if value is not None:
            if value <= 0:
                raise ValueError(f"{name} phải > 0")
            setattr(config, name, value)
    if config.final_top_k > config.candidate_pool_top_k:
        raise ValueError("final_top_k không được lớn hơn candidate_pool_top_k")

    print(
        "[Config] dense_top_k={}; sparse_top_k={}; candidate_pool_top_k={}; "
        "rrf_k={}; final_top_k={}; reranker_batch_size={}".format(
            config.dense_top_k,
            config.sparse_top_k,
            config.candidate_pool_top_k,
            config.rrf_k,
            config.final_top_k,
            config.reranker_batch_size,
        )
    )
    run_evaluation(
        questions_path=args.questions,
        output_dir=args.output_dir,
        eval_split=args.split,
        config=config,
        selected_pipeline=args.pipeline,
    )
