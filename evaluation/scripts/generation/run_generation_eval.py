"""Phase 5 generation evaluation on DEV plus OOD_020."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generation.config import GenerationConfig  # noqa: E402
from generation.generator import AnswerGenerator  # noqa: E402
from generation.prompt_builder import build_prompt  # noqa: E402
from query.rewriter import QueryRewriter  # noqa: E402
from retrieval.config import default_retrieval_config  # noqa: E402
from retrieval.dense_retriever import DenseRetriever  # noqa: E402
from retrieval.fusion import HybridRetriever  # noqa: E402
from retrieval.reranker import Reranker  # noqa: E402
from retrieval.sparse_retriever import SparseRetriever  # noqa: E402


def main() -> None:
    """Run 30 answerable DEV cases and OOD_020 generation check."""
    questions = json.loads((PROJECT_ROOT / "evaluation/datasets/eval_questions.json").read_text(encoding="utf-8"))
    in_scope_ids = {
        "EVAL_QD_776_002", "EVAL_QD_776_003", "EVAL_QD_776_004", "EVAL_QD_776_005",
        "EVAL_QD_776_009", "EVAL_QD_776_013", "EVAL_QD_776_017", "EVAL_QD_1139_002",
        "EVAL_QD_1139_005", "EVAL_QD_1139_009", "EVAL_QD_1139_012", "EVAL_QD_1139_015",
        "EVAL_QD_1154_003", "EVAL_QD_1154_008", "EVAL_QD_1359_006", "EVAL_QD_1359_009",
        "EVAL_QD_1359_013", "EVAL_QD_1359_017", "EVAL_QD_1359_020", "EVAL_QD_1359_024",
        "EVAL_QD_1470_006", "EVAL_QD_1470_010", "EVAL_QD_2323_002", "EVAL_QD_2323_005",
        "EVAL_QD_2323_009", "EVAL_QD_2323_012", "EVAL_QD_4128_006", "EVAL_QD_4128_010",
        "EVAL_QD_4128_013", "EVAL_QD_4128_017",
    }
    answerable = [q for q in questions if q["id"] in in_scope_ids]
    ood = next(q for q in questions if q["id"] == "OOD_020")
    config = default_retrieval_config(PROJECT_ROOT)
    dense = DenseRetriever(config)
    sparse = SparseRetriever(config)
    hybrid = HybridRetriever(dense, sparse, config)
    rewriter = QueryRewriter()
    cases = []
    for q in answerable + [ood]:
        query = q["question"]
        rq = rewriter.rewrite(query, "IN_SCOPE_INFO").rewritten_query
        cases.append((q, hybrid.retrieve(rq, config.candidate_pool_top_k)))
    dense.embedder.unload()
    del dense, hybrid
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    reranker = Reranker(config)
    generator = AnswerGenerator(GenerationConfig(), PROJECT_ROOT / "evaluation/results/phase5_generation/generation_eval")
    rows = []
    for q, pool in cases:
        ranked = reranker.rerank(q["question"], pool, 5)
        if ranked and ranked[0].score < generator.config.score_gate_threshold:
            result = generator.generate(build_prompt(q["question"], q["question"], ranked), ranked[0].score)
        else:
            result = generator.generate(build_prompt(q["question"], q["question"], ranked), ranked[0].score if ranked else 0.0)
        rows.append({"id": q["id"], "expected_group": q.get("benchmark_group"), "top1_score": ranked[0].score if ranked else 0.0, "method": result.method, "retries": result.retries, "latency_seconds": result.latency_seconds, "output": result.output.model_dump()})
        print(json.dumps(rows[-1], ensure_ascii=False))
    out = PROJECT_ROOT / "evaluation/results/phase5_generation/generation_eval/generation_eval_dev.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
