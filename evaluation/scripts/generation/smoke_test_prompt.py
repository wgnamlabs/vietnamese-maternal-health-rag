"""Real retrieval smoke test for Phase 4 prompt construction."""

from __future__ import annotations

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import gc
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from query.rewriter import QueryRewriter  # noqa: E402
from retrieval.config import default_retrieval_config  # noqa: E402
from retrieval.dense_retriever import DenseRetriever  # noqa: E402
from retrieval.fusion import HybridRetriever  # noqa: E402
from retrieval.reranker import Reranker  # noqa: E402
from retrieval.sparse_retriever import SparseRetriever  # noqa: E402
from generation.prompt_builder import build_prompt  # noqa: E402


def load_questions() -> list[dict]:
    """Load evaluation questions."""
    return json.loads((PROJECT_ROOT / "evaluation" / "datasets" / "eval_questions.json").read_text(encoding="utf-8"))


def main() -> None:
    """Retrieve real candidates, rerank final Top-5 and print full prompts."""
    questions = load_questions()
    table = next(q for q in questions if q.get("requires_table") and q.get("answerable"))
    ood = next(q for q in questions if q.get("benchmark_group") == "ood")
    candidate_questions = [
        q for q in questions
        if q.get("answerable") and q.get("benchmark_group") in {"standard", "hard_multi"}
    ]
    config = default_retrieval_config(PROJECT_ROOT)
    dense = DenseRetriever(config)
    sparse = SparseRetriever(config)
    hybrid = HybridRetriever(dense, sparse, config)
    rewriter = QueryRewriter()
    pools: dict[str, tuple[dict, list]] = {}
    for question in candidate_questions[:30] + [table, ood]:
        rewritten = rewriter.rewrite(question["question"], "IN_SCOPE_INFO").rewritten_query
        pools[question["id"]] = (question, hybrid.retrieve(rewritten, config.candidate_pool_top_k))
    dense.embedder.unload()
    del dense, hybrid
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reranker = Reranker(config)
    final: dict[str, tuple[dict, list]] = {}
    for question_id, (question, pool) in pools.items():
        ranked = reranker.rerank(question["question"], pool, config.final_top_k)
        sources = [str(x.metadata.get("source", "")) for x in ranked]
        breadcrumbs = [str(x.metadata.get("breadcrumb", "")) for x in ranked]
        same_source_sections = any(
            sources.count(source) >= 2
            and len({breadcrumbs[i] for i, value in enumerate(sources) if value == source}) >= 2
            for source in set(sources)
        )
        table_hit = any(bool(x.metadata.get("is_table")) for x in ranked)
        if question_id == table["id"] or question_id == ood["id"] or same_source_sections or table_hit:
            final[question_id] = (question, ranked)
    selected: list[tuple[str, tuple[dict, list]]] = []
    same = next(((qid, item) for qid, item in final.items() if qid not in {table["id"], ood["id"]} and any(True for _ in [1]) and any(item[1][i].metadata.get("source") == item[1][j].metadata.get("source") and item[1][i].metadata.get("breadcrumb") != item[1][j].metadata.get("breadcrumb") for i in range(len(item[1])) for j in range(i + 1, len(item[1])))), None)
    selected.append(("same_document_different_sections", same or next(iter(final.items()))))
    selected.append(("table_query", (table["id"], final[table["id"]])))
    selected.append(("ood_query", (ood["id"], final[ood["id"]])))
    output: list[dict] = []
    for case, (qid, (question, chunks)) in selected:
        rewritten = rewriter.rewrite(question["question"], "IN_SCOPE_INFO").rewritten_query
        payload = build_prompt(question["question"], rewritten, chunks)
        output.append({"case": case, "question_id": qid, "question": question["question"], "prompt": {"system_prompt": payload.system_prompt, "user_message": payload.user_message, "reference_map": payload.reference_map}, "chunks": [{"chunk_id": x.chunk_id, "source": x.metadata.get("source"), "breadcrumb": x.metadata.get("breadcrumb"), "is_table": x.metadata.get("is_table"), "score": float(x.score),} for x in chunks]})
    path = PROJECT_ROOT / "evaluation" / "results" / "phase5_generation" / "generation_eval" / "phase4_smoke_prompts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
