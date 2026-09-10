"""Run one real end-to-end generation smoke test before full DEV evaluation."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generation.config import GenerationConfig  # noqa: E402
from generation.generator import AnswerGenerator  # noqa: E402
from generation.prompt_builder import build_prompt  # noqa: E402
from query.classifier import QueryClassifier  # noqa: E402
from query.rewriter import QueryRewriter  # noqa: E402
from retrieval.config import default_retrieval_config  # noqa: E402
from retrieval.dense_retriever import DenseRetriever  # noqa: E402
from retrieval.fusion import HybridRetriever  # noqa: E402
from retrieval.reranker import Reranker  # noqa: E402
from retrieval.sparse_retriever import SparseRetriever  # noqa: E402


def main() -> None:
    """Run classifier → retrieval → prompt → Ollama → Pydantic parse once."""
    questions = json.loads(
        (PROJECT_ROOT / "evaluation" / "datasets" / "eval_questions.json").read_text(
            encoding="utf-8"
        )
    )
    question = next(
        item
        for item in questions
        if item.get("split") == "dev"
        and item.get("answerable") is True
    )
    raw_query = str(question["question"])
    classifier = QueryClassifier(PROJECT_ROOT / "src" / "query" / "rules")
    classification = classifier.classify(raw_query)
    if classification.label != "IN_SCOPE_INFO":
        raise RuntimeError(f"Smoke query không IN_SCOPE_INFO: {classification}")

    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite(raw_query, classification.label)
    config = default_retrieval_config(PROJECT_ROOT)
    dense = DenseRetriever(config)
    sparse = SparseRetriever(config)
    hybrid = HybridRetriever(dense, sparse, config)
    candidates = hybrid.retrieve(
        rewritten.rewritten_query, config.candidate_pool_top_k
    )
    dense.embedder.unload()
    del dense, hybrid
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reranker = Reranker(config)
    ranked = reranker.rerank(raw_query, candidates, 5)
    reranker.unload()
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload = build_prompt(raw_query, rewritten.rewritten_query, ranked)
    generator = AnswerGenerator(
        GenerationConfig(),
        PROJECT_ROOT / "evaluation" / "results" / "phase5_generation" / "generation_eval",
    )
    result = generator.generate(payload, ranked[0].score if ranked else 0.0)

    print("=== SINGLE GENERATION SMOKE TEST ===")
    print(f"question_id: {question['id']}")
    print(f"classification: {classification}")
    print(f"rewritten_query: {rewritten.rewritten_query}")
    print("=== PROMPT SYSTEM ===")
    print(payload.system_prompt)
    print("=== PROMPT USER ===")
    print(payload.user_message)
    print("=== REFERENCE MAP ===")
    print(json.dumps(payload.reference_map, ensure_ascii=False, indent=2))
    print("=== RAW OLLAMA RESPONSE ===")
    print(result.raw_output or "(score gate; no Ollama call)")
    print("=== PARSED GENERATION OUTPUT ===")
    print(result.output.model_dump_json(indent=2, ensure_ascii=False))
    print("=== GENERATION METADATA ===")
    print(json.dumps({
        "method": result.method,
        "retries": result.retries,
        "latency_seconds": result.latency_seconds,
        "invalid_references": result.invalid_references,
        "top1_score": ranked[0].score if ranked else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
