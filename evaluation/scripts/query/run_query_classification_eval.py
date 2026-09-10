"""Evaluate classifier regression against frozen scope ground truth."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from query.classifier import QueryClassifier  # noqa: E402

LABELS = [
    "IN_SCOPE_INFO",
    "OUT_OF_SCOPE_PERSONALIZED",
    "OUT_OF_SCOPE_PRESCRIPTION",
    "OUT_OF_SCOPE_ADMIN",
    "EMERGENCY_REDFLAG",
    "UNRESOLVED",
]


def load_questions() -> dict[str, str]:
    """Load all question text sources keyed by ID."""
    result: dict[str, str] = {}
    for name in (
        "eval_questions.json",
        "scope_eval_questions_draft.json",
        "emergency_expansion_questions.json",
    ):
        path = PROJECT_ROOT / "evaluation" / name
        result.update({str(x["id"]): str(x["question"]) for x in json.loads(path.read_text(encoding="utf-8"))})
    return result


def print_matrix(expected: list[str], predicted: list[str]) -> None:
    """Print full 6x6 confusion matrix."""
    counts = Counter(zip(expected, predicted))
    print("\n### Confusion matrix")
    print("| Expected \\ Predicted | " + " | ".join(LABELS) + " |")
    print("|---|" + "---:|" * len(LABELS))
    for label in LABELS:
        print("| " + label + " | " + " | ".join(str(counts[(label, p)]) for p in LABELS) + " |")


def run_fallback_ablation(
    classifier: QueryClassifier,
    ground: dict[str, Any],
    questions: dict[str, str],
    output: Path,
) -> None:
    """Run a separate shortened-query LLM fallback ablation."""
    cases: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for record in ground["records"]:
        label = str(record["query_scope_label"])
        if label == "UNRESOLVED" or counts[label] >= 5:
            continue
        original = questions[record["id"]]
        shortened = " ".join(original.split()[:4])
        cases.append({"id": record["id"], "expected": label, "query": shortened})
        counts[label] += 1
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        result = classifier.classify_with_fallback(case["query"])
        rows.append({**case, "predicted": result.label, "method": result.method, "raw_llm_output": result.raw_llm_output})
    elapsed = time.perf_counter() - started
    accuracy = sum(row["expected"] == row["predicted"] for row in rows) / len(rows) if rows else 0.0
    print("\\n### LLM fallback ablation")
    print(f"Cases: {len(rows)}")
    print(f"Accuracy: {sum(row['expected'] == row['predicted'] for row in rows)}/{len(rows)} = {accuracy:.4f}")
    print(f"Fallback latency: {elapsed:.6f}s total, {elapsed / len(rows) * 1000:.3f}ms/query")
    (output / "fallback_ablation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Run regression and report emergency recall plus latency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "phase3_query_processing" / "query_classification_eval")
    parser.add_argument(
        "--run-llm-fallback",
        action="store_true",
        help="Run local Ollama fallback ablation on deliberately shortened Q5 cases.",
    )
    args = parser.parse_args()
    ground = json.loads((PROJECT_ROOT / "evaluation" / "datasets" / "query_classification_labels.json").read_text(encoding="utf-8"))
    questions = load_questions()
    classifier = QueryClassifier(
        PROJECT_ROOT / "src" / "query" / "rules",
        ground_truth_path=PROJECT_ROOT / "evaluation" / "datasets" / "query_classification_labels.json",
        audit_path=args.output / "llm_fallback_audit.jsonl",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    expected: list[str] = []
    predicted: list[str] = []
    per_query: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in ground["records"]:
        result = classifier.classify(questions[record["id"]])
        expected.append(record["query_scope_label"])
        predicted.append(result.label)
        per_query.append({"id": record["id"], "expected": record["query_scope_label"], "predicted": result.label, "resolution_step": result.resolution_step})
    rule_seconds = time.perf_counter() - started
    mismatches = [row for row in per_query if row["expected"] != row["predicted"]]
    print(f"Regression records: {len(per_query)}")
    print(f"Regression mismatches: {len(mismatches)}")
    print(f"Rule-based latency: {rule_seconds:.6f}s total, {rule_seconds / len(per_query) * 1000:.3f}ms/query")
    emergency_expected = [x for x in ground["records"] if x["query_scope_label"] == "EMERGENCY_REDFLAG"]
    emergency_correct = sum(row["expected"] == row["predicted"] == "EMERGENCY_REDFLAG" for row in per_query)
    print(f"EMERGENCY_REDFLAG recall regression: {emergency_correct}/{len(emergency_expected)} = {emergency_correct / len(emergency_expected):.4f}")
    print_matrix(expected, predicted)
    (args.output / "regression_per_query.json").write_text(json.dumps(per_query, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.run_llm_fallback:
        run_fallback_ablation(classifier, ground, questions, args.output)
    if mismatches:
        print(json.dumps(mismatches, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print("Regression status: PASS")


if __name__ == "__main__":
    main()
