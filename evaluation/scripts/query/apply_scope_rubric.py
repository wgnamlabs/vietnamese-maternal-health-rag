"""Apply the fixed scope rubric to source and draft question sets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from query.rubric import ScopeRubric  # noqa: E402


def apply(path: Path, rubric: ScopeRubric) -> list[dict[str, Any]]:
    """Apply rubric and return auditable per-query records."""
    data = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for item in data:
        result = rubric.classify(str(item["question"]))
        output.append(
            {
                "id": item["id"],
                "question": item["question"],
                "benchmark_group": item.get("benchmark_group", ""),
                "predicted_label": result.predicted_label,
                "resolution_step": result.resolution_step,
                "matched_evidence": result.matched_evidence,
                "matched_pattern": result.matched_pattern,
                "method": result.method,
            }
        )
    return output


def report(name: str, rows: list[dict[str, Any]]) -> None:
    """Print counts and every unresolved question."""
    print(f"\n## {name}")
    print(json.dumps(Counter(row["predicted_label"] for row in rows), ensure_ascii=False))
    unresolved = [row for row in rows if row["predicted_label"] == "UNRESOLVED"]
    print(f"UNRESOLVED={len(unresolved)}")
    for row in unresolved:
        print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    """Run rubric application for source and optional draft files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "evaluation" / "datasets" / "eval_questions.json")
    parser.add_argument("--draft", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "phase3_query_processing" / "scope_labeling")
    args = parser.parse_args()
    rubric = ScopeRubric(PROJECT_ROOT / "src" / "query" / "rules")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = apply(args.questions, rubric)
    (args.output_dir / "scope_labels_raw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report("SOURCE", rows)
    if args.draft:
        draft_rows = apply(args.draft, rubric)
        (args.output_dir / "scope_labels_draft_raw.json").write_text(json.dumps(draft_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        report("DRAFT", draft_rows)


if __name__ == "__main__":
    main()
