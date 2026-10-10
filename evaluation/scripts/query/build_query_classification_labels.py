"""Build frozen, human-reviewed scope ground truth from rubric outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load(path: Path) -> list[dict[str, Any]]:
    """Load JSON records."""
    return json.loads(path.read_text(encoding="utf-8"))


def convert(
    rows: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    """Convert rubric output to frozen-label records."""
    return [
        {
            "id": row["id"],
            "query_scope_label": row["predicted_label"],
            "resolution_step": row["resolution_step"],
            "matched_evidence": row["matched_evidence"],
            "review_status": "human_reviewed",
            "source": source,
        }
        for row in rows
    ]


def main() -> None:
    """Write the official frozen ground-truth file."""
    result_dir = PROJECT_ROOT / "evaluation" / "results" / "phase3_query_processing" / "scope_labeling"
    original = load(result_dir / "scope_labels_raw.json")
    draft = load(result_dir / "scope_labels_draft_raw.json")
    expansion = load(result_dir / "scope_labeling" / "emergency_expansion_labels.json") if False else load(result_dir / "emergency_expansion_labels.json")
    records = convert(original, "original_dataset") + convert(
        draft, "draft_scope_eval"
    ) + convert(expansion, "emergency_expansion_set")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Ground truth có ID trùng.")
    for record in records:
        if record["id"] == "EVAL_ESC_001":
            record["reviewer_note"] = (
                "Băng huyết sau sinh mô tả định tính (không chỉ ngưỡng ml) — "
                "ban đầu bị bỏ sót do pattern coverage hẹp, được phát hiện và "
                "sửa qua vòng review emergency_pattern_coverage_expansion."
            )
    payload = {
        "schema_version": "scope_classification_ground_truth_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "human_review_rounds": 4,
        "review_scope": [
            "red_flag_boundary",
            "same_step_pattern_overlap",
            "Q1_first_person_vs_Q3_subject_omission",
            "emergency_pattern_coverage_expansion",
        ],
        "labeling_method": (
            "Deterministic Q1/Q1b/Q2/Q3/Q4 rubric plus human review; "
            "no LLM labels were used."
        ),
        "rubric_reference": "src/query/rubric.py",
        "change_log": {
            "before_expansion": {
                "original_total": 225,
                "original_emergency_redflag": 2,
                "frozen_total_before_expansion": 285,
                "coverage_issue": (
                    "EVAL_ESC_001 qualitative postpartum hemorrhage was not "
                    "matched by the narrow quantitative-only pattern."
                ),
            },
            "after_expansion": {
                "original_emergency_redflag": 3,
                "expansion_records": 20,
                "expansion_emergency_redflag": 12,
                "expansion_lookalike_in_scope": 8,
                "final_total": 305,
            },
        },
        "records": records,
    }
    output = PROJECT_ROOT / "evaluation" / "datasets" / "query_classification_labels.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
