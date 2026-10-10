"""Production query safety classifier backed by the approved rubric."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from .rubric import ScopeRubric

ClassificationLabel = Literal[
    "IN_SCOPE_INFO",
    "OUT_OF_SCOPE_PERSONALIZED",
    "OUT_OF_SCOPE_PRESCRIPTION",
    "OUT_OF_SCOPE_ADMIN",
    "EMERGENCY_REDFLAG",
    "UNRESOLVED",
]


@dataclass(frozen=True)
class ClassificationResult:
    """Auditable classification result."""

    label: ClassificationLabel
    resolution_step: str
    matched_evidence: list[str]
    method: Literal["rule_based", "llm_fallback"]
    matched_pattern: list[str]
    raw_llm_output: str | None = None


class QueryClassifier:
    """Classify queries using the frozen deterministic rubric.

    The rule-based path delegates to :class:`ScopeRubric` without duplicating
    patterns or business logic. LLM fallback is only eligible for ambiguous
    Q5 defaults and can never override a rule-based emergency result.

    Fallback limitations observed in the evaluation:
        The measured 28-second/query latency came from a general-purpose 4B
        model through Ollama for a simple classification task. A smaller
        1.5B–3B model or constrained JSON/structured output should be tested
        before concluding that fallback is infeasible for production.
        The 32% ablation accuracy is a lower-bound stress result because the
        test queries were intentionally truncated, making them harder than
        naturally ambiguous user queries. Importantly, 0/5 emergency cases
        were redirected to another label; most became UNRESOLVED, which is the
        intended fail-safe behavior and produced no dangerous false negative.
        These results are suggestion-only and pending human review; they do
        not establish production infeasibility.

    """

    _AMBIGUOUS_TERMS = (
        "là gì",
        "thế nào",
        "ra sao",
        "có sao không",
        "làm sao",
        "giúp tôi",
        "nguy hiểm không",
    )

    def __init__(
        self,
        rules_dir: Path,
        ground_truth_path: Path | None = None,
        audit_path: Path | None = None,
        llm_callable: Callable[[str], str] | None = None,
    ) -> None:
        self.rubric = ScopeRubric(rules_dir)
        self.ground_truth_path = ground_truth_path
        self.audit_path = audit_path
        self._llm_callable = llm_callable
        self._few_shot: list[dict[str, str]] | None = None

    def classify(self, query: str) -> ClassificationResult:
        """Classify one query using rule-based rubric logic."""
        result = self.rubric.classify(query)
        return ClassificationResult(
            label=result.predicted_label,
            resolution_step=result.resolution_step,
            matched_evidence=result.matched_evidence,
            matched_pattern=result.matched_pattern,
            method="rule_based",
        )

    def classify_batch(self, queries: list[str]) -> list[ClassificationResult]:
        """Classify a batch without changing rule evaluation order."""
        return [self.classify(query) for query in queries]

    @classmethod
    def _is_ambiguous_q5(cls, query: str, result: ClassificationResult) -> bool:
        """Use explicit, conservative heuristics to select fallback candidates."""
        if result.label != "IN_SCOPE_INFO" or result.resolution_step != "Q5":
            return False
        normalized = " ".join(query.strip().lower().split())
        if len(normalized.split()) <= 4:
            return True
        return any(term in normalized for term in cls._AMBIGUOUS_TERMS) and len(
            normalized.split()
        ) <= 10

    def classify_with_fallback(
        self,
        query: str,
        confidence_threshold: float = 0.8,
    ) -> ClassificationResult:
        """Use configured LLM only for explicitly ambiguous Q5 queries.

        Args:
            query: User query.
            confidence_threshold: Minimum LLM confidence required.

        Returns:
            Rule-based result when fallback is ineligible or unavailable;
            otherwise an auditable LLM-suggested result.
        """
        rule_result = self.classify(query)
        if rule_result.label == "EMERGENCY_REDFLAG":
            return rule_result
        if not self._is_ambiguous_q5(query, rule_result):
            return rule_result

        prompt = self._build_prompt(query, confidence_threshold)
        started = time.perf_counter()
        raw = self._call_llm(prompt)
        result = self._parse_llm_result(raw, confidence_threshold)
        self._write_audit(query, prompt, raw, result, time.perf_counter() - started)
        return result

    def _call_llm(self, prompt: str) -> str:
        if self._llm_callable is not None:
            return self._llm_callable(prompt)
        model = os.getenv("QUERY_CLASSIFIER_MODEL", "").strip()
        if not model:
            raise RuntimeError(
                "LLM fallback chưa được cấu hình: đặt QUERY_CLASSIFIER_MODEL "
                "hoặc truyền llm_callable."
            )
        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError("Cần package ollama để dùng LLM fallback.") from exc
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        response = Client(host=host).chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
            format="json",
        )
        return str(response["message"]["content"])

    def _load_few_shot(self) -> list[dict[str, str]]:
        if self._few_shot is not None:
            return self._few_shot
        if self.ground_truth_path is None:
            return []
        payload = json.loads(self.ground_truth_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        questions: dict[str, str] = {}
        for path in (
            self.ground_truth_path.parent / "eval_questions.json",
            self.ground_truth_path.parent / "scope_eval_questions_draft.json",
            self.ground_truth_path.parent / "emergency_expansion_questions.json",
        ):
            if path.exists():
                for item in json.loads(path.read_text(encoding="utf-8")):
                    questions[str(item["id"])] = str(item["question"])
        selected: list[dict[str, str]] = []
        counts: dict[str, int] = {}
        for record in records:
            label = str(record["query_scope_label"])
            if counts.get(label, 0) >= 5 or record["id"] not in questions:
                continue
            selected.append({"question": questions[record["id"]], "label": label})
            counts[label] = counts.get(label, 0) + 1
        self._few_shot = selected
        return selected

    def _build_prompt(self, query: str, threshold: float) -> str:
        examples = "\n".join(
            f"Example: {item['question']}\nLabel: {item['label']}"
            for item in self._load_few_shot()
        )
        labels = ", ".join(
            [
                "IN_SCOPE_INFO",
                "OUT_OF_SCOPE_PERSONALIZED",
                "OUT_OF_SCOPE_PRESCRIPTION",
                "OUT_OF_SCOPE_ADMIN",
                "EMERGENCY_REDFLAG",
                "UNRESOLVED",
            ]
        )
        return (
            "Classify the Vietnamese query using the fixed Q1-Q5 rubric. "
            "Return JSON only with label and confidence (0..1). "
            f"Allowed labels: {labels}. Required confidence >= {threshold}.\n"
            "Emergency is only for explicit first-person/current red-flag "
            "situations; never invent a red flag.\n\n"
            f"{examples}\n\nQuery: {query}"
        )

    @staticmethod
    def _parse_llm_result(
        raw: str,
        confidence_threshold: float,
    ) -> ClassificationResult:
        try:
            data = json.loads(raw)
            label = str(data.get("label", "UNRESOLVED"))
            confidence = float(data.get("confidence", 0.0))
        except (ValueError, TypeError, json.JSONDecodeError):
            label, confidence = "UNRESOLVED", 0.0
        allowed = {
            "IN_SCOPE_INFO",
            "OUT_OF_SCOPE_PERSONALIZED",
            "OUT_OF_SCOPE_PRESCRIPTION",
            "OUT_OF_SCOPE_ADMIN",
            "EMERGENCY_REDFLAG",
            "UNRESOLVED",
        }
        if label not in allowed or confidence < confidence_threshold:
            label = "UNRESOLVED"
        if label == "EMERGENCY_REDFLAG":
            # Fallback is never allowed to create/override an emergency.
            label = "UNRESOLVED"
        return ClassificationResult(
            label=label,  # type: ignore[arg-type]
            resolution_step="LLM_FALLBACK",
            matched_evidence=["LLM fallback suggestion; pending human review."],
            matched_pattern=[],
            method="llm_fallback",
            raw_llm_output=raw,
        )

    def _write_audit(
        self,
        query: str,
        prompt: str,
        raw: str,
        result: ClassificationResult,
        elapsed: float,
    ) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "query": query,
            "prompt": prompt,
            "raw_output": raw,
            "result": result.__dict__,
            "latency_seconds": elapsed,
            "review_status": "pending_human_review",
        }
        with self.audit_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
