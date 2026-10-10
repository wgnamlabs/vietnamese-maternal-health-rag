"""Deterministic scope-labeling rubric for query audit and evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ScopeLabel = Literal[
    "IN_SCOPE_INFO",
    "OUT_OF_SCOPE_PERSONALIZED",
    "OUT_OF_SCOPE_PRESCRIPTION",
    "OUT_OF_SCOPE_ADMIN",
    "EMERGENCY_REDFLAG",
    "UNRESOLVED",
]


@dataclass(frozen=True)
class ScopeRubricResult:
    """Kết quả áp dụng rubric deterministic trên một query."""

    predicted_label: ScopeLabel
    resolution_step: str
    matched_evidence: list[str]
    matched_pattern: list[str]
    method: Literal["rule_based"] = "rule_based"


class ScopeRubric:
    """Áp dụng Q1 → Q1b → Q2 → Q3 → Q4, mặc định Q5 là in-scope.

    Governance decisions:
        Q1 requires an explicit first-person/current signal because a false
        emergency block incorrectly rejects otherwise valid education.
        Q3 intentionally does not require first-person wording. Vietnamese
        advice requests commonly omit the subject, and missing a prescription
        request is a higher safety risk than a conservative block. Q3 matches
        a treatment decision/action (start, stop, switch, or change dose) tied
        to a concrete medication situation; objective medication information
        without an action request remains eligible for Q5.
    """

    def __init__(self, rules_dir: Path) -> None:
        self.rules_dir = rules_dir
        self.red_flags = self._load_rules(rules_dir / "red_flags.yaml")
        self.out_of_scope = self._load_rules(
            rules_dir / "out_of_scope_patterns.yaml"
        )

    @staticmethod
    def _load_rules(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Rule file phải là mapping: {path}")
        return data

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize NFC và lowercase, giữ nguyên dấu tiếng Việt."""
        return unicodedata.normalize("NFC", text).lower()

    @staticmethod
    def _patterns(group: dict[str, Any]) -> list[dict[str, Any]]:
        values = group.get("patterns", [])
        if not isinstance(values, list):
            raise ValueError("patterns phải là list")
        return [item for item in values if isinstance(item, dict)]

    def _match_group(
        self,
        text: str,
        group: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Return one evidence per matched sub-pattern, without duplicates."""
        matches: list[tuple[str, str]] = []
        for pattern in self._patterns(group):
            candidates = [str(pattern.get("regex", "")).strip()]
            candidates.extend(
                re.escape(str(value).strip())
                for value in pattern.get("variants", [])
                if str(value).strip()
            )
            for regex in candidates:
                if not regex:
                    continue
                match = re.search(regex, text, flags=re.IGNORECASE | re.UNICODE)
                if match:
                    matches.append((str(pattern.get("id", regex)), match.group(0)))
                    break
        return matches

    @staticmethod
    def _result(
        label: ScopeLabel,
        step: str,
        matches: list[tuple[str, str]],
    ) -> ScopeRubricResult:
        """Build an auditable result; same-step matches are aggregated."""
        return ScopeRubricResult(
            predicted_label=label,
            resolution_step=step,
            matched_evidence=[evidence for _, evidence in matches],
            matched_pattern=[pattern for pattern, _ in matches],
        )

    @staticmethod
    def _has_explicit_first_person_current_signal(text: str) -> bool:
        """Implement Q1b using explicit first-person/current-action signals.

        Non-personal conditional/protocol forms intentionally do not match:
        ``nếu thai phụ...``, ``khi trẻ...`` and ``đang tắm mà bé...``.
        """
        marker = (
            r"(?:\b(?:tôi|em|mình)\b(?!\s+b[eé])|"
            r"\b(?:con|bé|vợ|chồng|nhà)\s+(?:tôi|em)\b)"
        )
        current_state = r"(?:đang|bị|vừa)"
        current_match = re.search(
            rf"{marker}.{{0,100}}\b{current_state}\b|"
            rf"\b{current_state}\b.{{0,100}}{marker}",
            text,
            flags=re.IGNORECASE | re.UNICODE,
        )
        if current_match:
            prefix = text[: current_match.start()]
            if re.search(r"\b(?:nếu|khi)\s*$", prefix):
                return False
            return True

        return bool(
            re.search(
                rf"{marker}.{{0,100}}\b(?:phải|cần)\s+làm\s+gì\b",
                text,
                flags=re.IGNORECASE | re.UNICODE,
            )
        )

    def classify(self, query: str) -> ScopeRubricResult:
        """Classify query theo Q1/Q1b rồi Q2-Q5."""
        normalized = self.normalize(query)

        # Q1 + Q1b: red-flag alone is not enough; current explicit first-person
        # signal is required to escalate. Same-step Q1 matches are aggregated.
        red_flag_matches = self._match_group(normalized, self.red_flags)
        if red_flag_matches and self._has_explicit_first_person_current_signal(
            normalized
        ):
            return self._result(
                "EMERGENCY_REDFLAG", "Q1", red_flag_matches
            )

        step_groups = (
            ("Q2", "personalized", "OUT_OF_SCOPE_PERSONALIZED"),
            ("Q3", "prescription", "OUT_OF_SCOPE_PRESCRIPTION"),
            ("Q4", "admin", "OUT_OF_SCOPE_ADMIN"),
        )
        matched_steps: list[tuple[str, str, list[tuple[str, str]]]] = []
        for step, key, label in step_groups:
            matches = self._match_group(
                normalized, self.out_of_scope.get(key, {})
            )
            if matches:
                matched_steps.append((step, label, matches))

        # Multiple matches inside one Q are certain at that step and are
        # aggregated. Ambiguity is reserved for matches across Q steps.
        if len(matched_steps) > 1:
            all_matches = [
                match
                for _, _, matches in matched_steps
                for match in matches
            ]
            return self._result(
                "UNRESOLVED",
                "/".join(step for step, _, _ in matched_steps),
                all_matches,
            )
        if matched_steps:
            step, label, matches = matched_steps[0]
            return self._result(label, step, matches)  # type: ignore[arg-type]

        return ScopeRubricResult(
            predicted_label="IN_SCOPE_INFO",
            resolution_step="Q5",
            matched_evidence=["Không khớp Q1–Q4; mặc định theo rubric."],
            matched_pattern=["default_q5"],
        )

    def classify_batch(self, queries: list[str]) -> list[ScopeRubricResult]:
        """Classify nhiều query theo cùng một rubric."""
        return [self.classify(query) for query in queries]
