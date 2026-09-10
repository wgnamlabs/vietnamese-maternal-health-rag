"""Conservative dictionary-based query rewriting for in-scope queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteResult:
    """Original and normalized/expanded query."""

    original_query: str
    rewritten_query: str


class QueryRewriter:
    """Rewrite without LLM free-form generation or intent changes."""

    ABBREVIATIONS = {
        "bs": "bác sĩ",
        "bsi": "bác sĩ",
        "t.hợp": "trường hợp",
        "thai kì": "thai kỳ",
        "sản phụ": "phụ nữ mang thai",
    }
    SYNONYMS = {
        "thai": "thai bào thai em bé trong bụng",
        "em bé trong bụng": "thai bào thai",
        "bé sơ sinh": "trẻ sơ sinh",
        "ra máu": "chảy máu xuất huyết",
        "cao huyết áp": "tăng huyết áp",
    }

    @staticmethod
    def _normalize(text: str) -> str:
        return unicodedata.normalize("NFC", text).lower().strip()

    def rewrite(self, query: str, label: str) -> RewriteResult:
        """Rewrite only IN_SCOPE_INFO queries."""
        if label != "IN_SCOPE_INFO":
            return RewriteResult(query, query)
        rewritten = self._normalize(query)
        for source, target in self.ABBREVIATIONS.items():
            rewritten = re.sub(rf"\b{re.escape(source)}\b", target, rewritten)
        expansions = []
        for source, target in self.SYNONYMS.items():
            if source in rewritten:
                expansions.append(target)
        if expansions:
            rewritten = f"{rewritten} {' '.join(expansions)}"
        return RewriteResult(query, rewritten)
