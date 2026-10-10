"""Lightweight Vietnamese-language guard for the future generation phase."""

from __future__ import annotations

import re

VIETNAMESE_MARKERS = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
VIETNAMESE_FUNCTION_WORDS = {
    "và", "của", "trong", "không", "có", "là", "được", "người", "mang",
    "thai", "bạn", "tài", "liệu", "ngày", "mỗi", "cần", "nên", "theo",
}


def is_vietnamese_response(text: str) -> bool:
    """Return a conservative signal for Vietnamese text, including numeric answers.

    Numeric/unit-heavy answers are accepted when Vietnamese context words are
    present; numbers and Latin abbreviations alone must not trigger a false
    positive or false negative decision.
    """
    words = set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))
    if words.intersection(VIETNAMESE_FUNCTION_WORDS):
        return True
    letters = [char.lower() for char in text if char.isalpha()]
    if not letters:
        return False
    marker_ratio = sum(char in VIETNAMESE_MARKERS for char in letters) / len(letters)
    return marker_ratio >= 0.02
