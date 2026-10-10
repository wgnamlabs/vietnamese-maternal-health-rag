"""Construct the generation prompt without calling an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval.base import RetrievedChunk

from .context_builder import build_context
from .system_prompt import SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptPayload:
    """System/user messages and citation mapping for a generation call."""

    system_prompt: str
    user_message: str
    reference_map: dict[int, str]
    chunks: list[RetrievedChunk] = field(default_factory=list)


def build_prompt(
    query: str,
    rewritten_query: str,
    chunks: list[RetrievedChunk],
) -> PromptPayload:
    """Build prompt using original question while retaining rewrite for audit."""
    context, reference_map = build_context(chunks)
    user_message = (
        "CÂU HỎI GỐC CỦA NGƯỜI DÙNG:\n"
        f"{query}\n\n"
        "QUERY ĐÃ CHUẨN HÓA CHỈ DÙNG CHO RETRIEVAL:\n"
        f"{rewritten_query}\n\n"
        "NGỮ CẢNH:\n"
        f"{context}\n\n"
        "Hãy trả về duy nhất JSON hợp lệ theo schema đã nêu trong system prompt."
    )
    return PromptPayload(SYSTEM_PROMPT, user_message, reference_map, list(chunks))
