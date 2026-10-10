"""Construct the generation prompt without calling an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval.base import RetrievedChunk

from .context_builder import build_context, build_reference_sources
from .system_prompt import SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptPayload:
    """System/user messages and citation mapping for a generation call."""

    system_prompt: str
    user_message: str
    reference_map: dict[int, str]
    chunks: list[RetrievedChunk] = field(default_factory=list)
    reference_sources: dict[int, dict[str, str]] = field(default_factory=dict)


def build_prompt(
    query: str,
    chunks: list[RetrievedChunk],
) -> PromptPayload:
    """Build the basic RAG prompt from the user's original question and context."""
    context, reference_map = build_context(chunks)
    reference_sources = build_reference_sources(chunks)
    user_message = (
        "CÂU HỎI CỦA NGƯỜI DÙNG:\n"
        f"{query}\n\n"
        "NGỮ CẢNH:\n"
        f"{context}\n\n"
        "Hãy trả lời trực tiếp câu hỏi bằng văn bản tiếng Việt dựa trên NGỮ CẢNH."
    )
    return PromptPayload(
        SYSTEM_PROMPT,
        user_message,
        reference_map,
        list(chunks),
        reference_sources,
    )
