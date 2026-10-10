"""Pydantic output contracts for the future generation/verification phases."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation at claim/sentence level for Phase 6 verification.

    Phase 6 can split ``answer`` into claims, map each claim to its cited
    reference number and chunk ID, then compare extracted numbers/units in the
    claim with the original chunk text. Citations therefore must not collapse
    the whole answer into one citation.
    """

    claim: str = Field(..., description="Câu hoặc mệnh đề cụ thể trong answer.")
    reference_number: int = Field(..., ge=1)
    chunk_id: str = Field(..., min_length=1)


class GenerationOutput(BaseModel):
    """Strict JSON contract expected from the Phase 5 generator."""

    answer: str
    citations: list[Citation]
    insufficient_evidence: bool
    confidence: Literal["high", "low"]
