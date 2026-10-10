"""Configuration for local answer generation and deterministic score gating."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime configuration for Phase 5 generation."""

    model_name: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.1
    enable_thinking: bool = False
    num_predict: int = 4096
    num_ctx: int = 8192
    score_gate_threshold: float = 0.3
    max_retries: int = 2
    language_retry_count: int = 1
    enable_faithfulness_llm: bool = False
