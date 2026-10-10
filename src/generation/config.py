"""Configuration for local answer generation and deterministic score gating."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime configuration for Phase 5 generation."""

    # GPU-resident mode: Ollama is asked to offload all possible layers to GPU.
    # The runtime validates actual placement with `ollama ps`; on an 8 GB GPU
    # Ollama may still partially offload to CPU when all resident models do not fit.
    model_name: str = "qwen3.5:2b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.1
    enable_thinking: bool = False
    num_predict: int = 2048
    num_ctx: int = 8192
    num_gpu: int = -1
    # Ollama requires a duration with a unit; 24h keeps the model resident
    # for the intended local session without triggering the `-1` parse error.
    keep_alive: str = "24h"
    # Calibrated on evaluation/datasets/answerable_labels.json (2026-09-25).
    score_gate_threshold: float = 0.60
    max_retries: int = 2
    language_retry_count: int = 1
    # Citation reference-number repair has a separate retry budget so malformed
    # JSON cannot consume retries intended for an otherwise valid answer.
    citation_retry_count: int = 2
    use_few_shot_examples: bool = True
    enable_faithfulness_llm: bool = False

    def __post_init__(self) -> None:
        """Reject generation settings that cannot reserve output capacity."""
        if self.num_ctx <= 0:
            raise ValueError("num_ctx phải > 0")
        if self.num_predict <= 0:
            raise ValueError("num_predict phải > 0")
        if self.num_predict >= self.num_ctx:
            raise ValueError("num_predict phải nhỏ hơn num_ctx để chừa chỗ cho prompt")
