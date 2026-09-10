"""Local Ollama answer generation with deterministic score gating."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import GenerationConfig
from .language_check import is_vietnamese_response
from .prompt_builder import PromptPayload
from .schema import GenerationOutput


@dataclass(frozen=True)
class GenerationResult:
    """Generation result plus audit metadata."""

    output: GenerationOutput
    method: str
    retries: int
    latency_seconds: float
    invalid_references: list[int]
    raw_output: str | None = None


class AnswerGenerator:
    """Generate validated Vietnamese answers using local Ollama."""

    def __init__(self, config: GenerationConfig | None = None, log_dir: Path | None = None) -> None:
        self.config = config or GenerationConfig()
        self.log_dir = log_dir or Path("evaluation/results/phase5_generation/generation_eval")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt_payload: PromptPayload, top1_score: float) -> GenerationResult:
        """Apply score gate, then call Ollama with JSON output and validation retries."""
        started = time.perf_counter()
        if top1_score < self.config.score_gate_threshold:
            result = GenerationResult(
                output=GenerationOutput(
                    answer="Không có đủ thông tin trong tài liệu hiện có để trả lời câu hỏi này.",
                    citations=[],
                    insufficient_evidence=True,
                    confidence="low",
                ),
                method="score_gate",
                retries=0,
                latency_seconds=time.perf_counter() - started,
                invalid_references=[],
            )
            self._log(prompt_payload, top1_score, result)
            return result

        raw = ""
        retries = 0
        language_attempts = 0
        last_error = ""
        output: GenerationOutput | None = None
        for attempt in range(self.config.max_retries + 1):
            retries = attempt
            output = None
            raw = self._call(prompt_payload, last_error)
            try:
                candidate = GenerationOutput.model_validate(json.loads(raw))
                if not is_vietnamese_response(candidate.answer):
                    if language_attempts < self.config.language_retry_count:
                        language_attempts += 1
                        last_error = "answer không đạt kiểm tra tiếng Việt; hãy viết lại hoàn toàn bằng tiếng Việt"
                        continue
                    last_error = "answer không đạt kiểm tra tiếng Việt sau retry"
                    continue
                output = candidate
                break
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = f"JSON/schema validation error: {exc}"
        if output is None:
            raise RuntimeError(f"Generation thất bại sau {retries + 1} lần thử: {last_error}; raw={raw}")

        valid_refs = set(prompt_payload.reference_map)
        invalid = [citation.reference_number for citation in output.citations if citation.reference_number not in valid_refs]
        if invalid:
            output = output.model_copy(update={"confidence": "low"})
            self._append_jsonl("invalid_reference_flags.jsonl", {"invalid_references": invalid, "raw_output": raw})
        result = GenerationResult(output, "llm_generated", retries, time.perf_counter() - started, invalid, raw)
        self._log(prompt_payload, top1_score, result)
        return result

    def _call(self, payload: PromptPayload, error: str) -> str:
        user = payload.user_message
        if error:
            user += f"\n\nSỬA LỖI LẦN TRƯỚC: {error}"
        request_body = {
            "model": self.config.model_name,
            "messages": [{"role": "system", "content": payload.system_prompt}, {"role": "user", "content": user}],
            "think": self.config.enable_thinking,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "num_ctx": self.config.num_ctx,
            },
            "format": "json",
            "stream": False,
        }
        # Older ollama Python clients do not expose `think`; use the HTTP API
        # while preserving the same local Ollama endpoint and request fields.
        import requests
        response = requests.post(
            f"{self.config.ollama_host.rstrip('/')}/api/chat",
            json=request_body,
            timeout=900,
        )
        response.raise_for_status()
        response = response.json()
        if not isinstance(response, dict):
            raise RuntimeError(f"Ollama HTTP response không phải object: {response!r}")
        content = response.get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Ollama trả về response không có message.content: {response!r}")
        return str(content)

    def _append_jsonl(self, name: str, data: dict[str, Any]) -> None:
        with (self.log_dir / name).open("a", encoding="utf-8") as file:
            file.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _log(self, payload: PromptPayload, score: float, result: GenerationResult) -> None:
        chunk_audit = {
            chunk.chunk_id: {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": chunk.score,
                "source_type": chunk.source_type,
            }
            for chunk in payload.chunks
        }
        self._append_jsonl("generation_log.jsonl", {"query": payload.user_message, "top1_score": score, "method": result.method, "retries": result.retries, "latency_seconds": result.latency_seconds, "output": result.output.model_dump(), "raw_output": result.raw_output, "reference_map": payload.reference_map, "retrieved_chunks": chunk_audit})
