"""Local Ollama answer generation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import GenerationConfig
from .prompt_builder import PromptPayload


@dataclass(frozen=True)
class GenerationResult:
    """Generation result."""
    answer: str
    latency_seconds: float


class AnswerGenerator:
    """Generate Vietnamese answers using local Ollama."""

    _MAX_TRUNCATION_PASSES = 20
    _TRUNCATION_MARKER = "[... nội dung đã rút gọn do giới hạn context ...]"

    def __init__(self, config: GenerationConfig | None = None, log_dir: Path | None = None) -> None:
        self.config = config or GenerationConfig()
        self.log_dir = log_dir or Path("evaluation/results/generation/runtime_audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._generation_warmed = False

    @property
    def generation_warmed(self) -> bool:
        """Whether Ollama has accepted the model warm-up request."""
        return self._generation_warmed

    def _system_prompt(self, prompt: str) -> str:
        """Optionally remove the labelled few-shot section for controlled A/B runs."""
        if self.config.use_few_shot_examples:
            return prompt
        return re.sub(
            r"\n?<BEGIN_FEW_SHOT_EXAMPLES>.*?<END_FEW_SHOT_EXAMPLES>\n?",
            "",
            prompt,
            flags=re.DOTALL,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Conservatively estimate Vietnamese prompt tokens without a new runtime dependency."""
        return max(1, round(len(re.findall(r"\S+", text)) * 1.3)) if text else 0

    def _fit_prompt_budget(self, system: str, user: str) -> str:
        """Trim the longest context block when prompt budget leaves no output room."""
        maximum = max(1, self.config.num_ctx - self.config.num_predict)
        block_pattern = re.compile(
            r"(?P<prefix>(?:\n\n)?\[\d+\] Văn bản:.*?Nội dung nguyên văn:\n)"
            r"(?P<text>.*?)"
            r"(?P<suffix>(?=\n\n\[\d+\] Văn bản:|\n\nHãy trả lời|\Z))",
            flags=re.DOTALL,
        )
        matches = list(block_pattern.finditer(user))
        if not matches:
            return user

        blocks = [
            {"prefix": match.group("prefix"), "text": match.group("text"), "suffix": match.group("suffix")}
            for match in matches
        ]
        leading = user[: matches[0].start()]
        trailing = user[matches[-1].end() :]

        def render() -> str:
            return leading + "".join(
                f"{block['prefix']}{block['text']}{block['suffix']}"
                for block in blocks
            ) + trailing

        for _ in range(self._MAX_TRUNCATION_PASSES):
            rendered = render()
            if self._estimate_tokens(system + "\n" + rendered) <= maximum:
                break
            candidates = [
                (index, block)
                for index, block in enumerate(blocks)
                if self._TRUNCATION_MARKER not in str(block["text"])
            ]
            if not candidates:
                break
            index, longest = max(candidates, key=lambda item: len(str(item[1]["text"])))
            words = str(longest["text"]).split()
            if len(words) <= 16:
                break
            keep = max(16, int(len(words) * 0.8))
            if keep >= len(words):
                keep = len(words) - 1
            blocks[index]["text"] = " ".join(words[:keep]) + "\n" + self._TRUNCATION_MARKER
        
        return render()

    def _post_ollama(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Call local Ollama with actionable network failures."""
        import requests

        try:
            response = requests.post(
                f"{self.config.ollama_host.rstrip('/')}/api/chat",
                json=request_body,
                timeout=900,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError("Không kết nối được Ollama — kiểm tra service đang chạy chưa") from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Ollama timeout sau 900s") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc}") from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Ollama HTTP response không phải object: {payload!r}")
        return payload

    def warm_up(self) -> None:
        """Load the Ollama model before serving the first user request."""
        started = time.perf_counter()
        request_body = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": "Trả về {}"}],
            "think": self.config.enable_thinking,
            "options": {
                "temperature": 0.0,
                "num_predict": 1,
                "num_ctx": min(self.config.num_ctx, 512),
                "num_gpu": self.config.num_gpu,
            },
            "keep_alive": self.config.keep_alive,
            "stream": False,
        }
        self._post_ollama(request_body)
        self._generation_warmed = True
        print(
            f"[Generation] warm_up model={self.config.model_name}; "
            f"elapsed={time.perf_counter() - started:.3f}s"
        )

    def generate(self, prompt_payload: PromptPayload, top1_score: float) -> GenerationResult:
        """Basic generation without score gates or retries."""
        started = time.perf_counter()
        
        raw = self._call(prompt_payload)
        
        result = GenerationResult(
            answer=raw.strip(),
            latency_seconds=time.perf_counter() - started,
        )
        print(f"[Generation] model={self.config.model_name}; total={result.latency_seconds:.3f}s")
        return result

    def _call(self, payload: PromptPayload) -> str:
        """Call Ollama after ensuring a bounded prompt leaves output capacity."""
        user = payload.user_message
        system = self._system_prompt(payload.system_prompt)
        user = self._fit_prompt_budget(system, user)
        request_body = {
            "model": self.config.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "think": self.config.enable_thinking,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "num_ctx": self.config.num_ctx,
                "num_gpu": self.config.num_gpu,
            },
            "keep_alive": self.config.keep_alive,
            "stream": False,
        }
        response = self._post_ollama(request_body)
        content = response.get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Ollama trả về response không có message.content: {response!r}")
        return str(content)
