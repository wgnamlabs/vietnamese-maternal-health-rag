"""FastAPI web application for the local maternal-health RAG assistant.

Run from the project root with Qdrant and Ollama available locally.
"""

from __future__ import annotations

import sys
import time
from difflib import SequenceMatcher
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import FinalResponse, RAGPipeline  # noqa: E402

STATIC_DIR = PROJECT_ROOT / "static"
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline
    pipeline = RAGPipeline(PROJECT_ROOT)
    pipeline.warm_up()
    yield


app = FastAPI(title="Vietnamese Maternal-Health RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """User-facing response returned by the web API."""

    answer: str
    citations: list[dict]
    sources: list[dict]
    method: str
    verification_status: str | None = None
    insufficient_evidence: bool = False
    latency: dict[str, float]


def _get_pipeline() -> RAGPipeline:
    """Return the singleton local runtime."""
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(PROJECT_ROOT)
    return pipeline


def _citation_sources(result: FinalResponse) -> list[dict]:
    return []


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the chat UI."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("Chua c� file static/index.html", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    """Serve the favicon."""
    return FileResponse(STATIC_DIR / "favicon.ico")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer one user query through the local RAG assistant."""
    query = request.message.strip()
    if not query:
        return ChatResponse(
            answer="M? vui l�ng nh?p c�u h?i nh�!",
            citations=[],
            sources=[],
            method="input_validation",
            latency={"total_seconds": 0.0},
        )

    started = time.perf_counter()
    try:
        result = _get_pipeline().answer_query(query)
    except Exception as exc:
        print(f"[web_app] Chat request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="The local assistant is temporarily unavailable. Please verify Qdrant, Ollama, and local model artifacts.",
        ) from exc
    elapsed = time.perf_counter() - started
    return ChatResponse(
        answer=result.answer,
        citations=[],
        sources=[],
        method="llm_generated",
        verification_status=None,
        insufficient_evidence=False,
        latency={"total_seconds": round(elapsed, 3)},
    )


@app.get("/api/health")
def health() -> dict[str, object]:
    """Return a lightweight service health response."""
    return {
        "status": "ok",
        "runtime_initialized": pipeline is not None,
        "pipeline": "local maternal-health RAG assistant",
        "embedding_model": "AITeamVN/Vietnamese_Embedding_v2",
        "embedding_device": "cpu",
        "embedding_resident": pipeline is not None and pipeline.dense is not None,
        "reranker_device": "cuda",
        "reranker_resident": pipeline is not None and pipeline.reranker is not None,
        "generation_model": "None",
        "generation_warmed": False,
        "score_gate_threshold": 0.3,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
