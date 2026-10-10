"""Build auditable numbered context blocks from retrieved chunks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from retrieval.base import RetrievedChunk

CATALOG_PATH = Path(__file__).with_name("document_catalog.yaml")


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, dict[str, Any]]:
    """Load the approved source citation catalog once per process."""
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _citation_header(metadata: dict[str, Any], catalog: dict[str, Any]) -> str:
    """Format document identity without exposing incomplete source dates."""
    source = str(metadata.get("source", ""))
    record = catalog.get(source, {})
    number = record.get("document_number", "UNKNOWN")
    name = record.get("document_name", source)
    subtitle = record.get("document_subtitle")
    label = f"{name} — QĐ {number}"
    if not record.get("date_incomplete_in_source", True) and record.get("issue_date_normalized"):
        label += f" — {record['issue_date_normalized']}"
    if subtitle:
        label += f" — {subtitle}"
    return label


def build_reference_sources(chunks: list[RetrievedChunk]) -> dict[int, dict[str, str]]:
    """Build canonical display metadata indexed by request-local reference number."""
    catalog = _load_catalog()
    sources: dict[int, dict[str, str]] = {}
    for reference_number, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        sources[reference_number] = {
            "reference_number": str(reference_number),
            "chunk_id": chunk.chunk_id,
            "citation_header": _citation_header(metadata, catalog),
            "breadcrumb": str(metadata.get("breadcrumb") or metadata.get("section") or ""),
        }
    return sources


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[int, str]]:
    """Format retrieved chunks and return reference-number to chunk-ID map.

    Args:
        chunks: Retrieved chunks, normally the final Top-5 list.

    Returns:
        Tuple of context text and ``{reference_number: chunk_id}`` mapping.
    """
    catalog = _load_catalog()
    blocks: list[str] = []
    reference_map: dict[int, str] = {}
    for reference_number, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        reference_map[reference_number] = chunk.chunk_id
        document = _citation_header(metadata, catalog)
        breadcrumb = metadata.get("breadcrumb") or metadata.get("section") or "(không có breadcrumb)"
        blocks.append(
            f"[{reference_number}] Văn bản: {document}\n"
            f"Section/Breadcrumb: {breadcrumb}\n"
            f"Chunk ID nội bộ: {chunk.chunk_id}\n"
            f"Nội dung nguyên văn:\n{chunk.text}"
        )
    return "\n\n".join(blocks), reference_map
