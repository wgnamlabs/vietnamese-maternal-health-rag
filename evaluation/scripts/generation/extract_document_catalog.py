"""Extract document citation metadata verbatim from cleaned source files."""

from __future__ import annotations

import re
import sys
from datetime import date

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_PATH = PROJECT_ROOT / "src" / "generation" / "document_catalog.yaml"
DOCUMENT_NUMBER_RE = re.compile(r"Quyết định số\s+([^\s)]+)", re.IGNORECASE)
DATE_PATTERNS = (
    re.compile(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", re.IGNORECASE),
    re.compile(r"ngày\s+\d{1,2}/\d{1,2}/\d{4}", re.IGNORECASE),
    re.compile(r"ngày\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", re.IGNORECASE),
)


def first_lines(path: Path) -> list[str]:
    """Return the first non-empty lines without changing their content."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()[:15] if line.strip()]


def extract_record(path: Path) -> dict[str, Any]:
    """Extract one catalog record from the cleaned source header."""
    lines = first_lines(path)
    header_line = next(line for line in lines if line.startswith("# "))
    header = header_line[2:].strip()
    subtitle = next(
        (line.removeprefix("**Nội dung:**").strip() for line in lines if line.startswith("**Nội dung:**")),
        None,
    )
    if "(Ban hành" in header:
        header, embedded_issuance = header.split(" (Ban hành", 1)
        header = header.strip()
        embedded_issuance = "(Ban hành" + embedded_issuance
    else:
        embedded_issuance = None
    issue_line = next((line for line in lines if "Ban hành" in line and "Quyết định số" in line), embedded_issuance)
    if issue_line is None:
        raise ValueError(f"Không tìm thấy issuance line trong {path.name}")
    number_match = DOCUMENT_NUMBER_RE.search(issue_line)
    date_match = next((match for pattern in DATE_PATTERNS for match in [pattern.search(issue_line)] if match), None)
    if not number_match or not date_match:
        raise ValueError(f"Không trích xuất được metadata từ {path.name}: {issue_line}")
    issue_date_raw = date_match.group(0).rstrip("*). ")
    return {
        "document_number": number_match.group(1),
        "document_name": header,
        "document_subtitle": subtitle,
        "issue_date_raw": issue_date_raw,
        "issue_date_normalized": normalize_date(issue_date_raw),
        "date_incomplete_in_source": "ngày tháng" in issue_date_raw.lower(),
        "source_file": path.name,
        "issuance_line_verbatim": issue_line,
    }


def normalize_date(raw: str) -> str | None:
    """Return ISO date only when the source includes a day."""
    words = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", raw, re.IGNORECASE)
    if words:
        return date(int(words.group(3)), int(words.group(2)), int(words.group(1))).isoformat()
    slash = re.search(r"ngày\s+(\d{1,2})/(\d{1,2})/(\d{4})", raw, re.IGNORECASE)
    if slash:
        return date(int(slash.group(3)), int(slash.group(2)), int(slash.group(1))).isoformat()
    return None


def main() -> None:
    """Extract and write all seven records, then print them for review."""
    records = {path.name: extract_record(path) for path in sorted(CLEANED_DIR.glob("*.md"))}
    if len(records) != 7:
        raise ValueError(f"Expected 7 cleaned Markdown files, found {len(records)}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(records, allow_unicode=True, sort_keys=False, width=120)
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(content.encode("utf-8").decode("utf-8"))


if __name__ == "__main__":
    main()
