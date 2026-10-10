#!/usr/bin/env python
"""Check that the CURRENT project state is consistent with the frozen retrieval spec.

It compares five sources against the spec defined in this file:
  1. live configuration objects and hard-coded reranker settings
  2. built artifacts (Qdrant collection, BM25 index, chunk file) and their hashes
  3. result files (evaluation/reports and evaluation/results)
  4. README.md (numbers, stale strings) and the latency log
  5. requirements files and the git working tree

Usage (from the project root, venv active, Qdrant running):
    python evaluation/scripts/verify_freeze.py --chunks <path to hierarchical.json used for the build>

Exit code is 1 if any check FAILS. Old commits and old results are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
ROWS: list[tuple[str, str, str]] = []

# ----------------------------------------------------------------------------
# Frozen specification (single source of truth for this check)
# ----------------------------------------------------------------------------
EXPECTED_CONFIG = {
    "candidate_pool_top_k": 30,
    "dense_top_k": 30,
    "sparse_top_k": 30,
    "rrf_k": 60,
    "final_top_k": 5,
    "ood_threshold": 0.3,
    "reranker_model_name": "BAAI/bge-reranker-v2-m3",
    "reranker_device": "cuda",
    "reranker_batch_size": 16,
}
EXPECTED_EMBEDDER = {"model_name": "Qwen/Qwen3-Embedding-0.6B", "device": "cpu"}
EXPECTED_QDRANT = {"collection_name": "chunks", "vector_size": 1024, "distance": "cosine"}
EXPECTED_CHUNKING = {"max_tokens": 512, "tokenizer": "Qwen/Qwen3-Embedding-0.6B", "overlap": 100}
N_CHUNKS = 1182
BM25_TOKENIZER = "underthesea"

# hybrid_rerank, overall (in-domain) metrics
EXPECTED_METRICS = {
    "dev": {"recall@3": 0.9038, "recall@5": 0.9615, "recall@10": 0.9808, "mrr": 0.8103, "ndcg@5": 0.8465},
    "test": {"recall@3": 0.9720, "recall@5": 0.9930, "recall@10": 0.9930, "mrr": 0.8865, "ndcg@5": 0.9074},
}
TOL = 5e-4
README_MUST_CONTAIN = ["96.15%", "81.03%", "84.65%", "99.30%", "88.65%", "90.74%"]
README_MUST_NOT_CONTAIN = ["phu_san_chunks", "Qwen3-Embedding-4B", "91.13%", "92.14%"]
MANIFEST_KEYS = {
    "dense_top_k": 30, "sparse_top_k": 30, "candidate_pool_top_k": 30,
    "rrf_k": 60, "final_top_k": 5, "ood_threshold": 0.3, "reranker_batch_size": 16,
}


# ----------------------------------------------------------------------------
def record(status: str, name: str, detail: str = "") -> None:
    ROWS.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  ->  {detail}" if detail else ""))


def check(name: str, ok: bool, detail: str = "") -> None:
    record(PASS if ok else FAIL, name, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_values(obj, key):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(find_values(v, key))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_values(item, key))
    return found


def same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


# ----------------------------------------------------------------------------
def check_config(root: Path):
    sys.path.insert(0, str(root / "src"))
    try:
        from retrieval.config import default_retrieval_config

        cfg = default_retrieval_config(root)
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "load RetrievalConfig", repr(exc))
        return None

    for key, expected in EXPECTED_CONFIG.items():
        actual = getattr(cfg, key, None)
        check(f"config.{key}", same(actual, expected), f"actual={actual!r} expected={expected!r}")
    for key, expected in EXPECTED_EMBEDDER.items():
        actual = getattr(cfg.embedder, key, None)
        check(f"embedder.{key}", same(actual, expected), f"actual={actual!r} expected={expected!r}")
    for key, expected in EXPECTED_QDRANT.items():
        actual = getattr(cfg.qdrant, key, None)
        check(f"qdrant.{key}", same(actual, expected), f"actual={actual!r} expected={expected!r}")

    reranker_src = (root / "src" / "retrieval" / "reranker.py").read_text(encoding="utf-8")
    check("reranker max_length=1024 (hard-coded)", "max_length=1024" in reranker_src)
    check("reranker fp16 (hard-coded)", "float16" in reranker_src)

    try:
        from indexing.chunking.config import (
            DEFAULT_TOKENIZER_MODEL,
            SELECTED_MAX_CHUNK_TOKENS,
            HierarchicalChunkerConfig,
        )

        check("chunking.max_tokens", SELECTED_MAX_CHUNK_TOKENS == EXPECTED_CHUNKING["max_tokens"],
              f"actual={SELECTED_MAX_CHUNK_TOKENS}")
        check("chunking.tokenizer", DEFAULT_TOKENIZER_MODEL == EXPECTED_CHUNKING["tokenizer"],
              f"actual={DEFAULT_TOKENIZER_MODEL}")
        check("chunking.overlap", HierarchicalChunkerConfig().child_chunk_overlap == EXPECTED_CHUNKING["overlap"])
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "load chunking config", repr(exc))
    return cfg


def check_artifacts(root: Path, cfg, chunks_path: Path | None, manifest: dict | None, manifest_text: str):
    qdrant_ids, bm25_ids, chunk_ids = None, None, None

    if cfg is not None:
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
            name = cfg.qdrant.collection_name
            info = client.get_collection(name)
            vectors = info.config.params.vectors
            distance = str(getattr(vectors.distance, "value", vectors.distance)).lower()
            check("qdrant points_count", info.points_count == N_CHUNKS, f"actual={info.points_count}")
            check("qdrant vector size/distance", vectors.size == 1024 and distance == "cosine",
                  f"size={vectors.size} distance={distance}")
            qdrant_ids, offset = set(), None
            while True:
                points, offset = client.scroll(collection_name=name, limit=500, offset=offset,
                                               with_payload=["chunk_id"], with_vectors=False)
                qdrant_ids.update(str(p.payload.get("chunk_id")) for p in points)
                if offset is None:
                    break
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "query Qdrant", repr(exc))

        try:
            with cfg.bm25_path.open("rb") as handle:
                data = pickle.load(handle)
            bm25_ids = {str(x) for x in data["chunk_ids"]}
            check("bm25 chunk count", len(data["chunk_ids"]) == N_CHUNKS, f"actual={len(data['chunk_ids'])}")
            check("bm25 tokenizer", str(data["tokenizer"]) == BM25_TOKENIZER, f"actual={data['tokenizer']}")
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "load BM25 index", repr(exc))

    if chunks_path is None:
        record(SKIP, "chunk file checks", "pass --chunks to enable")
    else:
        try:
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else raw.get("chunks", raw)
            chunk_ids = {str(c.get("chunk_id") or c["metadata"]["chunk_id"]) for c in items}
            check("chunk file count", len(items) == N_CHUNKS, f"actual={len(items)}")
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "load chunk file", repr(exc))

    sets = {k: v for k, v in {"qdrant": qdrant_ids, "bm25": bm25_ids, "chunk_file": chunk_ids}.items() if v is not None}
    if len(sets) >= 2:
        reference = next(iter(sets.values()))
        check("chunk IDs identical across " + "/".join(sets), all(v == reference for v in sets.values()))

    # hashes recorded in manifest
    if manifest is None:
        record(SKIP, "manifest hashes", "manifest not found")
        return
    targets = {"bm25_index.pkl": cfg.bm25_path if cfg else root / "data/vector_db/bm25_index.pkl",
               "eval_questions.json": root / "evaluation/datasets/eval_questions.json"}
    if chunks_path is not None:
        targets["chunk file"] = chunks_path
    for label, path in targets.items():
        if not path.exists():
            record(FAIL, f"hash {label}", f"missing {path}")
            continue
        digest = sha256(path)
        check(f"hash {label} recorded in manifest", digest.lower() in manifest_text, digest[:16] + "...")


def check_manifest(manifest: dict | None, manifest_text: str):
    if manifest is None:
        return
    for key, expected in MANIFEST_KEYS.items():
        values = find_values(manifest, key)
        if not values:
            record(SKIP, f"manifest.{key}", "key not present")
        else:
            check(f"manifest.{key}", all(same(v, expected) for v in values), f"values={values} expected={expected}")
    for text in ("Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-reranker-v2-m3", str(N_CHUNKS)):
        check(f"manifest mentions {text}", text in manifest_text)
    check("manifest has no 4B / 2560 leftovers", "Qwen3-Embedding-4B" not in manifest_text and "2560" not in manifest_text)


def check_results(root: Path):
    reports = root / "evaluation" / "reports"
    results = root / "evaluation" / "results" / "retrieval" / "retrieval_eval"
    for split, expected in EXPECTED_METRICS.items():
        hashes = {}
        for label, base in (("reports", reports), ("results", results)):
            path = base / f"retrieval_summary_{split}.json"
            if not path.exists():
                record(WARN if label == "results" else FAIL, f"{label}/{path.name}", "file missing")
                continue
            hashes[label] = sha256(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            check(f"{label} {split}: pool/final/rrf",
                  data.get("candidate_pool_top_k") == 30 and data.get("final_top_k") == 5 and data.get("rrf_k") == 60,
                  f"pool={data.get('candidate_pool_top_k')} final={data.get('final_top_k')} rrf={data.get('rrf_k')}")
            check(f"{label} {split}: reranker", data.get("reranker_model") == "BAAI/bge-reranker-v2-m3")
            overall = data["hybrid_rerank"]["groups"]["overall"]
            for metric, value in expected.items():
                actual = overall[metric]
                check(f"{label} {split}: hybrid_rerank {metric}", abs(actual - value) <= TOL,
                      f"actual={actual:.4f} expected={value:.4f}")
        if len(hashes) == 2 and hashes["reports"] != hashes["results"]:
            record(WARN, f"{split}: reports copy differs from results file",
                   "metrics match the spec, but the files are not byte-identical (re-run or timing); re-copy the newest one")


def check_docs(root: Path):
    readme_path = root / "README.md"
    if not readme_path.exists():
        record(FAIL, "README.md", "missing")
        return
    readme = readme_path.read_text(encoding="utf-8")
    for text in README_MUST_CONTAIN:
        check(f"README contains {text}", text in readme)
    for text in README_MUST_NOT_CONTAIN:
        check(f"README does not contain {text}", text not in readme)

    log_path = root / "evaluation" / "reports" / "latency_production_log.txt"
    if not log_path.exists():
        record(FAIL, "latency log", f"missing {log_path}")
        return
    raw = log_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-16")
        record(WARN, "latency log encoding", "file is UTF-16; re-save as UTF-8")
    median = re.search(r"Median Latency:\s*([\d.]+)", text)
    p95 = re.search(r"p95 Latency:\s*([\d.]+)", text)
    if not (median and p95):
        record(FAIL, "latency log summary lines", "Median/p95 lines not found")
        return
    for label, value in (("median", median.group(1)), ("p95", p95.group(1))):
        shown = f"{float(value):.3f} s"
        check(f"README latency {label} = log ({shown})", shown in readme)


def check_environment(root: Path):
    for name in ("requirements.txt", "requirements.frozen.txt"):
        path = root / name
        if not path.exists():
            record(WARN, name, "missing")
            continue
        data = path.read_bytes()
        bad = [i + 1 for i, line in enumerate(data.splitlines()) if any(b > 127 for b in line)]
        check(f"{name} is ASCII-only (pip on Windows)", not bad, f"non-ASCII lines: {bad[:8]}" if bad else "")
        if name == "requirements.txt":
            check("requirements.txt has no ragas", b"ragas" not in data.lower())
    status = git(root, "status", "--porcelain")
    head = git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    check("git working tree clean", status.returncode == 0 and not status.stdout.strip(),
          f"HEAD={head}" if not status.stdout.strip() else status.stdout.strip().replace("\n", " | ")[:200])


# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="project root")
    parser.add_argument("--chunks", default=None, help="chunk file used for the build (hierarchical.json)")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    manifest_path = root / "data" / "vector_db" / "index_manifest.json"
    manifest, manifest_text = None, ""
    if manifest_path.exists():
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
    else:
        record(FAIL, "index_manifest.json", f"missing {manifest_path}")

    cfg = check_config(root)
    check_manifest(manifest, manifest_text)
    check_artifacts(root, cfg, Path(args.chunks).resolve() if args.chunks else None, manifest, manifest_text.lower())
    check_results(root)
    check_docs(root)
    check_environment(root)

    counts = {s: sum(1 for r in ROWS if r[0] == s) for s in (PASS, FAIL, WARN, SKIP)}
    print("\n" + "=" * 60)
    print("  ".join(f"{k}={v}" for k, v in counts.items()))
    if counts[FAIL]:
        print("NOT CONSISTENT. Failing checks:")
        for status, name, detail in ROWS:
            if status == FAIL:
                print(f"  - {name}: {detail}")
        return 1
    print("CONSISTENT: configuration, artifacts, results and README agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
