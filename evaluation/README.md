# Evaluation

Thư mục này chứa dataset, script tái lập đánh giá và artifact local của RAG pipeline. Không sửa dữ liệu ground truth hoặc nhãn thủ công chỉ để cải thiện metric.

## Cấu trúc

```text
evaluation/
├── datasets/
│   ├── eval_questions.json       # Benchmark retrieval gốc (DEV/TEST)
│   └── answerable_labels.json    # Nhãn thủ công cho score-gate/generation
├── scripts/
│   ├── indexing/                 # Chunking, embedding và build vector store
│   ├── retrieval/                # Dense/BM25/hybrid/reranker evaluation
│   └── generation/               # Score-gate calibration và generation evaluation
└── results/                      # Artifact local, Git-ignored
    ├── indexing/
    ├── retrieval/
    ├── calibration/
    └── generation/
```

## Datasets

- `datasets/eval_questions.json`: benchmark retrieval hiện có. Không sửa evidence/ground truth để cải thiện metric.
- `datasets/answerable_labels.json`: tập nhãn nội bộ cho calibration và generation. `answerable` được gán dựa trên evidence trong catalog hiện hành; `safety_violation` chỉ được điền sau review output thủ công.

## Chạy evaluation

Chạy từ project root trong PowerShell. Các output nên có tên experiment riêng để không ghi đè run cũ.

### Indexing

```powershell
python evaluation\scripts\indexing\run_chunking_eval.py `
  --cleaned-dir data\cleaned `
  --output-dir evaluation\results\indexing\chunks_cache
```

```powershell
python evaluation\scripts\indexing\run_embedding_eval.py `
  --split dev `
  --chunks-dir evaluation\results\indexing\vietnamese_embedding_v2\chunks_cache `
  --questions evaluation\datasets\eval_questions.json `
  --output-dir evaluation\results\indexing\embedding_eval_dev `
  --models AITeamVN/Vietnamese_Embedding_v2
```

### Retrieval

```powershell
python evaluation\scripts\retrieval\run_retrieval_eval.py `
  --split test `
  --qdrant-host localhost `
  --qdrant-port 6333 `
  --qdrant-collection chunks `
  --embedder-device cuda `
  --reranker-device cuda `
  --pipeline hybrid_rerank `
  --candidate-pool-top-k 50 `
  --output-dir evaluation\results\retrieval\reranker_pool50_test
```

### Calibrate score gate

```powershell
$env:PYTHONIOENCODING = "utf-8"
python evaluation\scripts\generation\calibrate_score_gate.py `
  --labels evaluation\datasets\answerable_labels.json `
  --output-dir evaluation\results\calibration\score_gate_v3 `
  --recall-priority
```

Chỉ cập nhật `GenerationConfig.score_gate_threshold` sau khi review `summary.json`, `threshold_metrics.csv`, per-question scores và false positives/false negatives. Threshold cần calibrate lại khi corpus, chunker, embedding model, reranker hoặc retrieval configuration thay đổi.

### Generation

Yêu cầu Qdrant và Ollama đang hoạt động. Output generation gồm answer, citation, retry metadata, numeric verification và latency.

```powershell
$env:PYTHONIOENCODING = "utf-8"
python evaluation\scripts\generation\eval_generation.py `
  --test-set evaluation\datasets\answerable_labels.json `
  --output-dir evaluation\results\generation\qwen35_2b_fewshot_on_v3 `
  --warm-up
```

## Diễn giải metric

- `citation_accuracy_rate`: chỉ đánh giá citation được LLM sinh ra có reference hợp lệ trong runtime context; không chứng minh semantic entailment.
- `faithfulness_pass_rate`: deterministic verifier chỉ kiểm chứng claim có số/đơn vị tại chunk được cite. `unverifiable` không đồng nghĩa answer sai.
- Khi raw verifier báo `mismatch`, cần so khớp claim với toàn Top-5 trước khi gọi là hallucination. Có thể là citation-indexing error: claim đúng nhưng model chọn sai số `[n]`.
- `safety_violation_count` chỉ có giá trị sau review thủ công output cho các safety cases. Không dùng giá trị mặc định/chưa review để kết luận hệ thống an toàn.

## Artifact local

`results/` và runtime audit JSONL là artifact tái tạo được, được Git-ignore. Không xóa kết quả benchmark/citation audit nếu chưa xác nhận đã lưu số liệu cần dùng cho báo cáo.
