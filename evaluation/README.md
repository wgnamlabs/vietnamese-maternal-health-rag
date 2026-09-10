# Evaluation

This directory contains the benchmark datasets, evaluation scripts, schemas, results, and archived diagnostics used to assess the assistant.

```text
evaluation/
├── datasets/        Benchmark questions and reviewed labels
├── schemas/         Evaluation schema
├── scripts/         Indexing, retrieval, query, and generation evaluation
├── results/         Organized evaluation outputs
└── archive/         Historical diagnostic artifacts
```

The main evaluation dimensions are:

- indexing and embedding quality;
- dense, sparse, hybrid, and reranked retrieval;
- query safety routing;
- prompt construction and answer generation.

The benchmark follows a controlled development/test split. Evidence and answer-related fields are kept separate from model outputs so results remain auditable.
