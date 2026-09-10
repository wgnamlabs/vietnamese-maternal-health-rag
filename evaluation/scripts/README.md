# Evaluation scripts

The scripts below are organized by responsibility. They support reproducible evaluation of the assistant's knowledge, retrieval, safety routing, and answer generation behavior.

```text
indexing/
  build_vector_store.py
  run_chunking_eval.py
  run_embedding_eval.py

retrieval/
  metrics.py
  run_retrieval_eval.py

query/
  apply_scope_rubric.py
  build_query_classification_labels.py
  run_query_classification_eval.py

generation/
  extract_document_catalog.py
  run_generation_eval.py
  smoke_test_generation_single.py
  smoke_test_prompt.py
```

Scripts resolve the project root from their nested location and use datasets/results under the organized `evaluation/` layout.
