"""Fine-tuning the cross-encoder reranker on the RFC domain (branch feat/reranker-finetune).

    data.py      training data: generation windows, boilerplate filtering, query cleaning, the
                 ground-truth leakage filter, hard-negative selection, the document-level split

The pipeline (scripts/reranker_data.py, scripts/generate_queries.py) and its write-up
(docs/reranker-finetune.md) describe how the pieces fit together.
"""
