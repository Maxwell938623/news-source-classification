# `src/`

Full ML pipeline source: data → train → evaluate → predict.

## Pipeline scripts (data side)

| Script | Stage |
|---|---|
| `collect_urls.py` | Discover Fox / NBC article URLs (seed-and-expand from index pages, with cursor state) |
| `continuous_scrape.py` | Long-running scraper supervisor with checkpointing |
| `scrape.py` | One-shot headline scraper (5-tier extraction cascade with retry/backoff and source-artifact stripping) |
| `filter_csv_urls.py` | Filter a URL CSV to only well-formed article URLs |
| `preprocess.py` | Clean scraped CSV → `data/processed/clean_headlines.csv` (+ 5 normalisation variants); strip source-leaking patterns |
| `split.py` | Stratified 70/15/15 split (seed 42) → `data/processed/splits/` |
| `build_random_sample_dataset.py` | Subsample helper (used to construct the 20 k curated subset) |

## Training scripts (model side)

| Script | Trains |
|---|---|
| `train_baseline.py` | Handout-spec baseline (TF-IDF top-100 + LogReg) |
| `train_best_model.py` | Sweeps every classical configuration (149 total), picks winner on val macro-F1, refits on train+val, writes `models/best_model.joblib` and `reports/experiment_results.csv`. Also produces `reports/figures/experiment_comparison.png` (top-10 view) |
| `train_transformer.py` | Generic HuggingFace fine-tuner. Used for DistilBERT, DeBERTa-v3-small |
| `train_modernbert.py` | ModernBERT-base specific trainer (different config requirements) |
| `train_requested_models.py` | Re-train each explicit model family the assignment asks about, side-by-side |
| `train_sentence_embedding_fast.py` | Sentence-transformer + linear head (uses `models/embedding_cache/`) |
| `tune_hybrid_best.py` | Grid search over weighted TF-IDF word+char + LR+SVC blends |
| `gbdt.py` | Gradient boosted trees over TF-IDF + stylometric features |
| `run_all_model_families.py` | Driver that calls each family trainer in sequence |

## Evaluation / inference

| Script | Purpose |
|---|---|
| `evaluate.py` | Score any trained model on the held-out test split; produces `reports/metrics_*.json` and `reports/figures/confusion_matrix_*.png` |
| `eval_all_joblib_on_submission_input.py` | Score every joblib in `models/` against the helper-provided URL set (sanity-checks the submission contract end-to-end) |
| `predict.py` | Single-headline / batch CLI wrapper around any trained model |
| `build_model_folders.py` | Build the four `submission/<model>/` packages from the trained checkpoints in `models/` |

## `models/` subdirectory

Per-family `scikit-learn` model definitions imported by `train_best_model.py` (and the per-family training scripts above):

```
src/models/
├── _base.py                  shared utilities (text-column selection, vectoriser builders)
├── tfidf_logreg.py           tfidf_nb.py            tfidf_svm.py
├── char_ngram.py             stylometric.py         hybrid.py
├── voting_ensemble.py        stacking_ensemble.py
├── sentence_embedding.py
└── __init__.py
```

Each module exposes a `build()` factory returning an unfitted `sklearn.pipeline.Pipeline` so that `train_best_model.py` can sweep over them uniformly. Transformer fine-tunes (DistilBERT, DeBERTa-v3-small, ModernBERT-base) live in the top-level `train_transformer.py` / `train_modernbert.py` instead, since they don't fit the sklearn-pipeline interface. FastText is trained from `train_requested_models.py`.
