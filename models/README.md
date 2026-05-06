# `models/`

Trained model weights produced by the `src/` training scripts. **All large binaries are tracked via Git LFS** — run `git lfs pull` to materialise them.

## Top-level joblib artifacts (classical models)

| File | Trained by | What it is |
|---|---|---|
| `baseline_pipeline.joblib` (+ `_metadata.json`) | `src/train_baseline.py` | Handout-spec baseline: TF-IDF top-100 + LogReg |
| `best_model.joblib` (+ `_metadata.json`) | `src/train_best_model.py` | Winner of the full 149-config sweep (selected on val macro-F1, refit on train+val) |
| `char_ngram_best.joblib` (+ `_metadata.json`) | `src/models/char_ngram.py` via `train_best_model.py` | Best char-ngram + LR config |
| `gbdt_best.joblib` (+ `_metadata.json`) | `src/gbdt.py` | Best gradient boosted tree config (TF-IDF + stylometry features) |
| `hybrid_best.joblib` (+ `_metadata.json`) | `src/tune_hybrid_best.py` | Best weighted blend (TF-IDF word + char + LR + SVC) |
| `fasttext_supervised_best.bin` | `src/models/fasttext_classifier.py` | FastText supervised classifier |

## Hugging Face transformer checkpoints

Each folder is a complete `transformers` save_pretrained directory (config + tokenizer + safetensors weights + Trainer checkpoints):

| Folder | Model |
|---|---|
| `distilbert_hf/` | DistilBERT-base uncased (initial run) |
| `distilbert_20k_cuda_hf/` | DistilBERT-base uncased on the 20k-curated split, EC2 CUDA training — **headline result (test acc 0.8264)** |
| `deberta_small_hf/` | DeBERTa-v3-small (initial run) |
| `deberta_small_20k_cuda_hf/` | DeBERTa-v3-small on the 20k-curated split, EC2 CUDA training |
| `modernbert_base/` | ModernBERT-base (Answer.AI) fine-tune |

## `embedding_cache/`

Cached sentence-embedding tensors (`.npy`) keyed by content hash, used by `src/train_sentence_embedding_fast.py` to avoid recomputing embeddings across re-runs. Safe to delete; will be regenerated on next training run.

See `reports/metrics_*.json` for the test-set metrics that correspond to each artifact.
