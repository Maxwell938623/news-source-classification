# `submission/`

Leaderboard-ready packages conforming to the `helpers/Project_submission.pdf` contract.

## Active package (what the leaderboard sees)

The three files at the root of this folder — `model.py`, `preprocess.py`, `model.pt` — are the *currently active* leaderboard submission. They are a copy of one of the four sub-package folders below.

To swap which model is active before submitting, copy a sub-package's three files up:

```bash
# example: make ModernBERT the active submission
cp submission/modernbert_model/{model.py,preprocess.py,model.pt} submission/
```

## The four pre-built packages

| Folder | Underlying model | Why it exists |
|---|---|---|
| [`distilbert_model/`](distilbert_model/) | DistilBERT-base uncased (fine-tuned) | Highest test macro-F1 in our sweep (0.8264) |
| [`modernbert_model/`](modernbert_model/) | ModernBERT-base (Answer.AI, Dec 2024) | Modern SOTA-class encoder; loads HF config from the Hub at predict time |
| [`stack_4base_histgbm_model/`](stack_4base_histgbm_model/) | 4-base stacking ensemble + HistGBM meta-learner | Best classical model (0.7957). scikit-learn `Pipeline` is serialised inside a torch `state_dict` via a byte-buffer wrapper so the backend's `torch.load` + `load_state_dict` machinery transparently reconstructs the joblib pipeline |
| [`gbdt_model/`](gbdt_model/) | Gradient boosted trees over TF-IDF + stylometry | Compact reference baseline; same `state_dict` trick |

Each sub-folder is self-contained: `model.py` (defines `NewsClassifier` / `Model.predict`), `preprocess.py` (defines `prepare_data`), `model.pt` (the weights, tracked via Git LFS). All four were validated locally with `helpers/eval_project_b.py` before submission.

The classical packages embed the entire fitted scikit-learn `Pipeline` (TF-IDF vocab + classifier weights) inside `model.pt` as a `joblib`-serialised byte buffer wrapped in a `torch.nn.Module`, so the leaderboard backend's `torch.load` + `load_state_dict` pathway transparently rehydrates a sklearn pipeline. The transformer packages stream their tokenizer + config from Hugging Face Hub at predict time and only ship the fine-tuned weights in `model.pt`.

See `main.tex` §2.5 for the leaderboard submission record.
