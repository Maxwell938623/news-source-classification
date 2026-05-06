# News Source Classification — Final Project

**Course:** CIS 4190 / 5190 Applied Machine Learning · Spring 2026 · Project B (News Source)
**Team:** Maxwell Zhang · Arjun Verma · Isaac Dcruz
**Task:** Binary classification of news headlines as **FoxNews (label `0`)** or **NBC (label `1`)**.

| Resource | Link |
|---|---|
| Project report (5-page PDF, source: `main.tex`) | compile `main.tex` → `main.pdf` |
| Hugging Face dataset | https://huggingface.co/datasets/Maxwell938/CIS5190Project/tree/main |
| Leaderboard submission packages | [`submission/`](submission/) (4 swappable packages, see below) |
| Course handout | [`helpers/CIS 5190 Final Project Descriptions.pdf`](helpers/) |
| Course submission contract | [`helpers/Project_submission.pdf`](helpers/) |

---

## Headline Results

Held-out test on the 20 000-headline curated subset (3 001-example test fold, stratified seed 42; full numbers and protocol in §2.3 of the report):

| Rank | Model | Test Accuracy | Macro-F1 |
|---|---|---|---|
| 1 | DistilBERT-base (fine-tuned) | **0.8264** | **0.8264** |
| 2 | DeBERTa-v3-small (fine-tuned) | 0.8247 | 0.8244 |
| 3 | ModernBERT-base (fine-tuned) | 0.8187 | 0.8181 |
| 4 | 4-base stacking + HistGBM meta (best classical) | 0.7957 | 0.7955 |
| — | Handout baseline (TF-IDF 100 + LR) | 0.6248 | 0.6162 |

Improvement over baseline: **+20.2 accuracy / +21.0 macro-F1** points (DistilBERT vs handout baseline on this split). Per-class F1 and confusion matrices in `reports/metrics_*.json` and `reports/figures/`.

---

## For Graders — Assignment Requirement Index

Every basic + exploratory requirement from §1.4 of the handout, mapped to a file path:

| Requirement | Where it is |
|---|---|
| Data collection procedure & cleaning | `main.tex` §2.1 · pipeline scripts: `src/collect_urls.py`, `src/scrape.py`, `src/preprocess.py`, `src/split.py` · stats: `data/processed/dataset_summary.txt` and `data/processed/splits/split_metadata.json` |
| Final dataset | `data/processed/clean_headlines.csv` (20 000 rows; 10 513 Fox / 9 487 NBC); also published on Hugging Face (link above) |
| Model design + iterative process | `main.tex` §2.2 · per-family training scripts under `src/models/` · transformer training in `src/train_transformer.py` |
| Evaluation protocol + metrics + model selection | `main.tex` §2.3 · selection logic in `src/train_best_model.py` · per-model artifacts in `reports/model_breakdown/<model>/` |
| Required model-comparison line chart (incl. baseline) | `reports/figures/experiment_comparison.png`, embedded as Figure 1 left of `main.tex` |
| Error analysis | `main.tex` §2.4 (per-class numerical breakdown) · raw confusion matrices in `reports/metrics_*.json` |
| Exploratory components (≥1; we have 5) | `main.tex` §3 — Code/engineering, Techniques & literature grounding, Dataset expansion, Analysis, SOTA Comparison |
| Leaderboard submission record | `main.tex` §2.5 · packaged code in `submission/` |
| Best-performing model artifact | `submission/model.pt` (LFS) — currently the active classical-stack package; swappable to any of the four sub-packages described below |
| Team contribution statements | `main.tex` §4 |

---

## Submission Package (`submission/`)

We pre-built **four** leaderboard-ready packages, all conforming to the `submission.txt` contract (`prepare_data` returns `(X, y)`; `NewsClassifier`/`Model`/`get_model` exposes `predict`; `model.pt` is loaded as a `state_dict`). Active-package selection happens before final leaderboard submission by copying one sub-folder's `model.py`/`preprocess.py`/`model.pt` to `submission/` root.

| Package | Underlying model | Notes |
|---|---|---|
| [`submission/distilbert_model/`](submission/distilbert_model/) | DistilBERT-base uncased (fine-tuned) | Highest test macro-F1 in our sweep |
| [`submission/modernbert_model/`](submission/modernbert_model/) | ModernBERT-base (Answer.AI, Dec 2024) | Modern SOTA-class encoder |
| [`submission/stack_4base_histgbm_model/`](submission/stack_4base_histgbm_model/) | 4-base stacking ensemble with HistGBM meta | scikit-learn `Pipeline` serialised inside a torch `state_dict` via byte-buffer wrapper so leaderboard's `torch.load`+`load_state_dict` machinery transparently reconstructs the joblib pipeline |
| [`submission/gbdt_model/`](submission/gbdt_model/) | Gradient boosted trees over TF-IDF + stylometry | Compact reference baseline |

The currently-active package at `submission/model.py` + `submission/model.pt` is the classical stacking ensemble (the swap-active default). All four `model.pt` files are tracked via Git LFS.

---

## Quick Reproduction

Required packages:

```bash
pip install -r requirements.txt
```

The full pipeline from raw URLs to a trained best model:

```bash
python src/collect_urls.py --output data/raw/original_urls.csv --mode backfill --per-source 5000
python src/scrape.py        --urls   data/raw/original_urls.csv
python src/preprocess.py    --input  data/scraped/raw_scraped_headlines.csv
python src/split.py         --input  data/processed/clean_headlines.csv --seed 42
python src/train_baseline.py
python src/train_best_model.py
python src/train_transformer.py --model distilbert-base-uncased
```

Local sanity-check of any submission package against the leaderboard contract:

```bash
python helpers/eval_project_b.py --submission_dir submission/
```

Pull the actual model weights (Git LFS, several GB total):

```bash
git lfs pull
```

---

## Repository Layout

Each top-level directory has its own `README.md` describing what's inside.

```
.
├── main.tex                       5-page project report (compile to main.pdf)
├── README.md                      this file (grader-facing entry point)
├── requirements.txt               Python dependencies
├── url_only_data.csv              course-provided URL CSV (input format reference)
│
├── helpers/        ── README ──   course-provided handouts, templates, eval script
├── src/            ── README ──   full ML pipeline source (data → train → eval → predict)
│   └── models/                    per-family sklearn model definitions
├── data/           ── README ──   raw URLs, scraped headlines, cleaned + split dataset
│   ├── raw/                       discovered URLs + collector state
│   ├── scraped/                   raw scraped headlines (Fox + NBC)
│   └── processed/                 cleaned + 5 normalisation variants + 70/15/15 splits
├── models/         ── README ──   trained model weights (LFS) + per-model metadata JSON
├── reports/        ── README ──   metrics JSON, full results CSV, figures, per-model breakdown folders
├── submission/     ── README ──   4 leaderboard-ready packages + currently-active model.py / preprocess.py / model.pt
└── logs/           ── README ──   per-run pipeline + training logs (gitignored, regenerable)
```

> Click into any folder above to read the orientation README before diving into individual files.

---

## Detailed Pipeline Reference

For per-step CLI options and intermediate file paths, see the script docstrings under `src/`. The headline scraper uses a five-tier extraction cascade (source-specific `<h1>` heuristics → `itemprop=headline` → first generic `<h1>` → `og:title` → `<title>`) with retry/backoff to handle transient failures gracefully. Preprocessing produces five normalisation variants (`headline_minimal`, `headline_lowercase`, `headline_nopunct`, `headline_nostop`, `headline_lemma`) so that ablations across normalisation strategies are fair.

Selection criterion across all classical configurations: validation macro-F1. Winner is refit on `train + val` and scored exactly once on held-out test to avoid implicit test leakage. Detailed per-model breakdowns (metrics JSON, classification report, confusion matrix, ROC curve) are generated under `reports/model_breakdown/<model_name>/`.

---

## Team Contributions

See `main.tex` §4 for the official statements. In brief: **Maxwell Zhang** — data acquisition, model experiments and evaluation. **Arjun Verma** — data cleaning, model experiments and evaluation. **Isaac Dcruz** — model experiments and evaluation.

---

## Notes for Reviewers

* Label encoding is `FoxNews=0, NBC=1` throughout. The handout's §3.3 baseline snippet contains an internal contradiction (the comment says `0 for FoxNews, 1 for NBC` but the lambda code uses the opposite mapping); we follow the comment. The leaderboard backend documents a robust 2-class remapping in `submission.txt` §4.2, so the reverse encoding would still be scored correctly. See `main.tex` §2.2 for the full discussion.
* All randomness is seeded (`42`). Splits are reproducible from `data/processed/splits/split_metadata.json`.
* Large model files (`*.pt`, `*.joblib`, `*.bin`, `*.safetensors`, `*.kv`, large `data/**/*.csv`) are tracked via Git LFS — see `.gitattributes`.
