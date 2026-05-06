# `reports/`

All metrics, figures, and per-model breakdown artifacts referenced by `main.tex`.

## Top-level files

| File | What it contains |
|---|---|
| `experiment_results.csv` | Full sweep over all classical configurations (149 rows × {features, model, train/val/test acc, macro-F1, weighted-F1}). Source for Figure 1 in the report |
| `hybrid_tuning_results.csv` | Grid-search results from `src/tune_hybrid_best.py` (TF-IDF + char-ngram + LR + SVC weighted blends) |
| `model_family_run_summary.json` | One-line summary per model family from `src/run_all_model_families.py` |
| `requested_model_comparison.csv` / `.json` | Side-by-side comparison of the explicit model families the assignment requested |
| `misclassified_sample.csv` | Sampled errors used for qualitative error analysis in `main.tex` §2.4 |

## Per-run metrics (one JSON per training run)

| Pattern | Origin |
|---|---|
| `metrics_baseline.json` | Handout-spec baseline (TF-IDF top-100 + LogReg) |
| `metrics_best.json` | Best classical model selected by `src/train_best_model.py` |
| `metrics_distilbert*.json` | DistilBERT-base fine-tunes (`*_20k_cuda` = the 20k-curated EC2 run, the headline result) |
| `metrics_deberta_small*.json` | DeBERTa-v3-small fine-tunes |
| `metrics_modernbert_base.json` | ModernBERT-base fine-tune |
| `metrics_eval_baseline.json` / `metrics_eval_best.json` | Re-evaluation outputs from `src/evaluate.py` (sanity-checking the joblib artifacts) |
| `joblib_eval_url_with_headlines.csv` / `_summary.json` | Output of `src/eval_all_joblib_on_submission_input.py` running every joblib in `models/` against the helper-provided URL set |

## Subdirectories

| Folder | Content |
|---|---|
| `figures/` | All PNG figures embedded in `main.tex` (confusion matrices per model, baseline-vs-best bar chart, top-feature plot, length-vs-error plot, top-10 experiment comparison) |
| `model_breakdown/<model_name>/` | Per-model {classification report, confusion matrix, ROC curve, metrics JSON} produced as a side-effect of training |
