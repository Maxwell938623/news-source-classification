# News Source Classification

Binary text classification project for predicting whether a headline came from:
- `0` = `FoxNews`
- `1` = `NBC`

This README focuses on exactly what you asked for:
1. gather data,
2. train models,
3. pick the best-performing model.

## Setup

```bash
pip install -r requirements.txt
```

Create the starter URL file at:
- `data/raw/original_urls.csv`

Expected URL file:
- must contain a URL column (`url`, `URL`, `link`, etc.)
- may optionally include a source column (`source`, `label`, etc.)
- if source is missing, source is inferred from domain (`foxnews.com`, `nbcnews.com`, `msnbc.com`)

## 1) Gather Data

### Step 0: Collect real Fox/NBC article links

```bash
python src/collect_urls.py --output data/raw/original_urls.csv --mode backfill --per-source 150
```

Large historical backfill (recommended for bigger training sets):
```bash
python src/collect_urls.py --output data/raw/original_urls.csv --mode backfill --per-source 5000 --max-sitemaps 4000
```

Restart backfill from newest again:
```bash
python src/collect_urls.py --output data/raw/original_urls.csv --mode backfill --reset-state
```

Maximum-volume pull from all discovered URLs in a run:
```bash
python src/collect_urls.py --output data/raw/original_urls.csv --mode backfill --per-source 0 --max-sitemaps 4000
```

Notes:
- collector now crawls sitemap indexes (including nested ones), then feeds, then section pages.
- in `backfill` mode it sorts by publication timestamp and advances a saved cursor each run.
- state file: `data/raw/collect_urls_state.json`
- this is the best practical way to get a lot of past + recent links.
- "every possible" link is not guaranteed because publishers can remove/omit old items from public sitemaps.

Main outputs:
- `data/raw/original_urls.csv`
- `logs/collect_urls.log`

### Step A: Scrape headlines from URLs

```bash
python src/scrape.py --urls data/raw/original_urls.csv
```

Useful options:
```bash
python src/scrape.py --urls data/raw/original_urls.csv --resume
python src/scrape.py --delay 1.5 --timeout 20 --max-retries 3
```

Main outputs:
- `data/scraped/raw_scraped_headlines.csv`
- `logs/scrape.log`

### Optional: Run continuously (poll forever)

```bash
python src/continuous_scrape.py --interval-minutes 20 --per-source 80 --collector-mode backfill
```

What it does each cycle:
- runs `collect_urls.py` to fetch fresh real links,
- `backfill` mode keeps moving to older publication dates via saved cursor state,
- merges only new URLs into `data/raw/original_urls.csv`,
- runs `scrape.py --resume` so already-scraped links are skipped,
- sleeps for the interval and repeats until you press `Ctrl+C`.

For deeper per-cycle backfill, increase sitemap depth:
```bash
python src/continuous_scrape.py --interval-minutes 20 --per-source 200 --collector-mode backfill --collector-max-sitemaps 2500
```

Use all discovered URLs each cycle (very heavy):
```bash
python src/continuous_scrape.py --interval-minutes 20 --per-source 0 --collector-mode backfill --collector-max-sitemaps 2500
```

Log file:
- `logs/continuous_scrape.log`

### Step B: Clean and preprocess

```bash
python src/preprocess.py --input data/scraped/raw_scraped_headlines.csv
```

Main outputs:
- `data/processed/clean_headlines.csv`
- `data/processed/headlines_minimal.csv`
- `data/processed/headlines_lowercase.csv`
- `data/processed/headlines_nopunct.csv`
- `data/processed/headlines_nostop.csv`
- `data/processed/headlines_lemma.csv`
- `data/processed/dataset_summary.txt`
- `logs/preprocess.log`

### Step C: Create reproducible train/val/test splits

```bash
python src/split.py --input data/processed/clean_headlines.csv --seed 42
```

Split ratio is stratified `70/15/15` (train/val/test).

Main outputs:
- `data/processed/splits/train.csv`
- `data/processed/splits/val.csv`
- `data/processed/splits/test.csv`
- `data/processed/splits/split_metadata.json`
- `logs/split.log`

## 2) Train the Models

### Train baseline (handout baseline)

```bash
python src/train_baseline.py
```

Outputs include:
- `models/baseline_pipeline.joblib`
- `models/baseline_pipeline_metadata.json`
- `reports/metrics_baseline.json`
- `reports/figures/baseline_confusion_matrix.png`
- `logs/train_baseline.log`

### Train all experiment families and best model

```bash
python src/train_best_model.py
```

Quick sanity run (fewer families):
```bash
python src/train_best_model.py --fast
```

Outputs include:
- `reports/experiment_results.csv`
- `models/best_model.joblib`
- `models/best_model_metadata.json`
- `reports/metrics_best.json`
- `reports/figures/experiment_comparison.png`
- `reports/figures/best_confusion_matrix.png`
- `logs/train_best_model.log`

## 3) Pick the Best-Performing Model

Best-model selection is already built into `train_best_model.py`:
- selection criterion: **highest validation `f1_macro`**
- after selection, winner is retrained on `train + val`
- final performance is reported once on held-out test

How to verify the winner:
1. Open `reports/experiment_results.csv` and sort by `val_f1_macro` descending (top row is selected model).
2. Confirm selected model metadata in `models/best_model_metadata.json`.
3. Compare test performance in:
   - `reports/metrics_baseline.json`
   - `reports/metrics_best.json`

Optional full evaluation for saved models:
```bash
python src/evaluate.py --model models/best_model.joblib
python src/evaluate.py --model models/baseline_pipeline.joblib --suffix baseline
```

This generates extra analysis (confusion matrix, misclassified samples, feature plots) under `reports/` and `reports/figures/`.



python src/continuous_scrape.py --interval-minutes 20 --per-source 0 --collector-mode backfill --collector-max-sitemaps-foxnews 25 --collector-max-sitemaps-nbc 400