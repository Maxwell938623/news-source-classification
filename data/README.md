# `data/`

The full data pipeline — raw URLs → scraped headlines → cleaned + variant-normalised dataset → stratified splits.

```
raw/         course URLs + our additional discovered URLs (scraping inputs)
scraped/     raw HTML-extracted headlines, one CSV per source + merged
processed/   cleaned dataset + 5 normalisation variants + 70/15/15 splits
```

## `raw/`

| File | What it is |
|---|---|
| `original_urls.csv` | Course-released starter URLs (3 815 articles, 2010 Fox / 1805 NBC) plus our backfill discoveries |
| `latest_collected_urls.csv` | Most recent URL discovery output from `src/collect_urls.py` |
| `collect_urls_state.json` | Cursor / resume state for the URL collector (so it doesn't re-discover URLs across runs) |

## `scraped/`

| File | Source | Rows |
|---|---|---|
| `raw_scraped_headlines.csv` | Fox scrape | per-row: `url`, `headline`, `source` |
| `raw_scraped_headlines_nbc.csv` | NBC scrape | same schema |
| `raw_scraped_headlines_merged.csv` | Concatenation of the above | scraping ground-truth |
| `helper_as_raw.csv` | The course-provided `helpers/url_with_headlines.csv` re-shaped to match our schema (used as a fallback) |

## `processed/`

| File | What it is |
|---|---|
| `clean_headlines.csv` | **The final 20 000-row curated dataset (10 513 Fox / 9 487 NBC)** after dedup, length filter, and source-leak stripping. Source for all model training |
| `headlines_minimal.csv` / `_lowercase.csv` / `_nopunct.csv` / `_nostop.csv` / `_lemma.csv` | Five preprocessing variants of `clean_headlines.csv`, used for normalisation ablations during model selection |
| `dataset_summary.txt` | Human-readable dataset stats (counts, length distribution) |

## `processed/splits/`

70/15/15 stratified split, seed = 42:

| File | Rows |
|---|---|
| `train.csv` | 13 999 |
| `val.csv` | 3 000 |
| `test.csv` | 3 001 |
| `split_metadata.json` | Splits provenance (seed, sizes, per-class counts) for reproducibility |

All large CSVs are tracked via Git LFS (see `.gitattributes`).
