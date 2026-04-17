#!/usr/bin/env python3
"""
preprocess.py — Clean and preprocess raw scraped headlines into analysis-ready datasets.

Label encoding (consistent across all project scripts):
    0  =  FoxNews
    1  =  NBC

Outputs (all written to data/processed/):
    clean_headlines.csv          master file with all variant columns
    headlines_minimal.csv        strip + normalize + deduplicated
    headlines_lowercase.csv      minimal + lowercase
    headlines_nopunct.csv        lowercase + punctuation removed
    headlines_nostop.csv         lowercase + stopwords removed        (NLTK)
    headlines_lemma.csv          lowercase + stopwords + lemmatized   (NLTK)
    dataset_summary.txt          human-readable quality report

Usage:
    python src/preprocess.py
    python src/preprocess.py --input data/scraped/raw_scraped_headlines.csv
    python src/preprocess.py --input data/scraped/raw_scraped_headlines.csv \\
                              --output-dir data/processed
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Optional NLTK — graceful degradation when unavailable
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    def _ensure_nltk_data() -> None:
        needed = {
            "corpora/stopwords": "stopwords",
            "corpora/wordnet": "wordnet",
            "corpora/omw-1.4": "omw-1.4",
        }
        for path, pkg in needed.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

    _ensure_nltk_data()
    NLTK_AVAILABLE = True
except (ImportError, Exception):  # noqa: BLE001 — also catches numpy ABI errors
    NLTK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRAPED_DIR = PROJECT_ROOT / "data" / "scraped"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_INPUT = SCRAPED_DIR / "raw_scraped_headlines.csv"

# ---------------------------------------------------------------------------
# Text-cleaning primitives
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")          # removes everything that isn't word-char or space


def _unescape(text: str) -> str:
    return html.unescape(text)


def _strip_html_tags(text: str) -> str:
    return _HTML_TAG.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    return _MULTI_WS.sub(" ", text).strip()


def clean_minimal(text: str) -> str:
    """
    Unescape HTML entities, strip any residual HTML tags, normalize whitespace.
    This is the base cleaning step applied to every headline.
    """
    text = _unescape(text)
    text = _strip_html_tags(text)
    text = _normalize_whitespace(text)
    return text


def clean_lowercase(text: str) -> str:
    return clean_minimal(text).lower()


def clean_nopunct(text: str) -> str:
    text = clean_lowercase(text)
    text = _PUNCT.sub(" ", text)
    return _normalize_whitespace(text)


def clean_nostop(text: str, stop_words: set[str]) -> str:
    text = clean_nopunct(text)
    tokens = [tok for tok in text.split() if tok not in stop_words]
    return " ".join(tokens)


def clean_lemma(text: str, stop_words: set[str], lemmatizer: "WordNetLemmatizer") -> str:
    text = clean_nopunct(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in text.split() if tok not in stop_words]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def preprocess(input_path: Path, output_dir: Path) -> None:
    # ---- Setup ----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging(LOGS_DIR / "preprocess.log")
    log = logging.getLogger(__name__)

    # ---- Load raw data --------------------------------------------------
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    log.info("Loading raw scraped data from: %s", input_path)
    df = pd.read_csv(input_path, dtype=str)
    n_raw = len(df)
    log.info("Total rows loaded: %d", n_raw)

    # Validate required columns
    required = {"url", "source", "label", "raw_headline", "scrape_status"}
    missing = required - set(df.columns)
    if missing:
        log.error("Input CSV is missing required columns: %s", missing)
        sys.exit(1)

    # ---- Keep only successful scrapes -----------------------------------
    df_success = df[df["scrape_status"] == "success"].copy()
    n_failed = n_raw - len(df_success)
    log.info(
        "Successful scrapes: %d  |  Failed/skipped: %d",
        len(df_success), n_failed,
    )

    # Failure breakdown (for summary report)
    fail_breakdown = (
        df[df["scrape_status"] != "success"]["scrape_status"]
        .value_counts()
        .to_dict()
    )

    # ---- Drop null / empty raw headlines --------------------------------
    before = len(df_success)
    df_success = df_success[
        df_success["raw_headline"].notna()
        & (df_success["raw_headline"].str.strip() != "")
    ].copy()
    n_null_drop = before - len(df_success)
    log.info("Dropped %d rows with null/empty raw headline.", n_null_drop)

    # ---- Apply minimal cleaning -----------------------------------------
    df_success["headline_minimal"] = df_success["raw_headline"].apply(clean_minimal)

    # Drop rows where minimal cleaning yields an empty string
    before = len(df_success)
    df_success = df_success[df_success["headline_minimal"].str.strip() != ""].copy()
    n_empty_after_clean = before - len(df_success)
    if n_empty_after_clean:
        log.info("Dropped %d rows empty after minimal cleaning.", n_empty_after_clean)

    # ---- Deduplicate: same headline + same source -----------------------
    before = len(df_success)
    df_success = df_success.drop_duplicates(subset=["headline_minimal", "source"])
    n_dup_same = before - len(df_success)
    log.info("Removed %d within-source duplicates.", n_dup_same)

    # ---- Deduplicate: headline appears in BOTH sources (label leakage) --
    cross_mask = df_success.duplicated(subset=["headline_minimal"], keep=False)
    n_dup_cross = cross_mask.sum()
    if n_dup_cross:
        log.warning(
            "%d headlines appear in both sources (potential label leakage) — removing.",
            n_dup_cross,
        )
        df_success = df_success[~cross_mask].copy()

    # ---- Coerce label to integer ----------------------------------------
    df_success["label"] = pd.to_numeric(df_success["label"], errors="coerce").astype("Int64")

    # ---- Sanity-check label values are 0 or 1 ---------------------------
    valid_labels = {0, 1}
    bad_labels = set(df_success["label"].dropna().unique()) - valid_labels
    if bad_labels:
        log.warning("Unexpected label values found: %s — these rows will be dropped.", bad_labels)
        df_success = df_success[df_success["label"].isin([0, 1])].copy()

    log.info("Final clean dataset size: %d rows", len(df_success))

    # ---- Additional preprocessing variants ------------------------------
    df_success["headline_lowercase"] = df_success["headline_minimal"].str.lower()
    df_success["headline_nopunct"] = df_success["headline_minimal"].apply(clean_nopunct)

    if NLTK_AVAILABLE:
        eng_stop = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        df_success["headline_nostop"] = df_success["headline_minimal"].apply(
            lambda t: clean_nostop(t, eng_stop)
        )
        df_success["headline_lemma"] = df_success["headline_minimal"].apply(
            lambda t: clean_lemma(t, eng_stop, lemmatizer)
        )
        log.info("Applied stopword removal and lemmatization (NLTK).")
    else:
        # Degrade gracefully: nostop/lemma fall back to lowercase
        df_success["headline_nostop"] = df_success["headline_lowercase"]
        df_success["headline_lemma"] = df_success["headline_lowercase"]
        log.warning(
            "NLTK not available — headline_nostop and headline_lemma fall back to lowercase."
        )

    # ---- Compute statistics for summary ---------------------------------
    source_counts = df_success["source"].value_counts().to_dict()
    label_counts = df_success["label"].value_counts().to_dict()
    lengths = df_success["headline_minimal"].str.split().str.len()
    avg_len = lengths.mean()
    min_len = int(lengths.min())
    max_len = int(lengths.max())
    median_len = float(lengths.median())

    # Short-headline check (< 3 words — possibly navigation text)
    very_short = df_success[lengths < 3]
    if len(very_short):
        log.warning(
            "%d headlines are fewer than 3 words — inspect for navigation/stub text.",
            len(very_short),
        )

    # ---- Save master clean file -----------------------------------------
    master_cols = [
        "url", "source", "label",
        "raw_headline",
        "headline_minimal",
        "headline_lowercase",
        "headline_nopunct",
        "headline_nostop",
        "headline_lemma",
    ]
    clean_df = df_success[[c for c in master_cols if c in df_success.columns]].reset_index(drop=True)
    master_path = output_dir / "clean_headlines.csv"
    clean_df.to_csv(master_path, index=False)
    log.info("Saved master clean dataset: %s", master_path)

    # ---- Save individual variant files ----------------------------------
    variant_cols = {
        "minimal":   "headline_minimal",
        "lowercase": "headline_lowercase",
        "nopunct":   "headline_nopunct",
        "nostop":    "headline_nostop",
        "lemma":     "headline_lemma",
    }
    base_cols = ["url", "source", "label"]
    for variant, col in variant_cols.items():
        var_df = clean_df[base_cols + [col]].rename(columns={col: "headline"})
        var_path = output_dir / f"headlines_{variant}.csv"
        var_df.to_csv(var_path, index=False)
    log.info("Saved %d variant CSV files to %s", len(variant_cols), output_dir)

    # ---- Build and save dataset summary ---------------------------------
    summary = _build_summary(
        input_path=input_path,
        output_dir=output_dir,
        n_raw=n_raw,
        n_failed=n_failed,
        fail_breakdown=fail_breakdown,
        n_null_drop=n_null_drop,
        n_empty_after_clean=n_empty_after_clean,
        n_dup_same=n_dup_same,
        n_dup_cross=n_dup_cross,
        source_counts=source_counts,
        label_counts=label_counts,
        avg_len=avg_len,
        min_len=min_len,
        max_len=max_len,
        median_len=median_len,
        n_final=len(clean_df),
        nltk_used=NLTK_AVAILABLE,
    )
    summary_path = output_dir / "dataset_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    log.info("Saved dataset summary: %s", summary_path)

    print("\n" + summary)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    input_path: Path,
    output_dir: Path,
    n_raw: int,
    n_failed: int,
    fail_breakdown: dict,
    n_null_drop: int,
    n_empty_after_clean: int,
    n_dup_same: int,
    n_dup_cross: int,
    source_counts: dict,
    label_counts: dict,
    avg_len: float,
    min_len: int,
    max_len: int,
    median_len: float,
    n_final: int,
    nltk_used: bool,
) -> str:
    lines: list[str] = [
        "=" * 64,
        "DATASET SUMMARY — Fox News / NBC News Classification",
        "=" * 64,
        "",
        f"  Input file:                 {input_path}",
        f"  Output directory:           {output_dir}",
        "",
        "LABEL ENCODING",
        "  0  =  FoxNews",
        "  1  =  NBC",
        "",
        "RAW COUNTS",
        f"  Total URLs in input:        {n_raw}",
        f"  Successful scrapes:         {n_raw - n_failed}",
        f"  Failed / skipped:           {n_failed}",
    ]

    if fail_breakdown:
        lines.append("  Failure breakdown:")
        for reason, cnt in sorted(fail_breakdown.items()):
            lines.append(f"    {reason:<30} {cnt}")

    lines += [
        "",
        "CLEANING STEPS (rows dropped)",
        f"  Null / empty raw headline:  {n_null_drop}",
        f"  Empty after minimal clean:  {n_empty_after_clean}",
        f"  Within-source duplicates:   {n_dup_same}",
        f"  Cross-source duplicates:    {n_dup_cross}",
        "",
        "FINAL DATASET",
        f"  Total headlines:            {n_final}",
        "",
        "  Class distribution:",
    ]

    for src, cnt in sorted(source_counts.items()):
        lbl = 0 if src == "FoxNews" else 1
        pct = 100 * cnt / max(n_final, 1)
        lines.append(f"    {src:<12} (label={lbl})  {cnt:>5}  ({pct:.1f}%)")

    lines += [
        "",
        "HEADLINE LENGTH STATISTICS (word count, minimal variant)",
        f"  Average:    {avg_len:.1f} words",
        f"  Median:     {median_len:.1f} words",
        f"  Minimum:    {min_len} words",
        f"  Maximum:    {max_len} words",
        "",
        "PREPROCESSING VARIANTS SAVED",
        "  headlines_minimal.csv    — strip + normalize + dedup",
        "  headlines_lowercase.csv  — minimal + lowercase",
        "  headlines_nopunct.csv    — lowercase + punctuation removed",
        f"  headlines_nostop.csv     — lowercase + stopwords removed  "
        f"({'NLTK' if nltk_used else 'fallback=lowercase'})",
        f"  headlines_lemma.csv      — lowercase + lemmatized         "
        f"({'NLTK' if nltk_used else 'fallback=lowercase'})",
        "",
        "=" * 64,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean and preprocess raw scraped headlines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        metavar="PATH",
        type=Path,
        default=DEFAULT_INPUT,
        help="Raw scraped headlines CSV.",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory where processed files will be written.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    preprocess(input_path=args.input, output_dir=args.output_dir)
