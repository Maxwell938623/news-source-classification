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
from urllib.parse import urlparse

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
DEFAULT_HELPER_DISTRIBUTION = PROJECT_ROOT / "helpers" / "url_with_headlines.csv"

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
# Topic/section alignment helpers
# ---------------------------------------------------------------------------

def _infer_source_from_url(url: str) -> str:
    host = urlparse(str(url)).netloc.lower()
    if "foxnews.com" in host:
        return "FoxNews"
    if "nbcnews.com" in host:
        return "NBC"
    return "Other"


def _extract_section_from_url(url: str) -> str:
    path = urlparse(str(url)).path.strip("/")
    if not path:
        return "_root"
    return path.split("/")[0].lower()


def _build_helper_section_distribution(helper_csv: Path) -> dict[str, dict[str, float]]:
    helper_df = pd.read_csv(helper_csv, dtype=str)
    if "url" not in helper_df.columns:
        raise ValueError(f"Helper file missing required column 'url': {helper_csv}")

    helper_df["source"] = helper_df["url"].apply(_infer_source_from_url)
    helper_df = helper_df[helper_df["source"].isin(["FoxNews", "NBC"])].copy()
    helper_df["section"] = helper_df["url"].apply(_extract_section_from_url)

    dist: dict[str, dict[str, float]] = {}
    for src, grp in helper_df.groupby("source"):
        counts = grp["section"].value_counts()
        total = counts.sum()
        dist[src] = {sec: float(c / total) for sec, c in counts.items()}
    return dist


def _build_helper_source_distribution(helper_csv: Path) -> dict[str, float]:
    helper_df = pd.read_csv(helper_csv, dtype=str)
    if "url" not in helper_df.columns:
        raise ValueError(f"Helper file missing required column 'url': {helper_csv}")
    helper_df["source"] = helper_df["url"].apply(_infer_source_from_url)
    helper_df = helper_df[helper_df["source"].isin(["FoxNews", "NBC"])].copy()
    counts = helper_df["source"].value_counts()
    total = counts.sum()
    return {src: float(c / total) for src, c in counts.items()}


def _trim_to_match_helper_sections(
    df: pd.DataFrame,
    helper_csv: Path,
    keep_fraction: float = 0.50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Trim rows to keep a target fraction while matching helper source/section ratios."""
    helper_dist = _build_helper_section_distribution(helper_csv)
    helper_source_props = _build_helper_source_distribution(helper_csv)

    work = df.copy()
    work = work[work["source"].isin(["FoxNews", "NBC"])].copy()
    work["section"] = work["url"].apply(_extract_section_from_url)

    trimmed_parts: list[pd.DataFrame] = []
    notes: list[str] = []
    per_source_state: dict[str, dict] = {}

    for src in ("FoxNews", "NBC"):
        sub = work[work["source"] == src].copy()
        if sub.empty or src not in helper_dist:
            continue

        avail = sub["section"].value_counts().to_dict()
        target_props = helper_dist[src]
        usable = {sec: p for sec, p in target_props.items() if avail.get(sec, 0) > 0}

        if not usable:
            per_source_state[src] = {"sub": sub, "usable": None, "max_total": len(sub)}
            continue

        usable_total = sum(usable.values())
        usable = {sec: p / usable_total for sec, p in usable.items()}
        per_source_state[src] = {
            "sub": sub,
            "avail": avail,
            "usable": usable,
            "max_total": len(sub),
            "missing_sections": [sec for sec in target_props if sec not in usable],
        }

    if not per_source_state:
        out = work.drop(columns=["section"])
        return out, "Section matching skipped (no usable overlap)."

    desired_total = int(round(len(work) * keep_fraction))
    desired_total = max(1, min(desired_total, len(work)))

    # Target per-source counts according to helper source ratio; then adjust to exact total.
    source_targets = {
        src: int(round(desired_total * helper_source_props.get(src, 0.0)))
        for src in per_source_state
    }
    for src in source_targets:
        source_targets[src] = min(source_targets[src], per_source_state[src]["max_total"])
    cur_total = sum(source_targets.values())
    if cur_total < desired_total:
        slack_sources = sorted(
            per_source_state.keys(),
            key=lambda s: per_source_state[s]["max_total"] - source_targets[s],
            reverse=True,
        )
        need = desired_total - cur_total
        for src in slack_sources:
            room = per_source_state[src]["max_total"] - source_targets[src]
            add = min(room, need)
            source_targets[src] += add
            need -= add
            if need <= 0:
                break
    elif cur_total > desired_total:
        reduce_sources = sorted(source_targets.keys(), key=lambda s: source_targets[s], reverse=True)
        extra = cur_total - desired_total
        for src in reduce_sources:
            dec = min(source_targets[src], extra)
            source_targets[src] -= dec
            extra -= dec
            if extra <= 0:
                break

    for src, state in per_source_state.items():
        sub = state["sub"]
        avail: dict[str, int] = state["avail"]
        usable = state.get("usable")
        if usable is None:
            trimmed_parts.append(sub)
            notes.append(f"{src}: no overlapping sections with helper; kept all {len(sub)} rows.")
            continue

        target_total = source_targets[src]
        target_total = max(1, min(target_total, state["max_total"], len(sub)))

        raw = {sec: target_total * usable[sec] for sec in usable}
        take = {sec: min(int(v), avail[sec]) for sec, v in raw.items()}
        remainder = target_total - sum(take.values())

        # Redistribute leftover to sections with spare capacity, weighted by helper probs.
        while remainder > 0:
            candidates = [sec for sec in usable if take[sec] < avail[sec]]
            if not candidates:
                break
            total_p = sum(usable[sec] for sec in candidates)
            if total_p <= 0:
                total_p = float(len(candidates))
                cand_p = {sec: 1.0 / total_p for sec in candidates}
            else:
                cand_p = {sec: usable[sec] / total_p for sec in candidates}

            add_raw = {sec: remainder * cand_p[sec] for sec in candidates}
            add = {
                sec: min(avail[sec] - take[sec], int(add_raw[sec]))
                for sec in candidates
            }
            gained = sum(add.values())
            for sec, n in add.items():
                take[sec] += n
            remainder -= gained

            if remainder <= 0:
                break

            # Allocate one-by-one by fractional part if floors were too coarse.
            by_frac = sorted(
                candidates,
                key=lambda s: (add_raw[s] - int(add_raw[s]), cand_p[s]),
                reverse=True,
            )
            moved = 0
            for sec in by_frac:
                if remainder <= 0:
                    break
                room = avail[sec] - take[sec]
                if room <= 0:
                    continue
                take[sec] += 1
                remainder -= 1
                moved += 1
            if moved == 0:
                break

        sampled = []
        for sec, n in take.items():
            if n <= 0:
                continue
            part = sub[sub["section"] == sec]
            sampled.append(part.sample(n=n, random_state=random_state))
        out_src = pd.concat(sampled, ignore_index=False) if sampled else sub.head(0)
        trimmed_parts.append(out_src)

        notes.append(
            f"{src}: {len(sub)} -> {len(out_src)} rows; "
            f"matched_sections={len(usable)}; missing_sections={len(state['missing_sections'])}."
        )

    if not trimmed_parts:
        out = work.drop(columns=["section"])
        return out, "Section matching skipped (no usable overlap)."

    out = (
        pd.concat(trimmed_parts, ignore_index=False)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    out = out.drop(columns=["section"])
    note = (
        f"Section-match trim applied (keep_fraction={keep_fraction:.2f}): "
        + " | ".join(notes)
    )
    return out, note


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def preprocess(
    input_path: Path,
    output_dir: Path,
    match_helper_distribution: bool = False,
    helper_distribution_csv: Path = DEFAULT_HELPER_DISTRIBUTION,
    helper_keep_fraction: float = 0.50,
) -> None:
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

    section_match_note: str | None = None
    if match_helper_distribution:
        if not helper_distribution_csv.exists():
            log.error("Helper distribution file not found: %s", helper_distribution_csv)
            sys.exit(1)
        if not (0.0 < helper_keep_fraction <= 1.0):
            log.error("helper_keep_fraction must be in (0, 1], got: %.4f", helper_keep_fraction)
            sys.exit(1)
        before_trim = len(df_success)
        df_success, section_match_note = _trim_to_match_helper_sections(
            df_success,
            helper_distribution_csv,
            keep_fraction=helper_keep_fraction,
            random_state=42,
        )
        log.info("Section-match trim: %d -> %d rows", before_trim, len(df_success))
        log.info("%s", section_match_note)

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
        section_match_note=section_match_note,
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
    section_match_note: str | None = None,
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
    ]
    if section_match_note:
        lines += [
            "HELPER DISTRIBUTION ALIGNMENT",
            f"  {section_match_note}",
            "",
        ]
    lines += ["=" * 64]
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
    p.add_argument(
        "--match-helper-distribution",
        action="store_true",
        help=(
            "Trim cleaned dataset so source x section(URL first path segment) "
            "distribution better matches helper/url_with_headlines.csv."
        ),
    )
    p.add_argument(
        "--helper-distribution-csv",
        metavar="PATH",
        type=Path,
        default=DEFAULT_HELPER_DISTRIBUTION,
        help="Helper CSV used as target section/topic distribution.",
    )
    p.add_argument(
        "--helper-keep-fraction",
        metavar="FLOAT",
        type=float,
        default=0.50,
        help=(
            "When matching helper distribution, keep approximately this fraction "
            "of cleaned rows (e.g., 0.50 keeps ~50%%)."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    preprocess(
        input_path=args.input,
        output_dir=args.output_dir,
        match_helper_distribution=args.match_helper_distribution,
        helper_distribution_csv=args.helper_distribution_csv,
        helper_keep_fraction=args.helper_keep_fraction,
    )
