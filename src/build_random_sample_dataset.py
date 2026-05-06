#!/usr/bin/env python3
"""
Build a 20,000-row headline dataset from the helper CSV plus a random sample
of the larger scraped corpus.

The helper CSV is kept as the anchor set. Additional rows are sampled from
the larger scrape after removing helper duplicates, with source-level targets
chosen to make the final dataset approximately balanced between FoxNews and
NBC. The output schema matches data/processed/clean_headlines.csv so it can be
fed directly into src/split.py and the training scripts.

Usage:
    python src/build_random_sample_dataset.py
    python src/build_random_sample_dataset.py --target 20000 --seed 42
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HELPER_CSV = PROJECT_ROOT / "helpers" / "url_with_headlines.csv"
SCRAPED_CSV = PROJECT_ROOT / "data" / "scraped" / "raw_scraped_headlines_merged.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "clean_headlines.csv"

LABELS = {"FoxNews": 0, "NBC": 1}
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def clean_minimal(text: str) -> str:
    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    return _MULTI_WS.sub(" ", text).strip()


def clean_nopunct(text: str) -> str:
    text = clean_minimal(text).lower()
    text = _PUNCT.sub(" ", text)
    return _MULTI_WS.sub(" ", text).strip()


def infer_source(url: str) -> str:
    url_lower = str(url).lower()
    if "foxnews.com" in url_lower:
        return "FoxNews"
    if "nbcnews.com" in url_lower:
        return "NBC"
    return "Other"


def add_text_variants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["headline_minimal"] = out["raw_headline"].map(clean_minimal)
    out["headline_lowercase"] = out["headline_minimal"].str.lower()
    out["headline_nopunct"] = out["headline_minimal"].map(clean_nopunct)

    tokens = out["headline_nopunct"].str.split()
    # Keep this script dependency-light; train-time code can still use the
    # richer NLTK variants produced by src/preprocess.py when needed.
    out["headline_nostop"] = tokens.str.join(" ")
    out["headline_lemma"] = out["headline_nostop"]
    return out


def normalized_key(series: pd.Series) -> pd.Series:
    return series.fillna("").map(clean_minimal).str.lower()


def load_helper(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {"url", "headline"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Helper CSV missing required columns: {sorted(missing)}")

    df = df.rename(columns={"headline": "raw_headline"})
    df["source"] = df["url"].map(infer_source)
    df = df[df["source"].isin(LABELS)].copy()
    df["label"] = df["source"].map(LABELS)
    df = add_text_variants(df)
    df = df[df["headline_minimal"] != ""].copy()
    return df.drop_duplicates(subset=["url", "headline_minimal", "source"])


def load_scraped(path: Path, helper: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {"url", "source", "label", "raw_headline", "scrape_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scraped CSV missing required columns: {sorted(missing)}")

    df = df[df["scrape_status"] == "success"].copy()
    df = df[df["source"].isin(LABELS)].copy()
    df = df[df["raw_headline"].notna() & (df["raw_headline"].str.strip() != "")].copy()
    df["label"] = df["source"].map(LABELS)
    df = add_text_variants(df)
    df = df[df["headline_minimal"] != ""].copy()
    df = df.drop_duplicates(subset=["url"])
    df = df.drop_duplicates(subset=["headline_minimal", "source"])

    helper_urls = set(helper["url"])
    helper_headlines = set(normalized_key(helper["headline_minimal"]))
    df = df[~df["url"].isin(helper_urls)].copy()
    df = df[~normalized_key(df["headline_minimal"]).isin(helper_headlines)].copy()
    return df


def source_additions_needed(helper: pd.DataFrame, target: int) -> dict[str, int]:
    target_per_source = target // 2
    remainder = target - (2 * target_per_source)
    desired = {"FoxNews": target_per_source + remainder, "NBC": target_per_source}
    helper_counts = helper["source"].value_counts().to_dict()
    return {
        source: max(0, desired[source] - int(helper_counts.get(source, 0)))
        for source in LABELS
    }


def build_dataset(
    helper_csv: Path,
    scraped_csv: Path,
    output_csv: Path,
    target: int,
    seed: int,
) -> pd.DataFrame:
    helper = load_helper(helper_csv)
    if len(helper) > target:
        raise ValueError(
            f"Helper CSV has {len(helper)} rows, which exceeds target={target}."
        )

    scraped = load_scraped(scraped_csv, helper)
    additions = source_additions_needed(helper, target)

    sampled_parts = []
    for source, n_needed in additions.items():
        pool = scraped[scraped["source"] == source]
        if len(pool) < n_needed:
            raise ValueError(
                f"Not enough {source} scraped rows: need {n_needed}, have {len(pool)}."
            )
        sampled_parts.append(pool.sample(n=n_needed, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else scraped.head(0)
    combined = pd.concat([helper, sampled], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    columns = [
        "url",
        "source",
        "label",
        "raw_headline",
        "headline_minimal",
        "headline_lowercase",
        "headline_nopunct",
        "headline_nostop",
        "headline_lemma",
    ]
    combined = combined[columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced random-sample headline dataset."
    )
    parser.add_argument("--helper-csv", type=Path, default=HELPER_CSV)
    parser.add_argument("--scraped-csv", type=Path, default=SCRAPED_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--target", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        df = build_dataset(
            helper_csv=args.helper_csv,
            scraped_csv=args.scraped_csv,
            output_csv=args.output_csv,
            target=args.target,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    counts = df["source"].value_counts().to_dict()
    print(f"Saved {len(df):,} rows to {args.output_csv}")
    print(f"Source counts: {counts}")


if __name__ == "__main__":
    main()
