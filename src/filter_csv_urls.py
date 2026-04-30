#!/usr/bin/env python3
"""
filter_csv_urls.py - Remove CSV rows whose URL contains blocked substrings.

Example:
    python src/filter_csv_urls.py \
        --input data/scraped/raw_scraped_headlines_nbc.csv \
        --output data/scraped/raw_scraped_headlines_nbc_clean.csv \
        --contains "/video" --contains "utm_" --contains "newsletter"

You can also provide substrings from a text file (one per line):
    python src/filter_csv_urls.py \
        --input data/raw/original_urls.csv \
        --output data/raw/original_urls_clean.csv \
        --contains-file data/raw/blocked_substrings.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "original_urls.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "original_urls_clean.csv"

URL_COLUMN_CANDIDATES = ("url", "URL", "link", "Link", "href", "Href")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter rows from a CSV when URL contains blocked substrings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Substring to block when found in URL. Repeat this flag for multiple values.",
    )
    parser.add_argument(
        "--contains-file",
        type=Path,
        default=None,
        help="Text file with blocked substrings (one per line; blank lines ignored).",
    )
    parser.add_argument(
        "--url-column",
        default=None,
        help="Explicit URL column name. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive matching (default is case-insensitive).",
    )
    return parser.parse_args()


def _load_patterns(args: argparse.Namespace) -> list[str]:
    patterns = [s for s in args.contains if s]

    if args.contains_file is not None:
        if not args.contains_file.exists():
            raise FileNotFoundError(f"Contains file not found: {args.contains_file}")
        for raw in args.contains_file.read_text(encoding="utf-8").splitlines():
            value = raw.strip()
            if value and not value.startswith("#"):
                patterns.append(value)

    # Keep order but remove exact duplicates.
    deduped = list(dict.fromkeys(patterns))
    if not deduped:
        raise ValueError(
            "No blocked substrings were provided. Use --contains and/or --contains-file."
        )
    return deduped


def _detect_url_column(fieldnames: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(
                f"Specified URL column '{explicit}' not found. Available columns: {fieldnames}"
            )
        return explicit

    for candidate in URL_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate

    raise ValueError(
        "Could not auto-detect URL column. Pass --url-column.\n"
        f"Available columns: {fieldnames}"
    )


def _matches_blocked(url: str, patterns: list[str], case_sensitive: bool) -> bool:
    if case_sensitive:
        return any(pat in url for pat in patterns)

    lower_url = url.lower()
    return any(pat.lower() in lower_url for pat in patterns)


def filter_csv(
    input_path: Path,
    output_path: Path,
    patterns: list[str],
    url_column: str | None = None,
    case_sensitive: bool = False,
) -> tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    removed = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as in_file:
        reader = csv.DictReader(in_file)
        if not reader.fieldnames:
            raise ValueError("Input CSV appears to be empty or missing headers.")

        chosen_url_col = _detect_url_column(reader.fieldnames, explicit=url_column)

        with output_path.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                url_value = str(row.get(chosen_url_col, "") or "")
                if _matches_blocked(url_value, patterns, case_sensitive):
                    removed += 1
                    continue

                writer.writerow(row)
                kept += 1

    return kept, removed


def main() -> None:
    args = _parse_args()
    try:
        patterns = _load_patterns(args)
        kept, removed = filter_csv(
            input_path=args.input,
            output_path=args.output,
            patterns=patterns,
            url_column=args.url_column,
            case_sensitive=args.case_sensitive,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    total = kept + removed
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Patterns:   {len(patterns)}")
    print(f"Rows total: {total}")
    print(f"Rows kept:  {kept}")
    print(f"Rows cut:   {removed}")


if __name__ == "__main__":
    main()
