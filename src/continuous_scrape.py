#!/usr/bin/env python3
"""
continuous_scrape.py - Run URL collection + scraping forever on a polling interval.

Workflow per cycle:
1) Collect fresh Fox/NBC article URLs into a temporary CSV.
2) Merge unique URLs into the master URLs CSV.
3) Run scrape.py in --resume mode to scrape only unseen URLs.
4) Sleep, then repeat until interrupted.

Usage:
    python src/continuous_scrape.py
    python src/continuous_scrape.py --interval-minutes 20 --per-source 60
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
SCRAPED_DIR = PROJECT_ROOT / "data" / "scraped"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_URLS = DATA_RAW_DIR / "original_urls.csv"
DEFAULT_TEMP_URLS = DATA_RAW_DIR / "latest_collected_urls.csv"
DEFAULT_COLLECTOR_STATE = DATA_RAW_DIR / "collect_urls_state.json"
DEFAULT_SCRAPED = SCRAPED_DIR / "raw_scraped_headlines.csv"
DEFAULT_LOG = LOGS_DIR / "continuous_scrape.log"


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_cmd(cmd: list[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def load_urls_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["url", "source"])
    df = pd.read_csv(path, dtype=str).fillna("")
    cols = {c.lower(): c for c in df.columns}
    if "url" not in cols or "source" not in cols:
        raise ValueError(f"{path} must contain 'url' and 'source' columns.")
    clean = df[[cols["url"], cols["source"]]].copy()
    clean.columns = ["url", "source"]
    clean["url"] = clean["url"].str.strip()
    clean["source"] = clean["source"].str.strip()
    clean = clean[(clean["url"] != "") & (clean["source"] != "")]
    return clean


def merge_unique_urls(master_path: Path, incoming_path: Path) -> int:
    master = load_urls_csv(master_path)
    incoming = load_urls_csv(incoming_path)
    if incoming.empty:
        return 0

    before = len(master)
    merged = pd.concat([master, incoming], ignore_index=True)
    merged = merged.drop_duplicates(subset=["url"], keep="first")
    merged.to_csv(master_path, index=False)
    added = len(merged) - before
    return max(0, added)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continuously collect and scrape Fox/NBC URLs on a schedule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--urls", type=Path, default=DEFAULT_URLS, help="Master URL CSV.")
    p.add_argument("--temp-urls", type=Path, default=DEFAULT_TEMP_URLS, help="Temporary per-cycle collected URL CSV.")
    p.add_argument("--output", type=Path, default=DEFAULT_SCRAPED, help="Scraped output CSV.")
    p.add_argument("--interval-minutes", type=float, default=20.0, help="Sleep time between cycles.")
    p.add_argument(
        "--per-source",
        type=int,
        default=80,
        help="URLs to collect per source each cycle. Use 0 to take all discovered URLs.",
    )
    p.add_argument("--collector-mode", choices=("backfill", "latest"), default="backfill", help="collect_urls.py selection mode.")
    p.add_argument("--collector-state-file", type=Path, default=DEFAULT_COLLECTOR_STATE, help="collect_urls.py backfill cursor state file.")
    p.add_argument("--collector-timeout", type=int, default=20, help="collect_urls.py timeout per request.")
    p.add_argument("--collector-retries", type=int, default=3, help="collect_urls.py retry count.")
    p.add_argument("--collector-delay", type=float, default=0.8, help="collect_urls.py base delay between requests.")
    p.add_argument("--collector-max-sitemaps", type=int, default=1500, help="collect_urls.py max sitemap files per source.")
    p.add_argument("--collector-max-sitemaps-foxnews", type=int, default=None, help="collect_urls.py FoxNews-only sitemap cap.")
    p.add_argument("--collector-max-sitemaps-nbc", type=int, default=None, help="collect_urls.py NBC-only sitemap cap.")
    p.add_argument("--scrape-timeout", type=int, default=15, help="scrape.py timeout per request.")
    p.add_argument("--scrape-retries", type=int, default=3, help="scrape.py max retries.")
    p.add_argument("--scrape-delay", type=float, default=1.0, help="scrape.py base delay.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(DEFAULT_LOG)

    args.urls.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Starting continuous scrape loop. Press Ctrl+C to stop.")
    cycle = 0
    try:
        while True:
            cycle += 1
            logging.info("=" * 60)
            logging.info("Cycle %d started", cycle)

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "collect_urls.py"),
                "--output",
                str(args.temp_urls),
                "--per-source",
                str(args.per_source),
                "--mode",
                str(args.collector_mode),
                "--state-file",
                str(args.collector_state_file),
                "--timeout",
                str(args.collector_timeout),
                "--max-retries",
                str(args.collector_retries),
                "--delay",
                str(args.collector_delay),
                "--max-sitemaps",
                str(args.collector_max_sitemaps),
            ]
            if args.collector_max_sitemaps_foxnews is not None:
                cmd.extend(["--max-sitemaps-foxnews", str(args.collector_max_sitemaps_foxnews)])
            if args.collector_max_sitemaps_nbc is not None:
                cmd.extend(["--max-sitemaps-nbc", str(args.collector_max_sitemaps_nbc)])
            run_cmd(cmd)

            added = merge_unique_urls(args.urls, args.temp_urls)
            logging.info("Cycle %d: added %d new URLs to %s", cycle, added, args.urls)

            run_cmd(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "src" / "scrape.py"),
                    "--urls",
                    str(args.urls),
                    "--output",
                    str(args.output),
                    "--resume",
                    "--delay",
                    str(args.scrape_delay),
                    "--timeout",
                    str(args.scrape_timeout),
                    "--max-retries",
                    str(args.scrape_retries),
                ]
            )

            sleep_seconds = max(1.0, args.interval_minutes * 60.0)
            logging.info("Cycle %d complete. Sleeping %.1f seconds.", cycle, sleep_seconds)
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        logging.info("Stopped by user after %d cycle(s).", cycle)


if __name__ == "__main__":
    main()
