#!/usr/bin/env python3
"""
scrape.py — Scrape Fox News and NBC News headlines from the provided URL CSV.

Label encoding (used everywhere in this project):
    0  =  FoxNews
    1  =  NBC

Usage:
    python src/scrape.py
    python src/scrape.py --urls data/raw/original_urls.csv
    python src/scrape.py --urls data/raw/original_urls.csv --resume
    python src/scrape.py --delay 1.5 --timeout 20 --max-retries 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
SCRAPED_DIR = PROJECT_ROOT / "data" / "scraped"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_URLS_CSV = DATA_RAW_DIR / "original_urls.csv"
DEFAULT_OUTPUT = SCRAPED_DIR / "raw_scraped_headlines.csv"
DEFAULT_LOG = LOGS_DIR / "scrape.log"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Label encoding — must be consistent with every other script in this project.
LABEL_MAP: dict[str, int] = {"FoxNews": 0, "NBC": 1}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

# Suffixes that og:title often appends; strip them before saving.
_OG_SUFFIX_RE = re.compile(
    r"\s*[\|–—\-]\s*"
    r"(Fox News|FOX NEWS|NBC News|NBC NEWS|NBCNews\.com|MSNBC\.com|MSNBC)"
    r"\s*$",
    re.IGNORECASE,
)

# Multiple-whitespace normaliser
_MULTI_WS = re.compile(r"\s+")

# Output CSV columns (order matters for DictWriter)
OUTPUT_FIELDS = ["url", "source", "label", "raw_headline", "scrape_status", "notes"]


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def infer_source_from_url(url: str) -> str | None:
    """
    Determine 'FoxNews' or 'NBC' purely from the URL domain.
    Returns None if the domain is unrecognised.
    """
    lower = url.lower()
    if "foxnews.com" in lower or "fox news" in lower:
        return "FoxNews"
    if "nbcnews.com" in lower or "msnbc.com" in lower:
        return "NBC"
    return None


# ---------------------------------------------------------------------------
# Headline extraction
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Strip and collapse whitespace."""
    return _MULTI_WS.sub(" ", text).strip()


def _try_h1_with_keyword(soup: BeautifulSoup, *keywords: str) -> str | None:
    """
    Find the first <h1> whose class attribute contains any of the given keywords
    (case-insensitive).  Returns cleaned text or None.
    """
    for h1 in soup.find_all("h1"):
        classes = " ".join(h1.get("class") or []).lower()
        if any(kw in classes for kw in keywords):
            text = _clean_text(h1.get_text(separator=" "))
            if text:
                return text
    return None


def _try_first_h1(soup: BeautifulSoup) -> str | None:
    """Return text from the very first <h1> on the page, or None."""
    h1 = soup.find("h1")
    if h1:
        text = _clean_text(h1.get_text(separator=" "))
        if text:
            return text
    return None


def _try_og_title(soup: BeautifulSoup) -> str | None:
    """
    Extract og:title meta content and strip trailing source suffix.
    Returns cleaned text or None.
    """
    og = soup.find("meta", attrs={"property": "og:title"})
    if og:
        raw = (og.get("content") or "").strip()
        if raw:
            cleaned = _OG_SUFFIX_RE.sub("", raw).strip()
            if cleaned:
                return cleaned
    return None


def _try_meta_title(soup: BeautifulSoup) -> str | None:
    """Last-resort: parse <title> tag and strip suffix."""
    title_tag = soup.find("title")
    if title_tag:
        raw = _clean_text(title_tag.get_text())
        cleaned = _OG_SUFFIX_RE.sub("", raw).strip()
        # Also strip common separator patterns if no og: was found
        cleaned = re.sub(r"\s*[\|–—\-]\s*.{0,30}$", "", cleaned).strip()
        if cleaned:
            return cleaned
    return None


def extract_headline(soup: BeautifulSoup, source: str) -> str | None:
    """
    Try a cascade of selectors appropriate to the source.
    Returns the best headline text found, or None if nothing usable was found.

    Cascade:
      1. source-specific class keywords on any <h1>
      2. any itemprop="headline" element
      3. first bare <h1>
      4. og:title (with suffix stripped)
      5. <title> (with suffix stripped, last resort)
    """
    if source == "FoxNews":
        result = _try_h1_with_keyword(soup, "headline", "article-head", "title")
    else:  # NBC / MSNBC
        result = _try_h1_with_keyword(soup, "headline", "article", "hero", "title")

    if result:
        return result

    # itemprop="headline" is used by many news CMSes
    el = soup.find(attrs={"itemprop": "headline"})
    if el:
        text = _clean_text(el.get_text(separator=" "))
        if text:
            return text

    result = _try_first_h1(soup)
    if result:
        return result

    result = _try_og_title(soup)
    if result:
        return result

    return _try_meta_title(soup)


# ---------------------------------------------------------------------------
# HTTP fetch with retry / backoff
# ---------------------------------------------------------------------------

def fetch_with_retry(
    url: str,
    session: requests.Session,
    timeout: int,
    max_retries: int,
    base_delay: float,
) -> tuple[requests.Response | None, str]:
    """
    GET a URL up to max_retries times with exponential backoff.

    Returns:
        (response, "")           on success
        (None,     error_msg)    on permanent or exhausted failure
    """
    last_msg = "unknown_error"
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=timeout,
                allow_redirects=True,
            )
            resp.raise_for_status()
            return resp, ""

        except requests.exceptions.TooManyRedirects:
            return None, "too_many_redirects"  # permanent — don't retry

        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            last_msg = f"http_{code}"
            if code in (400, 401, 403, 404, 410, 451):
                return None, last_msg  # client errors — don't retry

        except requests.exceptions.Timeout:
            last_msg = f"timeout_attempt_{attempt}"

        except requests.exceptions.ConnectionError as exc:
            last_msg = f"connection_error: {exc}"

        except Exception as exc:  # noqa: BLE001
            last_msg = f"unexpected: {exc}"

        if attempt < max_retries:
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 1.0)
            time.sleep(wait)

    return None, last_msg


# ---------------------------------------------------------------------------
# CSV / persistence helpers
# ---------------------------------------------------------------------------

def load_done_urls(output_path: Path) -> set[str]:
    """Return the set of URLs already written to the output file (for --resume)."""
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["url"], dtype=str)
        return set(existing["url"].dropna().str.strip().tolist())
    except Exception:
        return set()


def detect_url_column(df: pd.DataFrame) -> str:
    """Return the first column name that looks like a URL column."""
    for candidate in ("url", "URL", "link", "Link", "article_url", "article_link"):
        if candidate in df.columns:
            return candidate
    return df.columns[0]


def detect_source_column(df: pd.DataFrame) -> str | None:
    """Return a source/label column name if present, else None."""
    for candidate in ("source", "Source", "label", "Label", "class", "site"):
        if candidate in df.columns:
            return candidate
    return None


def get_source_for_row(
    df: pd.DataFrame,
    url_col: str,
    source_col: str | None,
    url: str,
) -> str | None:
    """
    Look up source label for a URL:
      - from an explicit source column if available
      - otherwise infer from the URL domain
    """
    if source_col is not None:
        matches = df.index[df[url_col] == url]
        if len(matches):
            return str(df.loc[matches[0], source_col]).strip()
    return infer_source_from_url(url)


# ---------------------------------------------------------------------------
# Main scrape routine
# ---------------------------------------------------------------------------

def scrape(
    urls_csv: Path,
    output_path: Path,
    delay: float,
    timeout: int,
    max_retries: int,
    resume: bool,
) -> None:
    # ---- Directories & logging ----------------------------------------
    SCRAPED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _setup_logging(DEFAULT_LOG)
    log = logging.getLogger(__name__)

    # ---- Load URL list ------------------------------------------------
    if not urls_csv.exists():
        log.error("URL file not found: %s", urls_csv)
        sys.exit(1)

    log.info("Loading URLs from: %s", urls_csv)
    url_df = pd.read_csv(urls_csv, dtype=str)
    url_col = detect_url_column(url_df)
    source_col = detect_source_column(url_df)

    if source_col:
        log.info("Explicit source column detected: '%s'", source_col)
    else:
        log.info("No source column found — inferring source from URL domain.")

    urls: list[str] = url_df[url_col].dropna().str.strip().tolist()
    log.info("Total URLs in file: %d", len(urls))

    # ---- Resume: skip already-scraped URLs ----------------------------
    done_urls: set[str] = set()
    if resume:
        done_urls = load_done_urls(output_path)
        log.info("Resume mode: %d URLs already scraped, will skip them.", len(done_urls))

    pending = [u for u in urls if u not in done_urls]
    log.info("URLs remaining to scrape: %d", len(pending))

    if not pending:
        log.info("Nothing to do — all URLs already scraped.")
        return

    # ---- Open output CSV ---------------------------------------------
    # Use append mode only when resuming; otherwise overwrite so stale rows
    # and duplicate headers from a previous non-resume run are not carried over.
    file_mode = "a" if resume else "w"
    write_header = (file_mode == "w") or (not output_path.exists()) or (output_path.stat().st_size == 0)
    out_fh = open(output_path, file_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    # ---- Scrape loop -------------------------------------------------
    session = requests.Session()
    counts = {"success": 0, "no_headline": 0, "fetch_failed": 0, "skipped": 0}

    try:
        for url in tqdm(pending, desc="Scraping", unit="url", file=sys.stdout):
            source = get_source_for_row(url_df, url_col, source_col, url)

            # Skip unrecognised sources
            if source not in LABEL_MAP:
                msg = f"unrecognised source '{source}'"
                log.warning("SKIP  %s  — %s", url, msg)
                writer.writerow({
                    "url": url,
                    "source": source or "unknown",
                    "label": "",
                    "raw_headline": "",
                    "scrape_status": "skipped",
                    "notes": msg,
                })
                counts["skipped"] += 1
                out_fh.flush()
                continue

            label = LABEL_MAP[source]

            # Fetch page
            resp, err = fetch_with_retry(
                url, session,
                timeout=timeout,
                max_retries=max_retries,
                base_delay=delay,
            )

            if resp is None:
                log.warning("FAIL  [%s]  %s  → %s", source, url, err)
                writer.writerow({
                    "url": url,
                    "source": source,
                    "label": label,
                    "raw_headline": "",
                    "scrape_status": "fetch_failed",
                    "notes": err,
                })
                counts["fetch_failed"] += 1
                out_fh.flush()
                time.sleep(delay + random.uniform(0.0, delay * 0.3))
                continue

            # Parse HTML and extract headline
            soup = BeautifulSoup(resp.text, "lxml")
            headline = extract_headline(soup, source)

            if headline:
                log.debug("OK    [%s]  %s  → %.80s", source, url, headline)
                writer.writerow({
                    "url": url,
                    "source": source,
                    "label": label,
                    "raw_headline": headline,
                    "scrape_status": "success",
                    "notes": "",
                })
                counts["success"] += 1
            else:
                log.warning("MISS  [%s]  %s  — no headline found", source, url)
                writer.writerow({
                    "url": url,
                    "source": source,
                    "label": label,
                    "raw_headline": "",
                    "scrape_status": "no_headline_found",
                    "notes": "no h1 / itemprop / og:title found",
                })
                counts["no_headline"] += 1

            out_fh.flush()

            # Polite delay with random jitter to avoid rate-limiting
            time.sleep(delay + random.uniform(0.0, delay * 0.5))

    finally:
        out_fh.close()
        session.close()

    # ---- Final summary -----------------------------------------------
    total_processed = sum(counts.values())
    log.info("=" * 60)
    log.info("Scraping complete.")
    log.info("  Processed:       %d", total_processed)
    log.info("  Success:         %d  (%.1f%%)", counts["success"],
             100 * counts["success"] / max(total_processed, 1))
    log.info("  No headline:     %d", counts["no_headline"])
    log.info("  Fetch failed:    %d", counts["fetch_failed"])
    log.info("  Skipped:         %d", counts["skipped"])
    log.info("  Output file:     %s", output_path)
    log.info("=" * 60)


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
        description="Scrape Fox News and NBC News headlines from a URL CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--urls",
        metavar="PATH",
        type=Path,
        default=DEFAULT_URLS_CSV,
        help="Path to the starter URL CSV.",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV for scraped results.",
    )
    p.add_argument(
        "--delay",
        metavar="SECONDS",
        type=float,
        default=1.0,
        help="Base polite delay between requests (jitter is added automatically).",
    )
    p.add_argument(
        "--timeout",
        metavar="SECONDS",
        type=int,
        default=15,
        help="Per-request timeout.",
    )
    p.add_argument(
        "--max-retries",
        metavar="N",
        type=int,
        default=3,
        help="Maximum retry attempts on transient failures.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip URLs already present in the output file.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    scrape(
        urls_csv=args.urls,
        output_path=args.output,
        delay=args.delay,
        timeout=args.timeout,
        max_retries=args.max_retries,
        resume=args.resume,
    )
