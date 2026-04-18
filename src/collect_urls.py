#!/usr/bin/env python3
"""
collect_urls.py - Build large real Fox News + NBC News URL lists.

Collection strategy:
1) Crawl robots.txt-discovered sitemaps (including nested sitemap indexes).
2) Crawl configured sitemap seed URLs.
3) Pull RSS/Atom feeds.
4) Pull article links from section homepages.

Backfill mode:
- Sort discovered URLs by publication timestamp (newest -> oldest).
- Use a persisted per-source cursor so each run takes the next older slice.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_OUTPUT = DATA_RAW_DIR / "original_urls.csv"
DEFAULT_LOG = LOGS_DIR / "collect_urls.log"
DEFAULT_STATE_FILE = DATA_RAW_DIR / "collect_urls_state.json"

# ---------------------------------------------------------------------------
# Source configs
# ---------------------------------------------------------------------------

SOURCE_CONFIG = {
    "FoxNews": {
        "allowed_domains": ("foxnews.com",),
        "robots_url": "https://www.foxnews.com/robots.txt",
        "sitemap_seeds": (
            "https://www.foxnews.com/sitemap.xml",
            "https://www.foxnews.com/sitemap-index.xml",
            "https://moxie.foxnews.com/google-publisher/sitemap.xml",
        ),
        "seed_pages": (
            "https://www.foxnews.com/",
            "https://www.foxnews.com/politics",
            "https://www.foxnews.com/us",
            "https://www.foxnews.com/world",
            "https://www.foxnews.com/business",
        ),
        "feeds": (
            "https://moxie.foxnews.com/google-publisher/latest.xml",
            "https://feeds.foxnews.com/foxnews/latest",
            "https://feeds.foxnews.com/foxnews/politics",
            "https://feeds.foxnews.com/foxnews/national",
            "https://feeds.foxnews.com/foxnews/world",
            "https://feeds.foxnews.com/foxnews/scitech",
        ),
    },
    "NBC": {
        "allowed_domains": ("nbcnews.com",),
        "robots_url": "https://www.nbcnews.com/robots.txt",
        "sitemap_seeds": (
            "https://www.nbcnews.com/sitemap.xml",
            "https://www.nbcnews.com/sitemaps/sitemap-index.xml",
            "https://www.nbcnews.com/news-sitemap.xml",
        ),
        "seed_pages": (
            "https://www.nbcnews.com/",
            "https://www.nbcnews.com/politics",
            "https://www.nbcnews.com/us-news",
            "https://www.nbcnews.com/world",
            "https://www.nbcnews.com/business",
        ),
        "feeds": (
            "https://feeds.nbcnews.com/nbcnews/public/news",
            "https://feeds.nbcnews.com/nbcnews/public/politics",
            "https://feeds.nbcnews.com/nbcnews/public/world",
            "https://feeds.nbcnews.com/nbcnews/public/business",
            "https://feeds.nbcnews.com/nbcnews/public/science",
            "https://feeds.nbcnews.com/nbcnews/public/tech",
        ),
    },
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

BLOCKED_PATH_PARTS = (
    "/video",
    "/videos",
    "/live",
    "/newsletter",
    "/weather",
    "/shows",
    "/tv",
    "/shop",
    "/deals",
    "/podcasts",
    "/topic/",
    "/tag/",
    "/author/",
    "/profile/",
    "/about",
    "/contact",
)

TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "cid",
    "cmpid",
    "intcmp",
    "fbclid",
    "gclid",
}

DATE_IN_PATH_RE = re.compile(r"/(20\d{2})/(0[1-9]|1[0-2])/([0-3]\d)(?:/|$)")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def fetch_with_retry(url: str, timeout: int, max_retries: int, base_delay: float) -> str | None:
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            # Most 4xx errors are permanent for the requested URL; don't retry.
            # Keep retry behavior for throttling/timeouts where retry can help.
            if status in (400, 401, 403, 404, 410, 451):
                logging.warning("Fetch failed (permanent %s): %s", status, url)
                return None
            logging.warning("Fetch failed (%s/%s): %s -> HTTP %s", attempt, max_retries, url, status)
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.8))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Fetch failed (%s/%s): %s -> %s", attempt, max_retries, url, exc)
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.8))
    return None


def parse_timestamp(raw: str | None) -> int | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None

    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        pass

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def infer_epoch_from_url(url: str) -> int | None:
    m = DATE_IN_PATH_RE.search(urlparse(url).path)
    if not m:
        return None
    year, month, day = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    try:
        return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


def parse_sitemap_lines_from_robots(robots_text: str) -> list[str]:
    out: list[str] = []
    for raw in robots_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("sitemap:"):
            _, value = line.split(":", 1)
            maybe = value.strip()
            if maybe.startswith(("http://", "https://")):
                out.append(maybe)
    return out


def normalize_url(raw_url: str, allowed_domains: tuple[str, ...]) -> str | None:
    raw_url = (raw_url or "").strip()
    if not raw_url:
        return None
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    if not raw_url.startswith(("http://", "https://")):
        return None

    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not any(host == d or host.endswith("." + d) for d in allowed_domains):
        return None

    path = parsed.path.rstrip("/")
    if not path:
        return None
    lower_path = path.lower()
    if any(part in lower_path for part in BLOCKED_PATH_PARTS):
        return None
    if len([segment for segment in path.split("/") if segment]) < 2:
        return None

    kept_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in TRACKING_QUERY_KEYS
    ]
    normalized = parsed._replace(
        scheme="https",
        netloc=host,
        query=urlencode(kept_query),
        fragment="",
    )
    return urlunparse(normalized)


def parse_sitemap_document(
    xml_text: str, allowed_domains: tuple[str, ...]
) -> tuple[list[str], list[dict[str, int | str | None]]]:
    soup = BeautifulSoup(xml_text, "xml")
    nested: list[str] = []
    records: list[dict[str, int | str | None]] = []

    for node in soup.find_all("sitemap"):
        loc = node.find("loc")
        if loc:
            raw = loc.get_text(strip=True)
            if raw.startswith(("http://", "https://")):
                nested.append(raw)

    for node in soup.find_all("url"):
        loc = node.find("loc")
        if not loc:
            continue
        norm = normalize_url(loc.get_text(strip=True), allowed_domains)
        if not norm:
            continue
        lastmod = node.find("lastmod")
        epoch = parse_timestamp(lastmod.get_text(strip=True) if lastmod else None)
        if epoch is None:
            epoch = infer_epoch_from_url(norm)
        records.append({"url": norm, "epoch": epoch})
    return nested, records


def parse_feed_records(feed_text: str, allowed_domains: tuple[str, ...]) -> list[dict[str, int | str | None]]:
    soup = BeautifulSoup(feed_text, "xml")
    out: list[dict[str, int | str | None]] = []

    for item in soup.find_all(["item", "entry"]):
        link = None
        link_node = item.find("link")
        if link_node:
            href = (link_node.get("href") or "").strip()
            text_link = link_node.get_text(strip=True)
            link = href or text_link
        if not link:
            continue
        norm = normalize_url(link, allowed_domains)
        if not norm:
            continue

        epoch = None
        for tag in ("pubDate", "published", "updated", "dc:date", "date"):
            t = item.find(tag)
            if t and t.get_text(strip=True):
                epoch = parse_timestamp(t.get_text(strip=True))
                if epoch is not None:
                    break
        if epoch is None:
            epoch = infer_epoch_from_url(norm)
        out.append({"url": norm, "epoch": epoch})
    return out


def extract_links_from_html(page_url: str, html: str, allowed_domains: tuple[str, ...]) -> list[dict[str, int | str | None]]:
    soup = BeautifulSoup(html, "lxml")
    out: list[dict[str, int | str | None]] = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(page_url, a["href"])
        norm = normalize_url(absolute, allowed_domains)
        if not norm:
            continue
        out.append({"url": norm, "epoch": infer_epoch_from_url(norm)})
    return out


def dedupe_records(records: list[dict[str, int | str | None]]) -> list[dict[str, int | str | None]]:
    by_url: dict[str, dict[str, int | str | None]] = {}
    for rec in records:
        url = str(rec["url"])
        epoch = rec.get("epoch")
        if url not in by_url:
            by_url[url] = {"url": url, "epoch": epoch}
            continue
        prior = by_url[url].get("epoch")
        if prior is None and epoch is not None:
            by_url[url]["epoch"] = epoch
        elif prior is not None and epoch is not None and int(epoch) > int(prior):
            by_url[url]["epoch"] = epoch
    return list(by_url.values())


def collect_from_sitemaps(
    source: str,
    timeout: int,
    max_retries: int,
    delay: float,
    max_sitemaps: int,
) -> list[dict[str, int | str | None]]:
    cfg = SOURCE_CONFIG[source]
    allowed_domains = cfg["allowed_domains"]
    seed_sitemaps: list[str] = []

    robots = fetch_with_retry(cfg["robots_url"], timeout, max_retries, delay)
    if robots:
        seed_sitemaps.extend(parse_sitemap_lines_from_robots(robots))
    seed_sitemaps.extend(list(cfg["sitemap_seeds"]))

    seen: set[str] = set()
    queue: deque[str] = deque(seed_sitemaps)
    out: list[dict[str, int | str | None]] = []

    while queue and len(seen) < max_sitemaps:
        sitemap_url = queue.popleft()
        if sitemap_url in seen:
            continue
        seen.add(sitemap_url)

        xml_text = fetch_with_retry(sitemap_url, timeout, max_retries, delay)
        if not xml_text:
            continue
        nested, records = parse_sitemap_document(xml_text, allowed_domains)
        out.extend(records)
        for nxt in nested:
            if nxt not in seen:
                queue.append(nxt)

        if len(seen) % 25 == 0:
            logging.info(
                "%s sitemap progress: %d sitemap files visited, %d raw records.",
                source,
                len(seen),
                len(out),
            )
        time.sleep(delay + random.uniform(0.0, 0.2))

    logging.info(
        "%s sitemap crawl done: %d sitemap files visited, %d raw records.",
        source,
        len(seen),
        len(out),
    )
    return out


def collect_source_records(
    source: str,
    timeout: int,
    max_retries: int,
    delay: float,
    max_sitemaps: int,
) -> list[dict[str, int | str | None]]:
    cfg = SOURCE_CONFIG[source]
    allowed_domains = cfg["allowed_domains"]
    records: list[dict[str, int | str | None]] = []

    logging.info("Collecting %s from sitemaps...", source)
    records.extend(
        collect_from_sitemaps(
            source=source,
            timeout=timeout,
            max_retries=max_retries,
            delay=delay,
            max_sitemaps=max_sitemaps,
        )
    )

    logging.info("Collecting %s from feeds...", source)
    for feed_url in cfg["feeds"]:
        text = fetch_with_retry(feed_url, timeout, max_retries, delay)
        if not text:
            continue
        records.extend(parse_feed_records(text, allowed_domains))
        time.sleep(delay + random.uniform(0.0, 0.4))

    logging.info("Collecting %s from seed pages...", source)
    for page_url in cfg["seed_pages"]:
        html = fetch_with_retry(page_url, timeout, max_retries, delay)
        if not html:
            continue
        records.extend(extract_links_from_html(page_url, html, allowed_domains))
        time.sleep(delay + random.uniform(0.0, 0.4))

    deduped = dedupe_records(records)
    logging.info("%s deduped records: %d", source, len(deduped))
    return deduped


def sort_by_newest(records: list[dict[str, int | str | None]]) -> list[dict[str, int | str | None]]:
    return sorted(
        records,
        key=lambda rec: (
            rec.get("epoch") is not None,
            int(rec["epoch"]) if rec.get("epoch") is not None else -1,
            str(rec["url"]),
        ),
        reverse=True,
    )


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def select_records(
    source: str,
    records: list[dict[str, int | str | None]],
    mode: str,
    per_source: int,
    state: dict,
) -> list[str]:
    ordered = sort_by_newest(records)
    if per_source <= 0:
        selected = ordered
        if mode == "backfill":
            state[source] = {
                "cursor": len(ordered),
                "total_candidates": len(ordered),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            logging.info(
                "%s backfill slice: start=0 end=%d total=%d (all records)",
                source,
                len(ordered),
                len(ordered),
            )
        return [str(r["url"]) for r in selected]

    if mode == "latest":
        return [str(r["url"]) for r in ordered[:per_source]]

    cursor = int(state.get(source, {}).get("cursor", 0))
    start = min(cursor, len(ordered))
    end = min(start + per_source, len(ordered))
    selected = ordered[start:end]

    state[source] = {
        "cursor": end,
        "total_candidates": len(ordered),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    logging.info(
        "%s backfill slice: start=%d end=%d total=%d",
        source,
        start,
        end,
        len(ordered),
    )
    return [str(r["url"]) for r in selected]


def write_output(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect real Fox News and NBC News article URLs into a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="Backfill cursor state JSON path.")
    parser.add_argument("--mode", choices=("backfill", "latest"), default="backfill", help="Selection mode.")
    parser.add_argument("--reset-state", action="store_true", help="Reset backfill cursor before collecting.")
    parser.add_argument(
        "--per-source",
        type=int,
        default=2000,
        help="URLs to output per source. Use 0 to output all discovered URLs.",
    )
    parser.add_argument(
        "--max-sitemaps",
        type=int,
        default=3000,
        help="Maximum sitemap XML files to crawl per source.",
    )
    parser.add_argument(
        "--max-sitemaps-foxnews",
        type=int,
        default=None,
        help="Override max sitemap XML files for FoxNews only.",
    )
    parser.add_argument(
        "--max-sitemaps-nbc",
        type=int,
        default=None,
        help="Override max sitemap XML files for NBC only.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per request.")
    parser.add_argument("--delay", type=float, default=0.8, help="Base delay between requests.")
    args = parser.parse_args()

    setup_logging(DEFAULT_LOG)
    state = {} if args.reset_state else load_state(args.state_file)

    rows: list[dict[str, str]] = []
    max_sitemaps_by_source = {
        "FoxNews": args.max_sitemaps_foxnews if args.max_sitemaps_foxnews is not None else args.max_sitemaps,
        "NBC": args.max_sitemaps_nbc if args.max_sitemaps_nbc is not None else args.max_sitemaps,
    }
    for source in ("FoxNews", "NBC"):
        records = collect_source_records(
            source=source,
            timeout=args.timeout,
            max_retries=args.max_retries,
            delay=args.delay,
            max_sitemaps=max_sitemaps_by_source[source],
        )
        urls = select_records(
            source=source,
            records=records,
            mode=args.mode,
            per_source=args.per_source,
            state=state,
        )
        for url in urls:
            rows.append({"url": url, "source": source})
        logging.info("%s: selected %d URLs for output.", source, len(urls))

    if args.mode == "backfill":
        save_state(args.state_file, state)
        logging.info("Saved state to %s", args.state_file)

    if not rows:
        raise SystemExit("No URLs selected. Increase max-sitemaps or reset state.")
    write_output(rows, args.output)
    logging.info("Wrote %d total URLs to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
