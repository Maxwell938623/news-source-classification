from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")


def _minimal_clean(text: str) -> str:
    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    return _MULTI_WS.sub(" ", text).strip()


def _resolve_text_column(df: pd.DataFrame) -> str:
    preferred = [
        "headline_minimal",
        "headline_lowercase",
        "headline_nopunct",
        "headline",
        "raw_headline",
        "text",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:
        low = col.lower()
        if "headline" in low or "text" in low:
            return col
    return df.columns[0]


def _resolve_labels(df: pd.DataFrame) -> List[int]:
    for col in ["source", "label", "target", "y"]:
        if col not in df.columns:
            continue

        vals = df[col].fillna("").astype(str).tolist()
        if col == "label":
            out: List[int] = []
            for v in vals:
                vv = v.strip().lower()
                if vv == "0":
                    out.append(0)
                elif vv == "1":
                    out.append(1)
                else:
                    # If label isn't numeric, try source-style parsing.
                    if "fox" in vv:
                        out.append(0)
                    elif "nbc" in vv:
                        out.append(1)
                    else:
                        out.append(-1)
            return out
        if col in ("source", "target", "y"):
            out: List[int] = []
            for v in vals:
                vv = v.strip().lower()
                if "fox" in vv:
                    out.append(0)
                elif "nbc" in vv:
                    out.append(1)
                else:
                    out.append(-1)
            return out

    if "url" in df.columns:
        out: List[int] = []
        for u in df["url"].fillna("").astype(str).tolist():
            lu = u.lower()
            if "foxnews.com" in lu:
                out.append(0)
            elif "nbcnews.com" in lu:
                out.append(1)
            else:
                out.append(-1)
        return out

    return [-1] * len(df)


def prepare_data(csv_path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(Path(csv_path), dtype=str)
    text_col = _resolve_text_column(df)
    X = [_minimal_clean(t) for t in df[text_col].fillna("").astype(str).tolist()]
    y = _resolve_labels(df)
    return X, y
