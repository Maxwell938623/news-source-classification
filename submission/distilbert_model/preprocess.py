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
    for col in ["headline_minimal", "headline", "raw_headline", "text"]:
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
        out: List[int] = []
        for val in df[col].fillna("").astype(str).tolist():
            v = val.strip().lower()
            if v == "0" or "fox" in v:
                out.append(0)
            elif v == "1" or "nbc" in v:
                out.append(1)
            else:
                out.append(-1)
        return out

    if "url" in df.columns:
        out = []
        for url in df["url"].fillna("").astype(str).tolist():
            u = url.lower()
            if "foxnews.com" in u:
                out.append(0)
            elif "nbcnews.com" in u:
                out.append(1)
            else:
                out.append(-1)
        return out

    return [-1] * len(df)


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(Path(path), dtype=str)
    text_col = _resolve_text_column(df)
    X = [_minimal_clean(t) for t in df[text_col].fillna("").astype(str).tolist()]
    y = _resolve_labels(df)
    return X, y
