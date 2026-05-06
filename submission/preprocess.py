from __future__ import annotations
import html
import re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")
def _minimal_clean(text: str) -> str:
    text = html.unescape(text)
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
        if "headline" in col.lower() or "text" in col.lower():
            return col
    return df.columns[0]
def _resolve_labels(df: pd.DataFrame) -> List[str]:
    for col in ["source", "label", "target", "y"]:
        if col in df.columns:
            vals = df[col].fillna("").astype(str).tolist()
            if col == "label":
                out: List[str] = []
                for v in vals:
                    vv = v.strip().lower()
                    if vv == "0":
                        out.append("FoxNews")
                    elif vv == "1":
                        out.append("NBC")
                    else:
                        out.append(v)
                return out
            return vals
    if "url" in df.columns:
        labels: List[str] = []
        for u in df["url"].fillna("").astype(str).tolist():
            lu = u.lower()
            if "foxnews.com" in lu:
                labels.append("FoxNews")
            elif "nbcnews.com" in lu:
                labels.append("NBC")
            else:
                labels.append("unknown")
        return labels
    return ["unknown"] * len(df)
def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    path = Path(csv_path)
    df = pd.read_csv(path, dtype=str)
    text_col = _resolve_text_column(df)
    X = [_minimal_clean(t) for t in df[text_col].fillna("").astype(str).tolist()]
    y = _resolve_labels(df)
    return X, y
