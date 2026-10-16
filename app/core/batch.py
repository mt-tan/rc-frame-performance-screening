"""Batch evaluation: a CSV/XLSX table with one frame per row (columns = schema keys, optional 'id')."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .predictor import CLS
from .schema import FIELDS
from .validator import parse


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    missing = [f.key for f in FIELDS if f.key not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    if "id" not in df.columns:
        df.insert(0, "id", [f"row{i + 1}" for i in range(len(df))])
    return df


def evaluate(predictor, df: pd.DataFrame) -> pd.DataFrame:
    """One output row per input row: DD-1/DD-2 levels (IO/LS/CP/C), probabilities, flags."""
    checks = [parse(r) for r in df.to_dict("records")]
    good = [(i, c) for i, c in enumerate(checks) if c.ok]
    plist = []
    for i, c in good:
        p = dict(c.params)
        p["name"] = str(i)                      # row position (ids may repeat)
        plist.append(p)
    pred = predictor.predict_many(plist) if plist else pd.DataFrame()
    rows = []
    for i, c in enumerate(checks):
        r = {"id": df["id"].iloc[i], "status": "ok" if c.ok else "error",
             "messages": " | ".join(c.errors or c.warnings), "extrapolation": bool(c.warnings)}
        if c.ok:
            sub = pred[pred["name"] == str(i)].set_index("level")
            for lev, tag in (("DD1", "DD1"), ("DD2", "DD2")):
                s = sub.loc[lev]
                r[f"{tag}_level"] = s["pred"]
                r[f"{tag}_confidence"] = round(float(s["confidence"]), 4)
                r[f"{tag}_low_confidence"] = bool(s["low_confidence"])
                for c_ in CLS:
                    r[f"{tag}_p_{c_}"] = round(float(s["p_" + c_]), 4)
        rows.append(r)
    return pd.DataFrame(rows)
