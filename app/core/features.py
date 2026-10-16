"""BASE + A2 feature rows for one frame (one row per hazard level), identical to
ml/data_m3.build_rows: BASE from features_m3.base_features(p, level) plus the level code,
A2 from features_m3.a2_features(p) with the level-specific A2_Sae/Sde/R/theta columns."""
from __future__ import annotations

import pandas as pd

from . import paths  # noqa: F401
import config
import features_m3 as F

LEVELS = list(config.HAZARD_LEVELS)          # ['DD1', 'DD2']
LEVEL_LABEL = {"DD1": "DD-1", "DD2": "DD-2"}


def frame_rows(p: dict) -> tuple[pd.DataFrame, dict]:
    """Returns (rows indexed by level with B__/A2__ columns, raw A2 dict)."""
    a2 = F.a2_features(p)
    rows = []
    for li, lev in enumerate(LEVELS):
        row = {"B__level": float(li)}
        for k, v in F.base_features(p, lev).items():
            row["B__" + k] = float(v)
        for k, v in a2.items():
            if k.endswith("_DD1") or k.endswith("_DD2"):
                if not k.endswith("_" + lev):
                    continue
                k = k[:-4]
            row["A2__" + k[3:]] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows, index=pd.Index(LEVELS, name="level"))
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    return df, a2


def spectrum(p: dict) -> dict:
    """Design-spectrum parameters of the frame's site and soil class for both levels."""
    return {lev: config.site_spectrum(p["site"], lev, p["soil_class"]) for lev in LEVELS}
