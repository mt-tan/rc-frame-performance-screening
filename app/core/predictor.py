"""Prediction pipeline: inputs -> parameters -> BASE+A2 rows -> RealMLP (main model) and
EBM (explanation only). Missing feature values are filled with the training medians, as in
ml/run_final.features."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import paths
from .features import LEVELS, frame_rows
from .params import build_params

CLS = ["IO", "LS", "CP", "C"]                         # performance levels, best to worst (paper and result files)
LEVEL_NAMES = {"IO": "Immediate Occupancy", "LS": "Life Safety",
               "CP": "Collapse Prevention", "C": "Collapse"}

REALMLP_PKL = paths.MODELS_DIR / "realmlp_BASE+A2_s42.pkl"
EBM_PKL = paths.MODELS_DIR / "ebm_BASE+A2_s42.pkl"
MEDIANS_JSON = paths.MODELS_DIR / "train_medians_BASE+A2.json"
RELIABILITY_JSON = paths.MODELS_DIR / "reliability_ext.json"


@dataclass
class LevelResult:
    level: str
    probs: dict                      # level (IO/LS/CP/C) -> probability (RealMLP)
    pred: str                        # predicted level
    confidence: float                # max probability
    low_confidence: bool
    bin_accuracy: float | None       # external-test accuracy of predictions with similar confidence
    p_worse_ext: float | None        # external test: share of this predicted level whose true level was worse
    p_worse_model: float             # model probability mass on levels worse than the prediction
    ebm_probs: dict
    ebm_pred: str
    ebm_terms: pd.DataFrame          # term, value, score for each class (logit scale)


@dataclass
class FrameResult:
    params: dict
    rows: pd.DataFrame               # model input rows (after median fill)
    a2: dict
    levels: dict = field(default_factory=dict)   # 'DD1'/'DD2' -> LevelResult


class Predictor:
    def __init__(self, realmlp_pkl=REALMLP_PKL, ebm_pkl=EBM_PKL):
        with open(realmlp_pkl, "rb") as fh:
            d = pickle.load(fh)
        self.model, self.columns = d["model"], d["columns"]
        with open(ebm_pkl, "rb") as fh:
            e = pickle.load(fh)
        self.ebm = e["model"]
        assert e["columns"] == self.columns, "EBM and RealMLP feature columns differ"
        self.medians = pd.Series(json.loads(MEDIANS_JSON.read_text(encoding="utf-8")))[self.columns]
        self.rel = json.loads(RELIABILITY_JSON.read_text(encoding="utf-8")) if RELIABILITY_JSON.exists() else None

    # ------------------------------------------------------------------ matrices
    def prepare(self, rows: pd.DataFrame) -> pd.DataFrame:
        return rows[self.columns].astype(float).fillna(self.medians)

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X.to_numpy(dtype=np.float64)), dtype=float)

    def ebm_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.ebm.predict_proba(X), dtype=float)

    # ------------------------------------------------------------------ one frame
    def predict_values(self, values: dict) -> FrameResult:
        return self.predict_params(build_params(values))

    def predict_params(self, p: dict) -> FrameResult:
        raw, a2 = frame_rows(p)
        X = self.prepare(raw)
        P, Pe = self.proba(X), self.ebm_proba(X)
        S = self.ebm.eval_terms(X)                        # (rows, terms, classes)
        res = FrameResult(params=p, rows=X, a2=a2)
        for i, lev in enumerate(LEVELS):
            k = int(P[i].argmax())
            pred = CLS[k]
            conf = float(P[i, k])
            bacc, low = self._bin(conf)
            pw = None
            if self.rel is not None:
                pw = self.rel["p_worse_given_pred"].get(CLS[k])
            terms = pd.DataFrame(S[i], columns=CLS)
            terms.insert(0, "term", self.ebm.term_names_)
            terms.insert(1, "value", [_term_value(X.iloc[i], t) for t in self.ebm.term_names_])
            res.levels[lev] = LevelResult(
                level=lev, probs={c: float(P[i, j]) for j, c in enumerate(CLS)},
                pred=pred, confidence=conf, low_confidence=low, bin_accuracy=bacc,
                p_worse_ext=pw, p_worse_model=float(P[i, k + 1:].sum()),
                ebm_probs={c: float(Pe[i, j]) for j, c in enumerate(CLS)},
                ebm_pred=CLS[int(Pe[i].argmax())], ebm_terms=terms)
        return res

    # ------------------------------------------------------------------ many frames
    def predict_many(self, plist: list[dict], ebm: bool = False) -> pd.DataFrame:
        """One row per (frame, level): name, level, p_IO..p_C, pred (IO/LS/CP/C), confidence,
        low-confidence flag; EBM probabilities when ``ebm``."""
        X = self.prepare(pd.concat([frame_rows(p)[0] for p in plist]))
        P = self.proba(X)
        out = pd.DataFrame({"name": np.repeat([p.get("name", "") for p in plist], len(LEVELS)),
                            "level": list(X.index)})
        for j, c in enumerate(CLS):
            out["p_" + c] = P[:, j]
        out["pred"] = np.array(CLS)[P.argmax(1)]
        out["confidence"] = P.max(1)
        out["low_confidence"] = [self._bin(c)[1] for c in out["confidence"]]
        if ebm:
            Pe = self.ebm_proba(X)
            for j, c in enumerate(CLS):
                out["ebm_p_" + c] = Pe[:, j]
            out["ebm_pred"] = np.array(CLS)[Pe.argmax(1)]
        return out

    def _bin(self, conf: float):
        if self.rel is None:
            return None, False
        for b in self.rel["confidence_bins"]:
            if b["lo"] <= conf <= b["hi"]:
                return b["acc"], bool(b["low"])
        return None, False


def _term_value(x: pd.Series, term: str) -> str:
    parts = [t.strip() for t in term.split("&")]
    return " & ".join(f"{x[t]:.4g}" for t in parts)
