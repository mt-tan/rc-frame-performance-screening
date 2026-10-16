"""Input parsing and checks.

errors   : the input cannot describe a frame (missing/non-numeric/non-positive values,
           tie legs exceeding the bars they hold) -> no prediction.
warnings : the frame lies outside the simulated population (generator acceptance filters,
           tbdy_filters.filter_all, or the training ranges in assets/domain.json) -> the
           prediction is an extrapolation and is flagged as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import paths  # noqa: F401
import tbdy_filters
from .params import build_params
from .schema import FIELDS, FIELD_BY_KEY, load_domain


@dataclass
class Check:
    values: dict = field(default_factory=dict)
    params: dict | None = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def parse(raw: dict) -> Check:
    c = Check()
    for f in FIELDS:
        s = raw.get(f.key, "")
        s = s.strip() if isinstance(s, str) else s
        try:
            if f.kind == "choice":
                s = str(s)
                if s not in f.choices:
                    raise ValueError
                c.values[f.key] = s
            elif f.kind == "list":
                items = s if isinstance(s, (list, tuple)) else [x for x in str(s).replace(";", ",").split(",") if x.strip()]
                vals = [float(x) for x in items]
                if not vals or min(vals) <= 0:
                    raise ValueError
                c.values[f.key] = vals
            elif f.kind == "int":
                v = float(s)
                if v != int(v) or v <= 0:
                    raise ValueError
                c.values[f.key] = int(v)
            else:
                v = float(s)
                if not v > 0:
                    raise ValueError
                c.values[f.key] = v
        except (TypeError, ValueError):
            c.errors.append(f"{f.label}: invalid value '{s}'")
    if c.errors:
        return c
    v = c.values
    if v["col_n_long_x"] < 2 or v["col_n_long_y"] < 2:
        c.errors.append("Column bars per face must be at least 2 (corner bars).")
    if v["col_n_etr_y"] > v["col_n_long_x"] or v["col_n_etr_x"] > v["col_n_long_y"]:
        c.errors.append("Tie legs cannot exceed the bars on the face they cross.")
    if v["kappa_col"] > 1.0 or v["kappa_beam"] > 1.0:
        c.errors.append("As-built reinforcement factor κ must not exceed 1.0.")
    if c.errors:
        return c
    try:
        c.params = build_params(v)
    except Exception as exc:                  # pragma: no cover - defensive
        c.errors.append(f"Parameter derivation failed: {exc}")
        return c
    ok, name, reason, _ = tbdy_filters.filter_all(c.params)
    if not ok:
        c.warnings.append(f"Outside the generator's acceptance rules ({name}): {reason}")
    _domain(c)
    return c


def _domain(c: Check):
    dom = load_domain()
    v, p = c.values, c.params
    checks = dict(v)
    checks.update({"N_bay": p["N_bay"], "bay_length_min": p["L_min"], "bay_length_max": p["L_max"],
                   "col_aspect": p["col_aspect"], "rho_col_actual": p["rho_col_actual"],
                   "rho_beam_top_actual": p["rho_beam_top_actual"],
                   "rho_beam_bot_actual": p["rho_beam_bot_actual"],
                   "axial_ratio_est": p["axial_ratio_est"]})
    for k, d in dom.items():
        if k not in checks:
            continue
        x = checks[k]
        label = FIELD_BY_KEY[k].label if k in FIELD_BY_KEY else k
        if "values" in d:
            if isinstance(x, (int, float)):
                inside = any(abs(x - float(t)) < 1e-6 for t in d["values"])
            else:
                inside = str(x) in [str(t) for t in d["values"]]
            if not inside:
                c.warnings.append(f"{label} = {x}: not in the training population {d['values']}")
        elif x < d["min"] - 1e-9 or x > d["max"] + 1e-9:
            c.warnings.append(f"{label} = {x:.4g}: outside the training range [{d['min']:.4g}, {d['max']:.4g}]")
