"""User inputs (engineering units) -> generator parameter dict (manifest conventions).

The derived quantities follow the generator exactly (modules/lhs_sampling.py, make_manifest.py):
  * column ρ = (2·nx + 2·ny − 4)·A_bar/(b·h); beam ρ = n·A_bar/(b·h); as-built ρ = κ·ρ
  * axial-load ratio: tbdy_filters.estimate_column_axial_load; ΣMc/ΣMb: tbdy_filters.scwb_ratio
  * steel properties: config.REBAR_GRADES (S220 plain, S420 deformed)
"""
from __future__ import annotations

import json
import math

from . import paths  # noqa: F401  (puts modules/ on sys.path)
import config
import tbdy_filters

CM, MM = 0.01, 0.001


def build_params(v: dict, name: str = "user") -> dict:
    """``v``: typed values keyed as in :data:`core.schema.FIELDS`."""
    bays = [float(x) for x in v["bay_lengths"]]
    nb = len(bays)
    grade = config.REBAR_GRADES[v["steel_grade"]]
    fc = float(v["fc_MPa"]) * config.MPa
    fy = grade["fy_MPa"] * config.MPa
    B, H = float(v["col_B"]) * CM, float(v["col_H"]) * CM
    bb, bh = float(v["beam_b"]) * CM, float(v["beam_h"]) * CM
    phc, phb = float(v["phi_long_col_mm"]) * MM, float(v["phi_long_beam_mm"]) * MM
    nx, ny = int(v["col_n_long_x"]), int(v["col_n_long_y"])
    nt, nbo = int(v["beam_n_long_top"]), int(v["beam_n_long_bot"])
    kc, kb = float(v["kappa_col"]), float(v["kappa_beam"])
    rho_col = (2 * nx + 2 * ny - 4) * (math.pi * phc ** 2 / 4.0) / (B * H)
    Ab = math.pi * phb ** 2 / 4.0
    rho_top, rho_bot = nt * Ab / (bb * bh), nbo * Ab / (bb * bh)
    L_mean = sum(bays) / nb
    p = {
        "name": name, "site": v["site"], "soil_class": v["soil_class"],
        "N_story": int(v["N_story"]), "N_bay": nb,
        "bay_lengths": bays, "Lbay": L_mean, "L_mean": L_mean,
        "L_max": max(bays), "L_min": min(bays), "W_total": sum(bays),
        "h1": float(v["h1"]), "htyp": float(v["htyp"]),
        "col_B": B, "col_H": H, "col_aspect": H / B,
        "beam_b": bb, "beam_h": bh, "cover": float(v["cover"]) * CM,
        "fc": fc, "fck": fc, "fc_MPa": float(v["fc_MPa"]),
        "G": float(v["G"]), "Q": float(v["Q"]),
        "steel_grade": v["steel_grade"], "fy": fy, "fy_MPa": grade["fy_MPa"],
        "eps_sy": grade["eps_sy"], "eps_su": grade["eps_su"], "fsu_MPa": grade["fsu_MPa"],
        "plain_bars": bool(grade["plain"]),
        "hook_deg": int(v["hook_deg"]), "kappa_col": kc, "kappa_beam": kb,
        "rho_col": rho_col, "rho_col_actual": rho_col * kc,
        "col_phi_long": phc, "col_n_long_x": nx, "col_n_long_y": ny,
        "col_n_long_total": 2 * nx + 2 * ny - 4, "phi_long_col_mm": phc / MM,
        "rho_beam_top": rho_top, "rho_beam_bot": rho_bot,
        "rho_beam_top_actual": rho_top * kb, "rho_beam_bot_actual": rho_bot * kb,
        "beam_phi_long": phb, "beam_n_long_top": nt, "beam_n_long_bot": nbo,
        "phi_long_beam_mm": phb / MM,
        "col_phi_etr": float(v["col_phi_etr_mm"]) * MM, "col_s_etr": float(v["col_s_etr_cm"]) * CM,
        "col_n_etr_x": int(v["col_n_etr_x"]), "col_n_etr_y": int(v["col_n_etr_y"]),
        "beam_phi_etr": float(v["beam_phi_etr_mm"]) * MM, "beam_s_etr": float(v["beam_s_etr_cm"]) * CM,
        "beam_n_etr_x": 2, "beam_n_etr_y": 2,        # closed 2-leg stirrups (generator convention)
    }
    p["Htot"] = p["h1"] + (p["N_story"] - 1) * p["htyp"]
    p["axial_ratio_est"] = tbdy_filters.estimate_column_axial_load(p)[2]
    p["scwb_ratio"] = tbdy_filters.scwb_ratio(p)
    return p


def values_from_manifest(r: dict) -> dict:
    """Manifest row (generator units) -> user input values (inverse of :func:`build_params`)."""
    bays = r["bay_lengths"]
    if isinstance(bays, str):
        bays = json.loads(bays)
    return {
        "site": r["site"], "soil_class": r["soil_class"], "N_story": int(r["N_story"]),
        "h1": float(r["h1"]), "htyp": float(r["htyp"]), "bay_lengths": [float(x) for x in bays],
        "col_B": r["col_B"] / CM, "col_H": r["col_H"] / CM, "beam_b": r["beam_b"] / CM,
        "beam_h": r["beam_h"] / CM, "cover": r["cover"] / CM, "fc_MPa": float(r["fc_MPa"]),
        "steel_grade": r["steel_grade"], "phi_long_col_mm": float(r["phi_long_col_mm"]),
        "col_n_long_x": int(r["col_n_long_x"]), "col_n_long_y": int(r["col_n_long_y"]),
        "phi_long_beam_mm": float(r["phi_long_beam_mm"]),
        "beam_n_long_top": int(r["beam_n_long_top"]), "beam_n_long_bot": int(r["beam_n_long_bot"]),
        "kappa_col": float(r["kappa_col"]), "kappa_beam": float(r["kappa_beam"]),
        "col_phi_etr_mm": r["col_phi_etr"] / MM, "col_s_etr_cm": r["col_s_etr"] / CM,
        "col_n_etr_x": int(r["col_n_etr_x"]), "col_n_etr_y": int(r["col_n_etr_y"]),
        "beam_phi_etr_mm": r["beam_phi_etr"] / MM, "beam_s_etr_cm": r["beam_s_etr"] / CM,
        "hook_deg": str(int(r["hook_deg"])), "G": float(r["G"]), "Q": float(r["Q"]),
    }


def manifest_params(r: dict) -> dict:
    """Manifest row as stored -> parameter dict used by the training pipeline (ml/data_m3.load_wave)."""
    p = dict(r)
    if isinstance(p.get("bay_lengths"), str):
        p["bay_lengths"] = json.loads(p["bay_lengths"])
    return p
