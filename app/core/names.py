"""Single name dictionary for the 60 model inputs (paper, EBM explanation, A2 table, README).

code -> (block, readable name, unit, display scale, definition). Values are stored in generator
units (m, kN, kPa-derived MPa, g, strain); ``scale`` converts them to the display unit.
Block: "building" = building and site data (BASE), "code" = hand-calculated code quantities (A2).
"""
from __future__ import annotations

import textwrap

NONE = "–"
WRAP = 30                      # characters per label line in the EBM chart

FEATURES: dict[str, tuple[str, str, str, float, str]] = {
    # ---- building and site data (BASE)
    "level": ("building", "Earthquake level", "0/1", 1, "0 = DD-1, 1 = DD-2"),
    "N_story": ("building", "Number of stories", NONE, 1, ""),
    "N_bay": ("building", "Number of bays", NONE, 1, ""),
    "L_mean": ("building", "Mean bay length", "m", 1, "axis to axis"),
    "L_max": ("building", "Maximum bay length", "m", 1, "axis to axis"),
    "L_min": ("building", "Minimum bay length", "m", 1, "axis to axis"),
    "W_total": ("building", "Frame length", "m", 1, "sum of the bay lengths"),
    "h1": ("building", "First-story height", "m", 1, ""),
    "htyp": ("building", "Typical story height", "m", 1, "stories 2 to N"),
    "Htot": ("building", "Building height", "m", 1, "h1 + (N − 1)·htyp"),
    "col_B": ("building", "Column width b", "cm", 100, "out-of-plane dimension"),
    "col_H": ("building", "Column depth h", "cm", 100, "in-plane dimension"),
    "col_aspect": ("building", "Column aspect ratio h/b", NONE, 1, ""),
    "beam_b": ("building", "Beam width", "cm", 100, ""),
    "beam_h": ("building", "Beam depth", "cm", 100, ""),
    "cover": ("building", "Clear cover", "cm", 100, ""),
    "fc_MPa": ("building", "Concrete strength fc", "MPa", 1, "existing strength"),
    "fy_MPa": ("building", "Steel yield strength fy", "MPa", 1, "S220 or S420"),
    "G": ("building", "Dead load G", "kN/m", 1, "distributed on the beams"),
    "Q": ("building", "Live load Q", "kN/m", 1, "distributed on the beams"),
    "kappa_col": ("building", "As-built reinforcement factor κ, columns", NONE, 1,
                  "as-built / design steel area (TBEC 15.2.5.2)"),
    "kappa_beam": ("building", "As-built reinforcement factor κ, beams", NONE, 1,
                   "as-built / design steel area (TBEC 15.2.5.2)"),
    "rho_col_actual": ("building", "Column steel ratio", "%", 100, "as-built: κ × design ratio, As/(b·h)"),
    "rho_beam_top_actual": ("building", "Beam top steel ratio", "%", 100, "as-built: κ × design ratio, As/(b·h)"),
    "rho_beam_bot_actual": ("building", "Beam bottom steel ratio", "%", 100, "as-built: κ × design ratio, As/(b·h)"),
    "phi_long_col_mm": ("building", "Column bar diameter", "mm", 1, ""),
    "phi_long_beam_mm": ("building", "Beam bar diameter", "mm", 1, ""),
    "col_phi_etr": ("building", "Column tie diameter", "mm", 1000, ""),
    "col_s_etr": ("building", "Column tie spacing", "cm", 100, ""),
    "col_n_etr_x": ("building", "Column tie legs ∥ b", NONE, 1, ""),
    "col_n_etr_y": ("building", "Column tie legs ∥ h", NONE, 1, "legs resisting in-plane shear"),
    "beam_phi_etr": ("building", "Beam stirrup diameter", "mm", 1000, ""),
    "beam_s_etr": ("building", "Beam stirrup spacing", "cm", 100, ""),
    "hook_deg": ("building", "Hook angle", "°", 1, "90° or 135°"),
    "plain_bars": ("building", "Plain bars", "0/1", 1, "1 = S220 plain bars"),
    "soil": ("building", "Soil class", "0–4", 1, "ZA … ZE"),
    "SDS": ("building", "Short-period design spectral acceleration S_DS", "g", 1, "short-period coefficient of the site, soil and level"),
    "SD1": ("building", "1-s design spectral acceleration S_D1", "g", 1, "1-s coefficient of the site, soil and level"),
    # ---- hand-calculated code quantities (A2)
    "TpA": ("code", "Empirical period TpA", "s", 1, "0.1·H^0.75 (TBEC 4.7.3.4)"),
    "Gamma_lin": ("code", "Participation factor Γ", NONE, 1, "linear mode shape, equal story masses"),
    "Vmech_W": ("code", "Mechanism shear / weight", NONE, 1, "min(column, beam mechanism shear) / W"),
    "Vcolmech_W": ("code", "Column-mechanism shear / weight", NONE, 1, "first-story columns: Σ2Mr/(h1 − hb), / W"),
    "Vbeammech_W": ("code", "Beam-mechanism shear / weight", NONE, 1,
                    "[Σ beams (Mr+ + Mr−) + Σ base columns Mr] / H_eff, / W"),
    "mech_beam_over_col": ("code", "Beam / column mechanism shear", NONE, 1, ""),
    "omega_we_col": ("code", "Confinement index ωwe, columns", NONE, 1, "TBEC Eq. 5.4, 90° hooks at 30 %"),
    "omega_we_beam": ("code", "Confinement index ωwe, beams", NONE, 1, "TBEC Eq. 5.4, 90° hooks at 30 %"),
    "eps_cGO_col": ("code", "CP concrete strain limit, columns", NONE, 1, "TBEC 5.8.1"),
    "eps_cGO_beam": ("code", "CP concrete strain limit, beams", NONE, 1, "TBEC 5.8.1"),
    "eps_sGO": ("code", "CP steel strain limit", NONE, 1, "TBEC 5.8.1"),
    "VeVr_col": ("code", "Column shear demand/capacity", NONE, 1, "Ve = 2Mr/ln, Vr per TS 500 (interior column)"),
    "VeVr_beam": ("code", "Beam shear demand/capacity", NONE, 1, "longest span, gravity + capacity shear"),
    "rv_col": ("code", "Column shear stress ratio", NONE, 1, "Ve/(bw·d·fctm)"),
    "rv_beam": ("code", "Beam shear stress ratio", NONE, 1, "Ve/(bw·d·fctm)"),
    "fv_col": ("code", "Column shear reduction factor", NONE, 1, "factor on strain limits (TBEC 15.7.1.4)"),
    "axial_ratio": ("code", "Column axial-load ratio", NONE, 1, "N/(Ac·fc), first-story interior column"),
    "scwb": ("code", "Column-to-beam moment capacity ratio", NONE, 1, "ΣMc/ΣMb at the critical interior joint"),
    "Sae": ("code", "Spectral acceleration Sae", "g", 1, "design spectrum at TpA, per level"),
    "Sde": ("code", "Spectral displacement Sde", "cm", 100, "design spectrum at TpA, per level"),
    "R": ("code", "Strength-to-demand ratio R", NONE, 1, "Vmech/(W·Sae), per level"),
    "theta": ("code", "Estimated roof drift ratio θ (hand calculation)", "%", 100, "Γ·CR·Sde/H, per level"),
}


def code_of(column: str) -> str:
    """Model column (B__fc_MPa, A2__R) or A2 dict key (A2_R_DD1) -> dictionary code."""
    c = column.split("__", 1)[-1]
    if c.startswith("A2_"):
        c = c[3:]
    for sfx in ("_DD1", "_DD2"):
        if c.endswith(sfx):
            c = c[: -len(sfx)]
    return c


def name(column: str) -> str:
    return FEATURES[code_of(column)][1]


def unit(column: str) -> str:
    return FEATURES[code_of(column)][2]


def fmt_value(column: str, x: float) -> str:
    _b, _n, u, scale, _d = FEATURES[code_of(column)]
    s = f"{x * scale:.3g}"
    if u in (NONE, "0/1", "0–4"):
        return s
    return s + ("°" if u == "°" else f" {u}")


def label(column: str, x: float, width: int | None = None) -> str:
    """'Name = value unit'; with ``width`` the name is wrapped at word boundaries and the
    '= value unit' part is never split."""
    tail = f"= {fmt_value(column, x)}"
    if width is None:
        return f"{name(column)} {tail}"
    lines = textwrap.wrap(name(column), width)
    if len(lines[-1]) + 1 + len(tail) <= width + 8:
        lines[-1] += " " + tail
    else:
        lines.append(tail)
    return "\n".join(lines)


def term_label(term: str, x) -> str:
    """EBM term ('B__rv_beam' or 'A2__rv_beam & A2__scwb') with the row's values -> readable label;
    interactions as 'A × B', one factor per line; long names wrapped at WRAP characters."""
    parts = [t.strip() for t in term.split("&")]
    return " ×\n".join(label(p, x[p], WRAP) for p in parts)


def markdown_table() -> str:
    rows = ["| Code | Block | Name | Unit | Definition |", "|---|---|---|---|---|"]
    for c, (b, n, u, _s, d) in FEATURES.items():
        rows.append(f"| `{c}` | {b} | {n} | {u} | {d} |")
    return "\n".join(rows)
