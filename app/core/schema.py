"""User-facing input schema (engineering units) and the domain of the training population.

Every field is something an engineer collects for an existing frame (drawings + on-site survey).
Conversions to the generator's units (m, kN, kPa) are done in :mod:`core.params`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .paths import ASSETS_DIR

SITES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
SOILS = ["ZA", "ZB", "ZC", "ZD", "ZE"]
GRADES = ["S220", "S420"]
HOOKS = ["135", "90"]


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    unit: str = ""
    kind: str = "float"            # float | int | choice | list
    choices: tuple = ()
    group: str = ""
    help: str = ""
    extra: dict = field(default_factory=dict)


GROUPS = ["Site", "Geometry", "Sections", "Materials", "Longitudinal bars",
          "Transverse reinforcement", "Gravity loads"]

FIELDS: list[Field] = [
    Field("site", "Site", kind="choice", choices=tuple(SITES), group="Site",
          help="AFAD TDTH site of the study (Kahramanmaraş region)"),
    Field("soil_class", "Soil class", kind="choice", choices=tuple(SOILS), group="Site"),
    Field("N_story", "Number of stories", kind="int", group="Geometry"),
    Field("h1", "First-story height", "m", group="Geometry"),
    Field("htyp", "Typical story height", "m", group="Geometry"),
    Field("bay_lengths", "Bay lengths", "m", kind="list", group="Geometry",
          help="comma-separated, axis to axis"),
    Field("col_B", "Column width b (out-of-plane)", "cm", group="Sections"),
    Field("col_H", "Column depth h (in-plane)", "cm", group="Sections"),
    Field("beam_b", "Beam width", "cm", group="Sections"),
    Field("beam_h", "Beam depth", "cm", group="Sections"),
    Field("cover", "Clear cover", "cm", group="Sections"),
    Field("fc_MPa", "Existing concrete strength fc", "MPa", group="Materials"),
    Field("steel_grade", "Steel grade", kind="choice", choices=tuple(GRADES), group="Materials",
          help="S220 = plain bars, S420 = deformed bars"),
    Field("phi_long_col_mm", "Column bar diameter", "mm", group="Longitudinal bars"),
    Field("col_n_long_x", "Column bars per b-face", kind="int", group="Longitudinal bars",
          help="incl. corners"),
    Field("col_n_long_y", "Column bars per h-face", kind="int", group="Longitudinal bars",
          help="incl. corners"),
    Field("phi_long_beam_mm", "Beam bar diameter", "mm", group="Longitudinal bars"),
    Field("beam_n_long_top", "Beam top bars", kind="int", group="Longitudinal bars"),
    Field("beam_n_long_bot", "Beam bottom bars", kind="int", group="Longitudinal bars"),
    Field("kappa_col", "As-built reinforcement factor κ, columns", group="Longitudinal bars",
          help="as-built / design steel area (TBEC 15.2.5.2)"),
    Field("kappa_beam", "As-built reinforcement factor κ, beams", group="Longitudinal bars"),
    Field("col_phi_etr_mm", "Column tie diameter", "mm", group="Transverse reinforcement"),
    Field("col_s_etr_cm", "Column tie spacing", "cm", group="Transverse reinforcement"),
    Field("col_n_etr_y", "Column tie legs ∥ h", kind="int", group="Transverse reinforcement",
          help="legs resisting in-plane shear"),
    Field("col_n_etr_x", "Column tie legs ∥ b", kind="int", group="Transverse reinforcement"),
    Field("beam_phi_etr_mm", "Beam stirrup diameter", "mm", group="Transverse reinforcement"),
    Field("beam_s_etr_cm", "Beam stirrup spacing", "cm", group="Transverse reinforcement"),
    Field("hook_deg", "Hook angle", "°", kind="choice", choices=tuple(HOOKS),
          group="Transverse reinforcement", help="90° hooks: 30 % of ρsh effective (TBEC 15.7.1.3)"),
    Field("G", "Dead load on beams G", "kN/m", group="Gravity loads"),
    Field("Q", "Live load on beams Q", "kN/m", group="Gravity loads"),
]
FIELD_BY_KEY = {f.key: f for f in FIELDS}


def load_domain() -> dict:
    """Training-population ranges (written by tools/build_assets.py from the training manifests)."""
    fp = ASSETS_DIR / "domain.json"
    return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
