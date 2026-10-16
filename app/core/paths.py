"""Project paths. The generator package (``modules/``) is imported read-only for the
closed-form (A2) quantities, exactly as in the training pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = APP_DIR.parent
MODULES_DIR = ROOT_DIR / "modules"
ASSETS_DIR = APP_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
EXAMPLES_DIR = ASSETS_DIR / "examples"
REPORTS_DIR = APP_DIR / "reports"

# Stored results of the study (read-only; used by the parity tests and by tools/)
STORED_PRED_DIR = ROOT_DIR / "ml" / "out" / "final"
STORED_EBM_PKL = STORED_PRED_DIR / "models" / "main_BASE+A2_EBM_s42.pkl"
ROWS_DIR = ROOT_DIR / "ml" / "cache"
RUNS_DIR = ROOT_DIR / "runs"

if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))
