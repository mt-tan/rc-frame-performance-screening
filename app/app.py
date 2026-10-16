"""Existing RC Frame Performance Screening — TBEC 2018 Section 15.

Desktop application of the study: the engineer enters the data of an existing RC frame
(geometry, sections, materials, as-built reinforcement, site); the closed-form A2 quantities
are computed with modules/features_m3.py and the RealMLP model (BASE + A2) predicts the
strain-based performance level (IO/LS/CP/C) for DD-1 and DD-2. An EBM model trained on the
same inputs provides the term-level explanation.

    .venv312/Scripts/python.exe app/app.py
"""
from __future__ import annotations

import logging
import sys
import traceback
import warnings
from pathlib import Path
from tkinter import messagebox

warnings.filterwarnings("ignore")
for _n in ("lightning", "lightning.pytorch", "pytorch_lightning", "lightning_fabric"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import customtkinter as ctk  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.paths import EXAMPLES_DIR, MODELS_DIR  # noqa: E402
from ui.frame_canvas import render_frame  # noqa: E402
from ui.input_panel import InputPanel  # noqa: E402
from ui.result_panel import ResultPanel  # noqa: E402

APP_TITLE = "Existing RC Frame Performance Screening — TBEC 2018 Section 15"
APP_VERSION = "1.0.0"


def model_info() -> dict:
    import json
    fp = MODELS_DIR / "reliability_ext.json"
    if not fp.exists():
        return {}
    r = json.loads(fp.read_text(encoding="utf-8"))
    return {"metrics": r["metrics"], "agreement": r["agreement_with_stored_s42"],
            "n_rows": sum(b["n"] for b in r["confidence_bins"]),
            "low_conf": f"{r['low_conf_acc']:.0%}"}


class ScreeningApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title(f"{APP_TITLE}  (v{APP_VERSION})")
        self.geometry("1560x920")
        self.minsize(1300, 780)
        self.configure(fg_color="#F4F5F7")
        self.predictor = None
        self.last_check = None

        body = ctk.CTkFrame(self, fg_color="#F4F5F7")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=0, minsize=430)
        body.grid_columnconfigure(1, weight=0, minsize=330)
        body.grid_columnconfigure(2, weight=1)
        body.grid_rowconfigure(0, weight=1)

        mid = ctk.CTkFrame(body, fg_color="#FFFFFF", corner_radius=10, border_width=1, border_color="#E0E0E0")
        mid.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        ctk.CTkLabel(mid, text="Frame", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#1F2328").pack(anchor="w", padx=12, pady=(10, 0))
        self.frame_fig = Figure(figsize=(3.3, 7.6), dpi=100 * ctk.ScalingTracker.get_widget_scaling(self),
                                facecolor="#FFFFFF")
        self.frame_canvas = FigureCanvasTkAgg(self.frame_fig, master=mid)
        self.frame_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(0, 8))

        self.result_panel = ResultPanel(body, info=model_info())
        self.result_panel.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)

        self.input_panel = InputPanel(body, examples_dir=EXAMPLES_DIR, on_change=self._on_change,
                                      on_predict=self.predict, on_batch=self._on_batch)
        self.input_panel.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        self.status = ctk.CTkLabel(self, text="Loading models…", anchor="w", fg_color="#FFFFFF",
                                   text_color="#666666", font=ctk.CTkFont(size=11), height=24)
        self.status.pack(fill="x", side="bottom")
        self.after(100, self._load_models)

    # ------------------------------------------------------------------
    def _load_models(self):
        try:
            from core.predictor import Predictor
            self.predictor = Predictor()
            self.status.configure(text="Ready — RealMLP (BASE+A2, seed 42) and EBM loaded from app/assets/models.")
            self.predict()
        except Exception as exc:
            self.predictor = None
            self.status.configure(text=f"Model loading failed: {exc}")
            traceback.print_exc()

    def _on_change(self, check):
        self.last_check = check
        render_frame(self.frame_fig, check.params if check.ok else None)
        self.frame_canvas.draw_idle()
        self.result_panel.clear("Inputs changed — press Predict." if check.ok else "Invalid inputs.")

    def predict(self):
        if self.predictor is None:
            return
        c = self.input_panel.check()
        if not c.ok:
            return
        try:
            res = self.predictor.predict_params(c.params)
        except Exception as exc:
            messagebox.showerror("Prediction", str(exc))
            return
        ex = self.input_panel.current_example
        self.result_panel.show(res, reference=ex["reference_pushover"] if ex else None, warnings=c.warnings)
        src = f"example {ex['name']} (external test frame)" if ex else "user input"
        self.status.configure(text=f"Predicted: {src}. DD-1 {res.levels['DD1'].pred}, DD-2 {res.levels['DD2'].pred}.")

    def _on_batch(self):
        if self.predictor is None:
            return
        from ui.batch_window import BatchWindow
        BatchWindow(self, self.predictor)


def main():
    app = ScreeningApp()
    app.mainloop()


if __name__ == "__main__":
    main()
