"""Batch window: evaluate a CSV/XLSX table of frames and export the results."""
from __future__ import annotations

from tkinter import filedialog, messagebox

import customtkinter as ctk

from core.batch import evaluate, read_table
from core.paths import EXAMPLES_DIR


class BatchWindow(ctk.CTkToplevel):
    def __init__(self, master, predictor):
        super().__init__(master)
        self.title("Batch evaluation")
        self.geometry("900x520")
        self.predictor = predictor
        self.out = None
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=10, pady=8)
        ctk.CTkButton(bar, text="Open table (CSV/XLSX)…", command=self._open).pack(side="left")
        self.save_btn = ctk.CTkButton(bar, text="Save results…", command=self._save, state="disabled")
        self.save_btn.pack(side="left", padx=6)
        ctk.CTkLabel(bar, text=f"Template: {EXAMPLES_DIR / 'batch_template.csv'}", text_color="#6B7280",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=8)
        self.text = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        self.text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.text.insert("0.0", "One frame per row; columns are the input keys of the template "
                                "(bay_lengths as a comma-separated list in quotes).")

    def run(self, path):
        df = read_table(path)
        self.out = evaluate(self.predictor, df)
        cols = ["id", "status", "DD1_level", "DD1_confidence", "DD1_low_confidence",
                "DD2_level", "DD2_confidence", "DD2_low_confidence", "extrapolation"]
        show = self.out[[c for c in cols if c in self.out.columns]]
        n_ok = int((self.out["status"] == "ok").sum())
        self.text.delete("0.0", "end")
        self.text.insert("0.0", f"{path}\n{n_ok}/{len(self.out)} rows evaluated\n\n" + show.to_string(index=False))
        self.save_btn.configure(state="normal")
        return self.out

    def _open(self):
        fp = filedialog.askopenfilename(parent=self, filetypes=[("Tables", "*.csv *.xlsx"), ("All", "*.*")],
                                        initialdir=str(EXAMPLES_DIR))
        if not fp:
            return
        try:
            self.run(fp)
        except Exception as exc:
            messagebox.showerror("Batch evaluation", str(exc), parent=self)

    def _save(self):
        fp = filedialog.asksaveasfilename(parent=self, defaultextension=".csv",
                                          filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")])
        if not fp:
            return
        if fp.lower().endswith(".xlsx"):
            self.out.to_excel(fp, index=False)
        else:
            self.out.to_csv(fp, index=False)
