"""Left panel: grouped inputs, example frames, validation banner, action buttons."""
from __future__ import annotations

import json
from pathlib import Path

import customtkinter as ctk

from core.schema import FIELDS, GROUPS
from core.validator import parse

TXT, MUTED, OK, WARN, ERR = "#1F2328", "#6B7280", "#1E6B3A", "#9A6700", "#B42318"


def fmt(x) -> str:
    if isinstance(x, (list, tuple)):
        return ", ".join(fmt(t) for t in x)
    if isinstance(x, float):
        s = f"{x:.4f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(x)


class InputPanel(ctk.CTkFrame):
    def __init__(self, master, examples_dir: Path, on_change=None, on_predict=None, on_batch=None):
        super().__init__(master, fg_color="#FFFFFF", corner_radius=10, border_width=1, border_color="#E0E0E0")
        self.on_change, self.on_predict, self.on_batch = on_change, on_predict, on_batch
        self.examples = {}
        for fp in sorted(Path(examples_dir).glob("*.json")):
            ex = json.loads(fp.read_text(encoding="utf-8"))
            self.examples[ex["label"]] = ex
        self.vars: dict[str, ctk.StringVar] = {}
        self._loading = False
        self.current_example: dict | None = None

        ctk.CTkLabel(self, text="Frame data", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=TXT).pack(anchor="w", padx=12, pady=(10, 2))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(row, text="Example:", text_color=MUTED).pack(side="left")
        self.example_var = ctk.StringVar(value="Custom")
        self.example_menu = ctk.CTkOptionMenu(row, variable=self.example_var, width=290, dynamic_resizing=False,
                                              values=["Custom"] + list(self.examples),
                                              command=self._apply_example)
        self.example_menu.pack(side="left", padx=6)

        body = ctk.CTkScrollableFrame(self, fg_color="#FFFFFF", width=380)
        body.pack(fill="both", expand=True, padx=4, pady=2)
        body.grid_columnconfigure(1, weight=1)
        r = 0
        for g in GROUPS:
            ctk.CTkLabel(body, text=g, font=ctk.CTkFont(size=12, weight="bold"),
                         text_color="#1D4E89").grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 2), padx=6)
            r += 1
            for f in [f for f in FIELDS if f.group == g]:
                var = ctk.StringVar()
                self.vars[f.key] = var
                lab = ctk.CTkLabel(body, text=f.label, text_color=TXT, anchor="w", font=ctk.CTkFont(size=12))
                lab.grid(row=r, column=0, sticky="w", padx=(8, 4), pady=1)
                if f.kind == "choice":
                    w = ctk.CTkOptionMenu(body, variable=var, values=list(f.choices), width=150, height=26)
                else:
                    w = ctk.CTkEntry(body, textvariable=var, width=150, height=26)
                w.grid(row=r, column=1, sticky="e", pady=1)
                ctk.CTkLabel(body, text=f.unit, text_color=MUTED, width=34, anchor="w",
                             font=ctk.CTkFont(size=11)).grid(row=r, column=2, sticky="w", padx=(4, 6))
                var.trace_add("write", lambda *_a: self._changed())
                r += 1

        self.banner = ctk.CTkLabel(self, text="", wraplength=380, justify="left", anchor="w",
                                   font=ctk.CTkFont(size=11))
        self.banner.pack(fill="x", padx=12, pady=(4, 2))
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=12, pady=(2, 10))
        self.predict_btn = ctk.CTkButton(btns, text="Predict performance level", height=34,
                                         command=lambda: self.on_predict and self.on_predict())
        self.predict_btn.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(btns, text="Batch…", width=80, height=34, fg_color="#E5E7EB", hover_color="#D1D5DB",
                      text_color=TXT, command=lambda: self.on_batch and self.on_batch()).pack(side="left", padx=(6, 0))

        if self.examples:
            first = next(iter(self.examples))
            self.example_var.set(first)
            self._apply_example(first)

    # ------------------------------------------------------------------
    def raw(self) -> dict:
        return {k: v.get() for k, v in self.vars.items()}

    def check(self):
        return parse(self.raw())

    def set_values(self, values: dict):
        self._loading = True
        try:
            for k, x in values.items():
                if k in self.vars:
                    self.vars[k].set(fmt(x))
        finally:
            self._loading = False
        self._changed(from_example=True)

    def _apply_example(self, label: str):
        if label not in self.examples:
            self.current_example = None
            return
        self.current_example = self.examples[label]
        self.set_values(self.current_example["values"])
        self.example_var.set(label)

    def _changed(self, from_example: bool = False):
        if self._loading:
            return
        if not from_example and self.example_var.get() != "Custom":
            self.example_var.set("Custom")
            self.current_example = None
        c = self.check()
        if c.errors:
            self.banner.configure(text="✖ " + "\n✖ ".join(c.errors[:4]), text_color=ERR)
            self.predict_btn.configure(state="disabled")
        elif c.warnings:
            self.banner.configure(text=f"⚠ {len(c.warnings)} warning(s) — prediction is an extrapolation:\n• "
                                  + "\n• ".join(c.warnings[:4]), text_color=WARN)
            self.predict_btn.configure(state="normal")
        else:
            self.banner.configure(text="✔ All inputs inside the simulated population.", text_color=OK)
            self.predict_btn.configure(state="normal")
        if self.on_change:
            self.on_change(c)
