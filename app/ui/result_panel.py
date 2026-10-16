"""Right panel: DD-1/DD-2 verdict cards, class probabilities, EBM explanation, A2 table, model notes."""
from __future__ import annotations

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import names
from core.features import LEVEL_LABEL, LEVELS
from core.predictor import CLS, LEVEL_NAMES

# one color scale for the four levels (cards, probability bars), best to worst
LEVEL_COLORS = {"IO": "#2E7D32", "LS": "#C99A06", "CP": "#E06C00", "C": "#C62828"}
ORDER = CLS
TXT, MUTED, WARN, OK = "#1F2328", "#6B7280", "#9A6700", "#1E6B3A"

A2_LEVEL = ["Sae", "Sde", "R", "theta"]                  # level-specific code quantities
A2_COMMON = [c for c, v in names.FEATURES.items() if v[0] == "code" and c not in A2_LEVEL]


class ResultPanel(ctk.CTkFrame):
    def __init__(self, master, info: dict | None = None):
        super().__init__(master, fg_color="#FFFFFF", corner_radius=10, border_width=1, border_color="#E0E0E0")
        self.info = info or {}
        self.result = None
        self.reference = None
        dpi = 100 * ctk.ScalingTracker.get_widget_scaling(self)   # matplotlib text scales with the widgets
        ctk.CTkLabel(self, text="Predicted TBEC 2018 performance level (strain-based, Section 15)",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color=TXT).pack(anchor="w", padx=12, pady=(10, 0))
        self.subtitle = ctk.CTkLabel(
            self, text="Main model: RealMLP on building data and hand-calculated code quantities "
                       "(no structural analysis)", text_color=MUTED, font=ctk.CTkFont(size=12),
            justify="left", anchor="w", wraplength=600)
        self.subtitle.pack(anchor="w", padx=12)
        self.bind("<Configure>", lambda e: self.subtitle.configure(
            wraplength=max(e.width / ctk.ScalingTracker.get_widget_scaling(self) - 28, 200)))

        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(fill="x", padx=8, pady=(6, 0))
        cards.grid_columnconfigure((0, 1), weight=1, uniform="c")
        self.cards = {}
        for j, lev in enumerate(LEVELS):
            self.cards[lev] = _Card(cards, LEVEL_LABEL[lev])
            self.cards[lev].grid(row=0, column=j, sticky="nsew", padx=4)

        self.prob_fig = Figure(figsize=(4.0, 1.5), dpi=dpi, facecolor="#FFFFFF")
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=self)
        self.prob_canvas.get_tk_widget().pack(fill="x", padx=10, pady=(4, 0))

        self.tabs = ctk.CTkTabview(self, height=300, fg_color="#FAFAFA")
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        t_ebm = self.tabs.add("Explanation (EBM)")
        t_a2 = self.tabs.add("Code quantities")
        t_mod = self.tabs.add("Model & limits")

        ctl = ctk.CTkFrame(t_ebm, fg_color="transparent")
        ctl.pack(fill="x")
        self.ebm_level = ctk.StringVar(value=LEVEL_LABEL[LEVELS[0]])
        self.ebm_class = ctk.StringVar(value="predicted")
        ctk.CTkSegmentedButton(ctl, values=[LEVEL_LABEL[x] for x in LEVELS], variable=self.ebm_level,
                               command=lambda _v: self._draw_ebm()).pack(side="left")
        ctk.CTkLabel(ctl, text="  class:", text_color=MUTED).pack(side="left")
        ctk.CTkOptionMenu(ctl, values=["predicted"] + ORDER, variable=self.ebm_class, width=110,
                          command=lambda _v: self._draw_ebm()).pack(side="left", padx=4)
        self.ebm_note = ctk.CTkLabel(t_ebm, text="", text_color=MUTED, justify="left", anchor="w",
                                     wraplength=400, font=ctk.CTkFont(size=12))
        self.ebm_note.pack(fill="x", pady=(2, 0))
        t_ebm.bind("<Configure>", lambda e: self.ebm_note.configure(
            wraplength=max(e.width / ctk.ScalingTracker.get_widget_scaling(self) - 16, 200)))
        self.ebm_fig = Figure(figsize=(4.0, 2.6), dpi=dpi, facecolor="#FAFAFA")
        self.ebm_fig.set_layout_engine("tight", pad=0.8)          # re-applied on every draw (resizes)
        self.ebm_canvas = FigureCanvasTkAgg(self.ebm_fig, master=t_ebm)
        self.ebm_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.a2_text = ctk.CTkTextbox(t_a2, font=ctk.CTkFont(family="Consolas", size=12))
        self.a2_text.pack(fill="both", expand=True)
        self.model_text = ctk.CTkTextbox(t_mod, font=ctk.CTkFont(size=12), wrap="word")
        self.model_text.pack(fill="both", expand=True)
        self._write_model_text()
        self._draw_probs()

    # ------------------------------------------------------------------
    def clear(self, message: str = ""):
        self.result = None
        for c in self.cards.values():
            c.clear(message)
        self._draw_probs()
        self.ebm_fig.clear()
        self.ebm_canvas.draw()
        self.a2_text.delete("0.0", "end")

    def show(self, result, reference: dict | None = None, warnings: list | None = None):
        self.result, self.reference = result, reference
        for lev in LEVELS:
            self.cards[lev].show(result.levels[lev], (reference or {}).get(lev), bool(warnings))
        self._draw_probs()
        self._draw_ebm()
        self._write_a2()

    # ------------------------------------------------------------------
    def _draw_probs(self):
        fig = self.prob_fig
        fig.clear()
        ax = fig.add_subplot(111)
        y = np.arange(len(LEVELS))
        if self.result is not None:
            left = np.zeros(len(LEVELS))
            for c in ORDER:
                w = np.array([self.result.levels[lev].probs[c] for lev in LEVELS])
                ax.barh(y, w, left=left, color=LEVEL_COLORS[c], height=0.55, label=c)
                for i in range(len(LEVELS)):
                    if w[i] >= 0.07:
                        ax.text(left[i] + w[i] / 2, y[i], f"{c} {w[i]:.0%}", ha="center", va="center",
                                fontsize=9, color="white", fontweight="bold")
                left += w
        ax.set_yticks(y, [LEVEL_LABEL[x] for x in LEVELS], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
        ax.tick_params(axis="x", labelsize=9)
        ax.set_title("Class probabilities (RealMLP)", fontsize=10, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.subplots_adjust(left=0.1, right=0.96, top=0.8, bottom=0.2)
        self.prob_canvas.draw()

    def _draw_ebm(self):
        fig = self.ebm_fig
        fig.clear()
        if self.result is None:
            self.ebm_canvas.draw()
            return
        lev = {LEVEL_LABEL[x]: x for x in LEVELS}[self.ebm_level.get()]
        r = self.result.levels[lev]
        cls = r.ebm_pred if self.ebm_class.get() == "predicted" else self.ebm_class.get()
        t = r.ebm_terms[["term", "value", cls]].copy()
        t["abs"] = t[cls].abs()
        t = t.sort_values("abs", ascending=False).head(10).iloc[::-1]
        ax = fig.add_subplot(111)
        x = self.result.rows.loc[lev]
        labels = [names.term_label(term, x) for term in t["term"]]
        rows = np.array([1.0 + 0.7 * lab.count("\n") for lab in labels])     # two-line labels get more room
        ys = np.cumsum(rows) - rows / 2
        ax.barh(ys, t[cls], color=["#1D4E89" if s > 0 else "#B42318" for s in t[cls]], height=0.6)
        ax.set_yticks(ys, labels, fontsize=9)
        ax.set_ylim(0, rows.sum())
        ax.axvline(0, color="#374151", lw=0.8)
        ax.set_xlabel(f"contribution to the {cls} logit (EBM)", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        self.ebm_canvas.draw()
        agree = "agrees" if r.ebm_pred == r.pred else "DIFFERS"
        probs = ", ".join(f"{c}\u00a0{r.ebm_probs[c]:.0%}" for c in ORDER)      # no line break inside a pair
        self.ebm_note.configure(
            text=f"EBM (interpretable glass-box model, same 60 inputs) predicts {r.ebm_pred} for "
                 f"{LEVEL_LABEL[lev]} ({probs}); it {agree} with the RealMLP verdict. The bars explain the EBM, "
                 f"not the RealMLP: they show which inputs drive the {cls} score, 10 largest terms.")

    def _write_a2(self):
        a2 = self.result.a2

        def v(code, key):
            return a2[key] * names.FEATURES[code][3]

        lines = [f"{'Hand-calculated code quantity (no analysis)':50s}{'value':>12s}  unit", "-" * 70]
        for c in A2_COMMON:
            lines.append(f"{names.name(c):50s}{v(c, 'A2_' + c):12.4g}  {names.unit(c)}")
        lines.append("")
        lines.append(f"{'Level-specific':50s}{'DD-1':>12s}{'DD-2':>12s}")
        for c in A2_LEVEL:
            lines.append(f"{names.name(c):50s}{v(c, f'A2_{c}_DD1'):12.4g}{v(c, f'A2_{c}_DD2'):12.4g}  {names.unit(c)}")
        p = self.result.params
        lines += ["", "Derived inputs", "-" * 70,
                  f"{'As-built ρ column (κ applied)':50s}{p['rho_col_actual']:12.4%}",
                  f"{'As-built ρ beam top / bottom':50s}{p['rho_beam_top_actual']:12.4%}{p['rho_beam_bot_actual']:12.4%}",
                  f"{'Axial ratio estimate (generator filter)':50s}{p['axial_ratio_est']:12.4g}"]
        self.a2_text.delete("0.0", "end")
        self.a2_text.insert("0.0", "\n".join(lines))

    def _write_model_text(self):
        i = self.info
        m = i.get("metrics", {})
        txt = ("Purpose: rapid screening of existing RC frames. The verdict approximates the TBEC 2018 "
               "strain-based performance level obtained from nonlinear pushover assessment (Section 15) for "
               "DD-1 and DD-2; it does not replace that assessment.\n\n")
        if m:
            txt += (f"External test ({i.get('n_rows', '?')} frame-level rows, unseen LHS frames): accuracy "
                    f"{m['acc']:.1%}, macro-F1 {m['f1_macro']:.3f}. Unsafe-direction errors (predicted level better "
                    f"than the pushover level): {m['optimistic']:.1%} of rows; conservative errors "
                    f"{m['conservative']:.1%}.\n\n")
        if i.get("agreement") is not None:
            txt += (f"The app model is a retraining of the published RealMLP (seed 42, same data and settings); "
                    f"its class predictions agree with the published model on {i['agreement']:.1%} of the external "
                    f"test rows.\n\n")
        txt += ("Low-confidence flag: raised when predictions of similar confidence were correct less than "
                f"{i.get('low_conf', '90 %')} of the time on the external test. A low-confidence or extrapolated "
                "result calls for a detailed (pushover) assessment.\n\n"
                "Scope: planar RC moment frames, 2–6 stories, 2–6 bays, rectangular columns, S220/S420, 90°/135° "
                "hooks, eight Kahramanmaraş-area sites (AFAD TDTH), soil classes ZA–ZE. Inputs outside the "
                "simulated population are flagged.")
        self.model_text.delete("0.0", "end")
        self.model_text.insert("0.0", txt)
        self.model_text.configure(state="disabled")


class _Card(ctk.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master, fg_color="#F7F8FA", corner_radius=8, border_width=1, border_color="#E5E7EB")
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=13, weight="bold"), text_color=MUTED).pack(
            anchor="w", padx=10, pady=(6, 0))
        self.verdict = ctk.CTkLabel(self, text="–", font=ctk.CTkFont(size=30, weight="bold"), text_color=TXT)
        self.verdict.pack(anchor="w", padx=10)
        self.name = ctk.CTkLabel(self, text="", text_color=TXT, font=ctk.CTkFont(size=13))
        self.name.pack(anchor="w", padx=10)
        self.conf = ctk.CTkLabel(self, text="", text_color=MUTED, font=ctk.CTkFont(size=12), justify="left",
                                 anchor="w", wraplength=300)
        self.conf.pack(anchor="w", padx=10)
        self.flag = ctk.CTkLabel(self, text="", text_color=WARN, font=ctk.CTkFont(size=12, weight="bold"),
                                 justify="left", anchor="w", wraplength=300)
        self.flag.pack(anchor="w", padx=10)
        self.scope = ctk.CTkLabel(self, text="", text_color=OK, font=ctk.CTkFont(size=12), justify="left",
                                  anchor="w", wraplength=300)
        self.scope.pack(anchor="w", padx=10, pady=(0, 6))
        self.bind("<Configure>", self._rewrap)

    def _rewrap(self, event):
        w = max(event.width / ctk.ScalingTracker.get_widget_scaling(self) - 24, 120)
        for lab in (self.conf, self.flag, self.scope):
            lab.configure(wraplength=w)

    def clear(self, message=""):
        self.verdict.configure(text="–", text_color=TXT)
        self.name.configure(text=message)
        for lab in (self.conf, self.flag, self.scope):
            lab.configure(text="")

    def show(self, r, reference=None, extrapolated=False):
        self.verdict.configure(text=r.pred, text_color=LEVEL_COLORS[r.pred])
        self.name.configure(text=LEVEL_NAMES[r.pred])
        lines = [f"Confidence {r.confidence:.0%}"]
        if r.bin_accuracy is not None:
            lines.append(f"External-test accuracy at this confidence: {r.bin_accuracy:.1%}")
        if r.p_worse_ext is not None and r.pred != "C":
            lines.append(f"External test: {r.p_worse_ext:.1%} of {r.pred} predictions had a worse pushover level")
        if reference:
            lines.append(f"Pushover reference (example frame): {reference}")
        self.conf.configure(text="\n".join(lines))
        self.flag.configure(text="⚠ Low confidence: detailed assessment recommended" if r.low_confidence else "")
        if extrapolated:
            self.scope.configure(text="⚠ Inputs outside the simulated population: extrapolation", text_color=WARN)
        else:
            self.scope.configure(text="✔ Inputs inside the simulated population", text_color=OK)
