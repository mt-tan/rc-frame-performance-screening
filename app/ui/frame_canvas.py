"""Frame elevation and column cross-section sketch (matplotlib, drawn from the derived parameters)."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Rectangle

COL, BEAM, BAR, TIE = "#4B5563", "#9CA3AF", "#B42318", "#1D4E89"


def render_frame(fig, p: dict | None):
    fig.clear()
    if p is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Enter valid frame data", ha="center", va="center", color="#6B7280")
        ax.axis("off")
        return
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.12, top=0.97, bottom=0.03)
    _elevation(fig.add_subplot(gs[0]), p)
    _section(fig.add_subplot(gs[1]), p)


def _elevation(ax, p):
    bays = p["bay_lengths"]
    xs = np.concatenate([[0.0], np.cumsum(bays)])
    hs = [p["h1"]] + [p["htyp"]] * (p["N_story"] - 1)
    ys = np.concatenate([[0.0], np.cumsum(hs)])
    cw, bh = p["col_H"], p["beam_h"]
    for x in xs:
        ax.add_patch(Rectangle((x - cw / 2, 0), cw, ys[-1], color=COL, lw=0, alpha=0.85))
    for y in ys[1:]:
        ax.add_patch(Rectangle((xs[0] - cw / 2, y - bh), xs[-1] + cw, bh, color=BEAM, lw=0))
    ax.plot([xs[0] - 0.8, xs[-1] + 0.8], [0, 0], color="#111827", lw=1.5)
    for x in xs:
        ax.plot([x - 0.35, x + 0.35], [-0.12, -0.12], color="#111827", lw=0.8)
    for i, L in enumerate(bays):
        ax.text((xs[i] + xs[i + 1]) / 2, -0.55, f"{L:.2f}", ha="center", va="top", fontsize=7, color="#374151")
    ax.text(xs[-1] + 0.9, ys[1] / 2, f"h1 = {p['h1']:.2f}", fontsize=7, va="center", color="#374151")
    if p["N_story"] > 1:
        ax.text(xs[-1] + 0.9, (ys[1] + ys[2]) / 2, f"h = {p['htyp']:.2f}", fontsize=7, va="center", color="#374151")
    ax.set_title(f"{p['N_story']}-story, {p['N_bay']}-bay frame  (H = {ys[-1]:.2f} m)", fontsize=9)
    ax.set_xlim(xs[0] - 1.0, xs[-1] + 3.2)
    ax.set_ylim(-1.3, ys[-1] + 0.6)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")


def _section(ax, p):
    B, H, c = p["col_B"], p["col_H"], p["cover"]
    phi, pe = p["col_phi_long"], p["col_phi_etr"]
    nx, ny = p["col_n_long_x"], p["col_n_long_y"]
    # drawn with h (in-plane) horizontal, b vertical, matching the elevation
    ax.add_patch(Rectangle((-H / 2, -B / 2), H, B, fc="#F3F4F6", ec=COL, lw=1.2))
    ax.add_patch(Rectangle((-H / 2 + c, -B / 2 + c), H - 2 * c, B - 2 * c, fc="none", ec=TIE, lw=1.0))
    yb = H / 2 - c - pe - phi / 2
    zb = B / 2 - c - pe - phi / 2
    pts = set()
    for z in np.linspace(-zb, zb, nx):           # b-faces at ±yb
        pts.update({(round(-yb, 6), round(z, 6)), (round(yb, 6), round(z, 6))})
    for y in np.linspace(-yb, yb, ny):           # h-faces at ±zb
        pts.update({(round(y, 6), round(-zb, 6)), (round(y, 6), round(zb, 6))})
    r = max(phi / 2, 0.006)
    for y, z in pts:
        ax.add_patch(Circle((y, z), r, color=BAR))
    ax.set_title(f"Column {B * 100:.0f}×{H * 100:.0f} cm, {len(pts)}Ø{phi * 1000:.0f}, "
                 f"Ø{pe * 1000:.0f}/{p['col_s_etr'] * 100:.0f} cm, {p['hook_deg']}°", fontsize=8)
    ax.text(0, -B / 2 - 0.04, "h (in-plane)", ha="center", va="top", fontsize=7, color="#6B7280")
    m = max(B, H) / 2 + 0.08
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
