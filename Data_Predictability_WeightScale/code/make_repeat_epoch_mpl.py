#!/usr/bin/env python
"""Repetition -> inverted-U (repeat ~ multi-epoch proxy) — matplotlib redraw of p3_repeat_epoch.
Data pipeline verbatim (numbers unchanged); English, no in-figure title, no callout arrow (守#82),
shaded Muennighoff regions. p3_style. B2 6-22. 守 #19/#76/#47.
"""
import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p3_style import apply_style, PALETTE
RS = [1, 2, 4, 8, 16, 32]
def Y2(f): j = json.load(open(f)); return (j[-1]["lambda"]**2 - j[0]["lambda"]**2) * 1e4
r, Y = [], []
for k in RS:
    f = f"_tmp_repeat/repeat_r{k:02d}_s0/lambda_trajectory_continue.json"
    if os.path.exists(f): r.append(k); Y.append(Y2(f))
r = np.array(r, float); Y = np.array(Y); x = np.log2(r)
c = np.polyfit(x, Y, 2); xs = np.linspace(x.min(), x.max(), 120); ys = np.polyval(c, xs)
ipk = int(np.argmax(Y))

apply_style()
fig, ax = plt.subplots(figsize=(6.5, 4.2))
# phase regions marked by neutral dividers + top labels (no solid colour fills)
REG = [(0.8, 4, r"$\lesssim$4 ep: $\approx$ new data"),
       (4, 16, "4-16 ep: diminishing"),
       (16, 40, ">16 ep: useless")]
for xb in (4, 16):
    ax.axvline(xb, ls=(0, (2, 3)), lw=0.8, color=PALETTE["NEUTRAL"], alpha=0.55, zorder=0)
for x0, x1, lab in REG:
    ax.text(np.sqrt(x0 * x1), 0.97, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7.5, color=PALETTE["NEUTRAL"])
ax.plot(2**xs, ys, "--", color=PALETTE["FIT"], lw=1.8, label="inverted-U trend (quadratic)")
ax.scatter(r, Y, s=45, color=PALETTE["FIT"], edgecolors=PALETTE["INK"], linewidths=0.5, zorder=4, label=r"$\lambda^2-\lambda_0^2$ (n=1)")
for xi, yi in zip(r, Y): ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
ax.scatter([r[ipk]], [Y[ipk]], s=180, marker="*", color=PALETTE["ACCENT"], edgecolors=PALETTE["INK"], linewidths=0.5, zorder=5)
ax.annotate(f"peak r$\\approx${int(r[ipk])}", (r[ipk], Y[ipk]), textcoords="offset points", xytext=(14, -2),
            ha="left", va="center", fontsize=8, color=PALETTE["ACCENT"])
ax.axhline(Y[0], ls=":", lw=1.0, color=PALETTE["NEUTRAL"])
ax.text(r[-1], Y[0], r" $r{=}1$ baseline", ha="right", va="bottom", fontsize=7.5, color=PALETTE["NEUTRAL"])
ax.set_xscale("log", base=2); ax.set_xticks(RS); ax.set_xticklabels([str(k) for k in RS])
ax.set_xlim(0.8, 40); ax.set_ylim(1.35, 2.32)
ax.set_xlabel(r"repeat factor $r$  $\approx$  epochs over $1/r$ unique data (fixed 30M-token budget)")
ax.set_ylabel(r"$\lambda^2-\lambda_0^2\ (\times10^{-4})$")
ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.01, 0.87))
for ext in ("pdf","png"): fig.savefig(f"03_paper/figures/p3_repeat_epoch.{ext}", bbox_inches="tight")
print(f"WROTE p3_repeat_epoch  (peak r={int(r[ipk])} Y={Y[ipk]:.2f})")
