#!/usr/bin/env python3
"""Fig 7 v6 (260917, 老丁): v5 + five highlighted groups (cell sets, colours off the RdBu map) + smaller canvas. v5: v4 + channel-space names spelled out. v4: 2 x 2 layout, upper triangle only (the matrix is symmetric), larger cells.
Data, factor order, block borders, colour scale and label rule unchanged from v3.
Reads only 08_paper5_draft/data/p5t1_V_factor_matrix_v1.json.
Writes 08_paper5_draft/p5_compile/figure_v2/figures/P5v2_S19_factor_topology_v6.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
FIG = ROOT / "08_paper5_draft/p5_compile/figure_v2/figures/P5v2_S19_factor_topology_v6.png"
if "--staircase" in __import__("sys").argv:
    FIG = FIG.with_name("P5v2_S19_factor_topology_v4_staircase_preview.png")
FM = json.load(open(ROOT / "08_paper5_draft/data/p5t1_V_factor_matrix_v1.json"))
SLOTS = FM["slots"]; N = len(SLOTS)
LAB = [s["label"] for s in SLOTS]
SPACES = list(dict.fromkeys(s["space"] for s in SLOTS))
SPACE_NAME = {"resid-pre": "attention input", "resid-post": "FFN input", "resid-err": "residual-output\nerror",
              "qk-logit": "q/k logit\nchannels", "head/value": "head/value\nchannels", "ffn-hidden": "FFN hidden\nunits"}
assert set(SPACES) == set(SPACE_NAME), SPACES
SHOW = [200, 800, 3200, 10000]
assert all(str(t) in FM["V_matrix_by_step"] for t in SHOW), list(FM["V_matrix_by_step"])
assert FM["last_step"] == SHOW[-1]
import sys
DIAG_CUT = "--staircase" not in sys.argv
LABEL_THRESH = 0.6
HI = [("gate.col$\\leftrightarrow$up.col: 1.00 throughout", "#1B9E77", "-", [("gate.col", "up.col")]),
      ("attention-input pairs: rise gradually", "#E6A100", "-", [("q.col", "k.col"), ("q.col", "v.col"), ("k.col", "v.col")]),
      ("v.row$\\leftrightarrow$o.col: appears late", "#7B3294", "-", [("v.row", "o.col")]),
      ("FFN hidden-unit pairs: peak, then relax", "#00A3A3", "-", [("gate.row", "up.row"), ("gate.row", "down.col"), ("up.row", "down.col")]),
      ("attention $\\times$ FFN input: decouple", "0.15", (0, (2.2, 1.6)), [(a, b) for a in ("q.col", "k.col", "v.col") for b in ("gate.col", "up.col")])]
IDX = {s["label"]: i for i, s in enumerate(SLOTS)}
assert all(a in IDX and b in IDX for _, _, _, cells in HI for a, b in cells)
edges, cur = [], SLOTS[0]["space"]
for i, s in enumerate(SLOTS):
    if s["space"] != cur:
        edges.append(i); cur = s["space"]
bounds = [0] + edges + [N]
same_space = np.zeros((N, N), bool)
for gi in range(len(bounds) - 1):
    a, b = bounds[gi], bounds[gi + 1]; same_space[a:b, a:b] = True

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
fig = plt.figure(figsize=(8.0, 7.5))
gs = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.16, left=0.085, right=0.80, top=0.95, bottom=0.15)
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.90")
im = None
for ax_i, t in enumerate(SHOW):
    r_, c_ = divmod(ax_i, 2)
    ax = fig.add_subplot(gs[r_, c_])
    M = np.array(FM["V_matrix_by_step"][str(t)], float)
    assert M.shape == (N, N) and np.allclose(np.diag(M), 1.0, atol=1e-9)
    assert np.allclose(np.nan_to_num(M), np.nan_to_num(M.T), atol=1e-9), "matrix must be symmetric"
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-1, vmax=1)
    for e in edges:
        ax.plot([e - .5, N - .5], [e - .5, e - .5], color="k", lw=1.1, zorder=4)   # horizontal, right of diagonal
        ax.plot([e - .5, e - .5], [-.5, e - .5], color="k", lw=1.1, zorder=4)      # vertical, above diagonal
    for i in range(N):
        for j in range(i + 1, N):
            if np.isnan(M[i, j]):
                continue
            if same_space[i, j] or abs(M[i, j]) >= LABEL_THRESH:
                ax.text(j, i, f"{M[i,j]:.2f}".lstrip("0").replace("-0", "-"), ha="center", va="center",
                        fontsize=6.8, color="white" if abs(M[i, j]) > .7 else "0.15", zorder=5)
    for _, col, ls, cells in HI:
        rr = sorted(set(min(IDX[a], IDX[b]) for a, b in cells)); cc = sorted(set(max(IDX[a], IDX[b]) for a, b in cells))
        if len(cells) == len(rr) * len(cc):      # full rectangle -> one box
            ax.add_patch(Rectangle((cc[0] - .5, rr[0] - .5), len(cc), len(rr), fill=False, ec=col, lw=1.7, ls=ls, zorder=6))
        else:
            for a, b in cells:
                r, c = sorted((IDX[a], IDX[b]))
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False, ec=col, lw=1.7, ls=ls, zorder=6))
    # hide the lower triangle (symmetric). DIAG_CUT: straight cut through the diagonal cells;
    # otherwise a staircase that keeps the diagonal cells whole.
    if DIAG_CUT:
        ax.add_patch(Polygon([(-.5, -.5), (-.5, N - .5), (N - .5, N - .5)], closed=True, fc="white", ec="white",
                             lw=0, zorder=3))
        ax.plot([-.5, N - .5], [-.5, N - .5], color="k", lw=0.9, zorder=4)
    else:
        ax.add_patch(Polygon([(-.5, .5), (-.5, N - .5), (N - 1.5, N - .5)], closed=True, fc="white", ec="white",
                             lw=0, zorder=3))
        for i in range(N):
            ax.plot([i - .5, i + .5], [i - .5, i - .5], color="k", lw=0.8, zorder=4)
            if i > 0:
                ax.plot([i - .5, i - .5], [i - .5, i + .5], color="k", lw=0.8, zorder=4)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(LAB if r_ == 1 else [], rotation=90, fontsize=7.5)
    ax.set_yticklabels(LAB if c_ == 0 else [], fontsize=7.5)
    ax.tick_params(length=0)
    for sp_ in ("top", "right", "left", "bottom"):
        ax.spines[sp_].set_visible(False)
    ax.set_title(f"({'abcd'[ax_i]})  step {t:,}", loc="left")
    if c_ == 1:
        for gi, sp in enumerate(SPACES):
            a, b = bounds[gi], bounds[gi + 1]
            ax.text(N - 0.2, (a + b - 1) / 2, SPACE_NAME[sp], fontsize=7.6, va="center", ha="left", color="0.2", linespacing=1.1)
cax = fig.add_axes([0.94, 0.33, 0.014, 0.40])
cb = fig.colorbar(im, cax=cax); cb.set_label("Spearman correlation of the two factors", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
handles = [Line2D([], [], ls="none", marker="s", mfc="none", mec=col, mew=1.7, ms=7, label=lab) for lab, col, ls, _ in HI[:4]]
handles.append(Line2D([], [], color="0.15", lw=1.4, ls=(0, (2.2, 1.6)), label=HI[4][0]))
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.4, bbox_to_anchor=(0.45, 0.0),
           handletextpad=0.5, columnspacing=1.3, labelspacing=0.35)
fig.savefig(FIG, dpi=220, facecolor="white")
print("wrote", FIG.relative_to(ROOT))
for lab, _, _, cells in HI:
    print(lab)
    for a, b in cells:
        print("   %-18s" % f"{a}~{b}", "  ".join(f"{t}:{np.array(FM['V_matrix_by_step'][str(t)])[IDX[a], IDX[b]]:+.2f}" for t in SHOW))
iu = np.triu_indices(N, 1)
