"""Paper#5 appendix figure v3 (260923): house style — legends at the top of each panel with raised y limits; (a) pooled value and R0^2 printed at each kind, 6/pi^2 line with markers and value; (c) model encoding as in Figure 3(d) (grey = baseline, light / full kind colour) instead of blue/red that clashed with the kind colours; encodings moved from the caption into the legends.
v2 (260919): panels (a)(b) transposed to the Figure 2 convention
(kinds on the horizontal axis, coefficient on the vertical, translucent bar = 12-run range, hollow marker = pooled fit).
Base (260916 late): bridge coefficient detail, one row of three panels.

Lifted from the former main-text panels when Figure 3 v3 absorbed the two-axis scatter:
  (a) one-axis protocol coefficient alpha_P per kind (pooled fit, 12-run range, per-row R0^2)   [was Fig 3 v2 (c)]
  (b) two-axis alpha_r / alpha_c per kind with the 12-run range, clipped at +-2.5                [was Fig 4 v2 (b)]
  (c) leave-one-run-out MAE per kind for the row-only / row+column / +cross-term models          [was Fig 4 v2 (c)]

Reads ONLY stored JSON:
  06_hrow/kmix_bridge_v1_1.json                          partA (per-kind alpha, run_alpha_range, R0_sq)
  08_paper5_draft/data/p5v2_twoaxis_bridge_v1.json       fits.<kind>.{M1,M2,M3}
Writes:
  08_paper5_draft/p5_compile/figure_v2/figures/P5v3_SB_bridge_coefficients_v2.png
Symbols: alpha_P (one-axis, Sec. 3.2 eq:bridge), alpha_r / alpha_c (two-axis, eq:bridge2).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve()
DRAFT = HERE.parents[3]
SM = DRAFT.parent
OUT = HERE.parents[1] / "figures" / "P5v3_SB_bridge_coefficients_v3.png"

A = json.load(open(SM / "06_hrow/kmix_bridge_v1_1.json"))
D = json.load(open(DRAFT / "data/p5v2_twoaxis_bridge_v1.json"))
assert D["schema"] == "p5v2-twoaxis-bridge-v1", D["schema"]
ALPHA_WEIBULL = 6 / np.pi ** 2

FAM5 = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
FAM7 = list(D["families"])
assert FAM7 == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], FAM7
LAB = {"q_proj": "$q$", "k_proj": "$k$", "v_proj": "$v$", "o_proj": "$o$", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
C_QK, C_VO, C_FFN = "#B2182B", "#2166AC", "#1B7837"
COL = {"q_proj": C_QK, "k_proj": C_QK, "v_proj": C_VO, "o_proj": C_VO, "gate_proj": C_FFN, "up_proj": C_FFN, "down_proj": C_FFN}
MRK = {"q_proj": "o", "k_proj": "^", "v_proj": "o", "o_proj": "^", "gate_proj": "o", "up_proj": "^", "down_proj": "s"}

# ---- asserts
assert A["partA"]["J_A_v11"] == "supported"
for f in FAM5:
    p = A["partA"][f]
    assert p["runs_same_sign"] == 12 and len(p["run_alpha_range"]) == 2 and 0 < p["R0_sq"] <= 1, f
fits = D["fits"]
MODELS = (("M1", None, r"$H^W_{\mathrm{row}}$ only"), ("M2", 0.5, r"$+\,H^W_{\mathrm{col}}$"), ("M3", 1.0, r"$+$ cross term"))
for f in FAM7:
    assert fits[f]["M2"]["coef_names"] == ["a_r", "a_c"], f
    for m, _, _ in MODELS:
        assert "median" in fits[f][m]["pred_mae_LORO"] and "max" in fits[f][m]["pred_mae_LORO"], (f, m)


# ---- figure (Figure 2 convention: kinds on x, value on y)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10})
fig = plt.figure(figsize=(14.6, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.05, 1.05], wspace=0.28, left=0.05, right=0.99, top=0.90, bottom=0.13)

# (a) alpha_P per kind
ax = fig.add_subplot(gs[0, 0])
for i, f in enumerate(FAM5):
    p = A["partA"][f]; lo, hi = p["run_alpha_range"]
    ax.plot([i, i], [lo, hi], color=COL[f], lw=2.6, alpha=0.18, solid_capstyle="butt", zorder=2)
    ax.scatter(i, p["alpha"], marker=MRK[f], s=48, facecolor="white", edgecolor=COL[f], lw=1.5, zorder=4)
    ax.text(i, hi + 0.015, f"{p['alpha']:.3f}\n" + rf"$R^2_0$ {p['R0_sq']:.3f}", ha="center", va="bottom", fontsize=7.4, color="0.3", linespacing=1.1)
xw = np.linspace(-0.6, len(FAM5) - 0.4, 60)
ax.plot(xw, np.full_like(xw, ALPHA_WEIBULL), color="0.55", lw=1.0, ls=(0, (5, 3)), marker="D", ms=3.8, mfc="white",
        mec="0.35", mew=0.8, markevery=(4, 10), zorder=1)
ax.set_xticks(range(len(FAM5))); ax.set_xticklabels([LAB[f] for f in FAM5])
ax.set_xlim(-0.6, len(FAM5) - 0.4)
ax.set_ylim(0.36, max(A["partA"][f]["run_alpha_range"][1] for f in FAM5) + 0.36)
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
ax.set_ylabel(r"Protocol coefficient $\alpha_P$")
ax.set_title(r"(a)  $\alpha_P$ per kind, pooled and per run", loc="left")
h_a = [Line2D([], [], ls="none", marker="o", mfc="white", mec="0.3", mew=1.5, ms=6, label="pooled fit (value, $R^2_0$)"),
       Line2D([], [], color="0.3", lw=2.6, alpha=0.25, label="range over 12 runs"),
       Line2D([], [], color="0.55", lw=1.0, ls=(0, (5, 3)), marker="D", ms=3.8, mfc="white", mec="0.35", mew=0.8,
              label=r"$6/\pi^2 = 0.608$, full-Weibull reference")]
ax.legend(handles=h_a, loc="upper left", ncol=1, fontsize=8, frameon=False, handlelength=1.6, columnspacing=1.0,
          borderpad=0.2, bbox_to_anchor=(0.0, 1.0))

# (b) two-axis alpha_r / alpha_c per kind
ax = fig.add_subplot(gs[0, 1])
YL = 2.5; OFFB = 0.17; clipped = []
for i, f in enumerate(FAM7):
    m2 = fits[f]["M2"]; lo, hi = m2["run_coef_range"]
    if min(lo) < -YL or max(hi) > YL: clipped.append(LAB[f])
    for j, (mk, fc) in enumerate((("o", COL[f]), ("s", "white"))):
        x = i + (j - 0.5) * 2 * OFFB
        ax.plot([x, x], [max(lo[j], -YL), min(hi[j], YL)], color=COL[f], lw=2.6, alpha=0.18, solid_capstyle="butt", zorder=2)
        ax.scatter(x, m2["coef"][j], marker=mk, s=44, facecolor=fc, edgecolor=COL[f], lw=1.4, zorder=4)
    ax.axvline(i + 0.5, color="0.90", lw=0.8, zorder=0)
ax.axhline(0, color="0.55", lw=0.8, zorder=1)
ax.set_xticks(range(len(FAM7))); ax.set_xticklabels([LAB[f] for f in FAM7])
ax.set_xlim(-0.6, len(FAM7) - 0.4); ax.set_ylim(-YL, YL + 0.9)
ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_ylabel(r"Two-axis coefficient")
ax.set_title(r"(b)  $\alpha_r$ and $\alpha_c$ per kind", loc="left")
h_b = [Line2D([], [], ls="none", marker="o", color="0.3", ms=6, label=r"$\alpha_r$ (rows)"),
       Line2D([], [], ls="none", marker="s", mfc="white", mec="0.3", mew=1.4, ms=6, label=r"$\alpha_c$ (columns)"),
       Line2D([], [], color="0.3", lw=2.6, alpha=0.25, label=r"12-run range, clipped at $\pm 2.5$")]
ax.legend(handles=h_b, loc="upper left", ncol=3, fontsize=8, frameon=False, handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.0, 1.0))

# (c) LORO MAE per kind and model (unchanged)
ax = fig.add_subplot(gs[0, 2])
w = 0.26; x = np.arange(len(FAM7))
for j, (m, c, lab) in enumerate(MODELS):
    med = [fits[f][m]["pred_mae_LORO"]["median"] for f in FAM7]
    mx = [fits[f][m]["pred_mae_LORO"]["max"] for f in FAM7]
    ax.bar(x + (j - 1) * w, med, w * 0.9, color=("0.75" if c is None else [COL[f] for f in FAM7]), alpha=(1.0 if c is None else c), zorder=3)
    ax.plot(x + (j - 1) * w, mx, ls="none", marker="_", color="0.15", ms=7, mew=1.2, zorder=4)
ax.set_xticks(x); ax.set_xticklabels([LAB[f] for f in FAM7])
ax.set_ylabel(r"$|\widehat{k}_{\mathrm{raw}} - k_{\mathrm{raw}}|$, median of 12 folds")
ax.set_ylim(0, 0.043)
ax.set_title("(c)  Leave-one-run-out error by model", loc="left")
from matplotlib.patches import Patch
h_c = [Patch(facecolor="0.75", label=MODELS[0][2]),
       Patch(facecolor="0.4", alpha=0.5, label=MODELS[1][2]),
       Patch(facecolor="0.4", label=MODELS[2][2]),
       Line2D([], [], marker="_", ls="none", color="0.15", ms=8, mew=1.2, label="worst fold")]
ax.legend(handles=h_c, loc="upper left", ncol=2, fontsize=8, frameon=False, handlelength=1.2, handletextpad=0.5,
          columnspacing=1.0, borderpad=0.2, bbox_to_anchor=(0.0, 1.0))
for tl, f in zip(ax.get_xticklabels(), FAM7): tl.set_color(COL[f])

for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", OUT); print("clipped at +-%.1f:" % YL, clipped)
