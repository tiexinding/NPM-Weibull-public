"""Paper#5 appendix figure v6 (260923): caption encodings moved into the figure (arm legend already; one point per matrix, symlog, unit-area density, mean-centred field label).
v4 (260920): distinct arm palette + arm meanings in legends, (b) one colour per curve, all solid.
v3 (260919): two panels only, (a) skewness/kurtosis by kind and arm, (b) representative
fields (examples). Panels (b) participation ratio and (d) bimodality of v2 are dropped (老丁 260919). Base: v2 (260916): the scale field beyond its width. Plot-only, reads
08_paper5_draft/data/p5v2_field_distribution_v1.json (written by scripts/p5v2_field_distribution_v1.py).
Writes p5_compile/figure_v2/figures/P5v2_S12_field_distribution_v5.png.

Changes vs v1: (a) the twin-axis panel is split into two stacked panels (skewness / excess kurtosis) sharing
the x axis; (b) one small panel per kind (medians over seeds per family) instead of 21 lines in one axes;
(c) legend carries identities only, statistics go to the caption; (d) title note moved to the caption, only
non-zero dip shares are printed.
"""
import json
from pathlib import Path
import numpy as np
RNG = np.random.default_rng(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve()
DRAFT = HERE.parents[3]
OUT = HERE.parents[1] / "figures" / "P5v2_S12_field_distribution_v6.png"
D = json.load(open(DRAFT / "data/p5v2_field_distribution_v1.json"))
assert D["schema"] == "p5v2-field-distribution-v1", D["schema"]

FAM = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAB = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
SIDE = {"q_proj": "row", "k_proj": "row", "v_proj": "row", "o_proj": "col", "gate_proj": "row", "up_proj": "row", "down_proj": "col"}
ARMS = ["D1", "D2", "D3", "D4"]; STEP_A = 5000
INITS = ["gaussian", "laplace", "uniform"]; EA_STEPS = [0, 800, 3200, 10000, 20000, 25000, 30000]
C_ARM = {"D1": "#1B5E9E", "D2": "#E6A100", "D3": "#C0392B", "D4": "#2E8B57"}
ARM_LABEL = {"D1": "D1 structured, single pass", "D2": "D2 shuffled, single pass", "D3": "D3 shuffled, 64$\\times$ repeated", "D4": "D4 structured, 64$\\times$ repeated"}
C_INIT = {"gaussian": "black", "laplace": "#B2182B", "uniform": "#2166AC"}
INIT_LABEL = {"gaussian": "Gaussian", "laplace": "Laplace", "uniform": "uniform"}
BC_T = float(D["BC_threshold"]); assert abs(BC_T - 5 / 9) < 1e-12, BC_T
KAS = D["tclock_kind_arm_step"]; KIS = D["ea_kind_init_step"]; HX = D["hist_examples"]
for f in FAM:
    for a in ARMS:
        assert KAS[f"{a}|{STEP_A}|{LAB[f]}"]["side"] == SIDE[f], (f, a)
PM = [m for m in D["per_matrix"]["tclock"] if m["step"] == STEP_A]
assert len(PM) == 7 * 6 * 12 == 504, len(PM)


def sx(v):
    v = np.asarray(v, float); return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))



plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10})
fig = plt.figure(figsize=(13.0, 4.6))
outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.18, left=0.05, right=0.99, top=0.90, bottom=0.13)
xs = np.arange(len(FAM))
# ---- (a) skewness (top) and excess kurtosis (bottom), grid at step 5,000
ga = outer[0, 0].subgridspec(2, 1, hspace=0.10)
ax1 = fig.add_subplot(ga[0]); ax2 = fig.add_subplot(ga[1], sharex=ax1)
for i, f in enumerate(FAM):
    for ai, arm in enumerate(ARMS):
        ms = [m[SIDE[f]] for m in PM if m["kind"] == f and m["arm"] == arm]
        assert len(ms) == 18, (f, arm, len(ms))
        off = (ai - 1.5) * 0.19
        for ax, key in ((ax1, "skew"), (ax2, "kurt_excess")):
            y = [m[key] for m in ms]
            ax.scatter(i + off + RNG.uniform(-0.035, 0.035, len(y)), y, s=7, facecolors="none", edgecolors=C_ARM[arm], alpha=0.85, lw=0.6, zorder=2)
            ax.plot([i + off - 0.06, i + off + 0.06], [np.median(y)] * 2, color=C_ARM[arm], lw=2.0, zorder=3)
for ax in (ax1, ax2):
    ax.axhline(0, color="0.6", lw=0.7, zorder=1); ax.grid(alpha=0.18, lw=0.5)
ax2.set_yscale("symlog", linthresh=5); ax2.set_ylim(-3, 160); ax2.set_yticks([0, 5, 20, 100]); ax2.set_yticklabels(["0", "5", "20", "100"])
ax1.set_ylabel(r"Skewness of $\tilde h$"); ax2.set_ylabel(r"Excess kurtosis (symlog)")
ax1.tick_params(labelbottom=False)
ax2.set_xticks(xs); ax2.set_xticklabels([("$%s$" % LAB[f]) if LAB[f] in ("q", "k", "v", "o") else LAB[f] for f in FAM])
ax1.set_ylim(-4.5, 18.5)
ax1.set_title("(a)  Skewness and tail of the identity-axis field, grid at step 5,000", loc="left")
ax1.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=C_ARM[a], color=C_ARM[a], ms=4, label=ARM_LABEL[a]) for a in ARMS] +
           [Line2D([], [], color="0.3", lw=2.0, label="median of 18 matrices"),
            Line2D([], [], marker="o", ls="", mfc="none", mec="0.3", ms=4, label="one matrix")],
           loc="upper left", frameon=False, ncol=3, columnspacing=1.2, handlelength=1.2, handletextpad=0.4, fontsize=8, borderpad=0.1)
# ---- (b) representative fields (examples)
ax = fig.add_subplot(outer[0, 1])
order = [("D1|q|L3", "row_field", "q row, layer 3, D1 (structured)", "#C0392B", "-"),
         ("D3|q|L3", "row_field", "q row, layer 3, D3 (shuffled, repeated)", "#E6A100", "-"),
         ("D3|gate|L0", "row_field", "gate row, layer 0, D3 (shuffled, repeated)", "#2E8B57", "-"),
         ("D1|v|L3", "row_field", "v row, layer 3, D1 (structured)", "#1B5E9E", "-"),
         ("D1|v|L3", "col_field", "v column, layer 3, D1 (structured)", "#7B3FA0", "-")]
hmin = min(min(HX[k][fld]) for k, fld, *_ in order); hmax = max(max(HX[k][fld]) for k, fld, *_ in order)
bins = np.linspace(np.floor(hmin * 10) / 10, np.ceil(hmax * 10) / 10, 61)
for k, fld, lab, c, ls in order:
    ax.hist(np.array(HX[k][fld]), bins=bins, density=True, histtype="step", color=c, ls=ls, lw=1.6, label=lab)
ax.set_xlabel(r"Mean-centred field  $\tilde h$"); ax.set_ylabel("Density (unit area per histogram)")
ax.set_title("(b)  Five example fields, seed 1, step 5,000", loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.8)
ax.grid(alpha=0.18, lw=0.5)
for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8.6)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
for k, fld, lab, *_ in order:
    st = HX[k]["row" if fld == "row_field" else "col"]
    print(f"{lab:24s} H {st['sd']:.2f} skew {st['skew']:+.2f}")
for f in FAM:
    print(LAB[f], "kurt", {a: round(KAS[f'{a}|{STEP_A}|{LAB[f]}']["kurt_excess"], 1) for a in ARMS}, "skew", {a: round(KAS[f'{a}|{STEP_A}|{LAB[f]}']["skew"], 2) for a in ARMS})
