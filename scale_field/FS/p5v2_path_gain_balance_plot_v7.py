"""Paper#5 appendix gain/balance figure v6 (260923): caption encodings moved into legends (line style/marker per source, null and mismatch step, median/IQR, index vs sorted note).
v5 (260921): legend labels renamed to the Table 4 source names (initialization families 30k / controlled data grid 5k). v4 (260921): legends of both panels moved to the top inside the axes, y ranges
raised for headroom, figure slightly taller (老丁 260921). v3 (260920): panels (a),(b) only; (c) per-unit scatter and (d) by-arm bars
removed (老丁 260920; by-arm medians go to the text, values still printed below). Based on v2 (260916): path gain vs balance. Plot-only, reads
08_paper5_draft/data/p5v2_path_gain_balance_v1.json (written by scripts/p5v2_path_gain_balance_v1.py).
Writes p5_compile/figure_v2/figures/P5v2_S13_path_gain_balance_v5.png.

Changes vs v1: (a) legend moved out of the data (own column); in-panel statistics of (c) removed (caption);
axis titles shortened (definitions of g, b in caption); up/down colour = paper FFN green; thinner lines.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve()
DRAFT = HERE.parents[3]
OUT = HERE.parents[1] / "figures" / "P5v2_S13_path_gain_balance_v7.png"
D = json.load(open(DRAFT / "data/p5v2_path_gain_balance_v1.json"))
assert D["schema"] == "p5v2-path-gain-balance-v1", D["schema"]
S = D["summary"]

MAIN = ["QK_pair", "VO", "UD"]
GCOL = {"QK_pair": "#B2182B", "VO": "#2166AC", "UD": "#1B7837"}
GLAB = {"QK_pair": "$q$/$k$ by RoPE pair", "VO": "$v$ rows / $o$ cols", "UD": "up rows / down cols"}
GSHORT = {"QK_pair": "$q$/$k$ pair", "VO": "$v$/$o$", "UD": "up/down"}
STEPS_E = sorted(int(s) for s in S["ea_by_step_all_inits"]); STEPS_T = sorted(int(s) for s in S["tclock_by_step"])
assert STEPS_E == [0, 800, 3200, 10000, 20000, 25000, 30000] and STEPS_T == [800, 1600, 3200, 5000], (STEPS_E, STEPS_T)
ARMS = list(S["tclock_by_arm_step5000"]); assert ARMS == ["D1", "D2", "D3", "D4"], ARMS
ex = D["example_D1_s1_step5000_L3_UD"]
hu, hd = np.array(ex["h_up"]), np.array(ex["h_down"]); assert hu.size == hd.size == 1376, hu.size


def sx(v):
    v = np.asarray(v, float)
    return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))


plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5})
fig = plt.figure(figsize=(12.2, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28,
                      left=0.07, right=0.99, top=0.91, bottom=0.14)

# (a) ratio over training
ax = fig.add_subplot(gs[0, 0])
for g in MAIN:
    ye = [S["ea_by_step_all_inits"][str(s)][g]["ratio"]["median"] for s in STEPS_E]
    ax.plot(sx(STEPS_E), ye, color=GCOL[g], lw=1.7, marker="o", ms=3.4, mec="white", mew=0.6, zorder=4)
    yt = [S["tclock_by_step"][str(s)][g]["ratio"]["median"] for s in STEPS_T]
    ax.plot(sx(STEPS_T), yt, color=GCOL[g], lw=1.1, ls=(0, (3, 2)), marker="s", ms=3.2, mfc="white", zorder=3)
    nl = S["tclock_step5000"][g]["null_ratio_q025_median"]; nh = S["tclock_step5000"][g]["null_ratio_q975_median"]
    xo = sx(5000) + {MAIN[0]: -0.07, MAIN[1]: 0.0, MAIN[2]: 0.07}[g]   # null computed at step 5k only: short range bar per path
    ax.vlines(xo, nl, nh, color=GCOL[g], lw=5, alpha=0.28, zorder=2)
    ax.scatter([xo], [S["tclock_step5000"][g]["mismatch_ratio"]["median"]], marker="x", color=GCOL[g], s=36, zorder=5)
ax.axhline(1.0, color="0.4", lw=0.7)
xt = [0, 800, 3200, 10000, 30000]; ax.set_xticks(sx(xt)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
ax.set_xlabel("Training step  (linear to 800, log above)"); ax.set_ylabel("var($g$) / var($b$), median"); ax.set_yscale("log"); ax.set_ylim(0.4, 450); ax.set_yticks([1, 10, 100]); ax.set_yticklabels(["1", "10", "100"])
ax.set_title("(a)  Gain-to-balance ratio over training", loc="left", pad=6)
h = [Line2D([], [], color=GCOL[g], lw=1.7, label=GLAB[g]) for g in MAIN] + \
    [Line2D([], [], color="0.4", lw=1.7, marker="o", ms=3.4, mec="white", label="initialization families (9 runs, to 30k)"),
     Line2D([], [], color="0.4", lw=1.1, ls=(0, (3, 2)), marker="s", ms=3.2, mfc="white", label="data grid (12 runs, to 5k)"),
     Line2D([], [], marker="x", ls="", color="0.4", label="next-layer pairing, step 5k"),
     Line2D([], [], color="0.4", lw=5, alpha=0.28, label="within-layer re-pairing null (95%), step 5k")]
ax.legend(handles=h, fontsize=9.6, frameon=False, loc="upper left", ncol=2, handlelength=1.6, handletextpad=0.5,
          borderpad=0.2, labelspacing=0.5, columnspacing=1.2)

# (b) stability
ax = fig.add_subplot(gs[0, 1])
labels = []; x = 0
for g, lab in (("QK_pair", "$q$/$k$ pair\n(index)"), ("VO_head", "$v$/$o$ head\n(index)"), ("UD", "up/down\n(sorted)"), ("VO", "$v$/$o$\n(sorted)")):
    lvl = "index" if "index" in lab else "sorted"
    for src, mk in (("stability_tclock_cross_seed_step5000", "o"), ("stability_ea_30000_cross_init", "^")):
        sg = S[src][g][f"g_{lvl}"]; sb = S[src][g][f"b_{lvl}"]
        c = GCOL["VO" if g == "VO_head" else g]
        ax.errorbar(x - 0.15, sg["median"], yerr=[[sg["median"] - sg["q25"]], [sg["q75"] - sg["median"]]], fmt=mk, color=c, ms=5, capsize=2, lw=1.0, mfc=c if mk == "o" else "white")
        ax.errorbar(x + 0.15, sb["median"], yerr=[[sb["median"] - sb["q25"]], [sb["q75"] - sb["median"]]], fmt=mk, color="0.45", ms=5, capsize=2, lw=1.0, mfc="0.45" if mk == "o" else "white")
    labels.append(lab); x += 1
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=11.5)
ax.axhline(0, color="0.4", lw=0.7); ax.set_ylim(-0.3, 1.62)
ax.text(0.02, 0.79, "index: units paired by architectural coordinate\nsorted: sorted values compared", transform=ax.transAxes, ha="left", va="top", fontsize=9.6, color="0.4"); ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylabel("Correlation between runs")
ax.set_title("(b)  Reproducibility of $g$ and $b$", loc="left", pad=6)
h = [Line2D([], [], marker="o", ls="", color="0.3", ms=5, label="data grid, across seeds (5k)"),
     Line2D([], [], marker="^", ls="", color="0.3", mfc="white", ms=5, label="across initialization families (30k)"),
     Line2D([], [], marker="s", ls="", color=GCOL["UD"], ms=5, label="$g$ (colour)"),
     Line2D([], [], marker="s", ls="", color="0.45", ms=5, label="$b$ (grey)"),
     Line2D([], [], color="0.3", lw=1.0, marker="_", ms=6, label="median, interquartile range")]
ax.legend(handles=h, fontsize=9.6, frameon=False, loc="upper left", ncol=2, handletextpad=0.4, columnspacing=1.2, borderpad=0.2, labelspacing=0.5)

for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=11.5)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", OUT)

# caption read-outs
vg, vb = hu + hd, hu - hd
print("(c) D1_s1 L3 step %d n=%d var(g)/var(b)=%.1f r=%.2f" % (ex["step"], hu.size, vg.var() / vb.var(), np.corrcoef(hu, hd)[0, 1]))
for g in MAIN:
    print(g, "ea ratio by step", [round(S["ea_by_step_all_inits"][str(s)][g]["ratio"]["median"], 2) for s in STEPS_E],
          "grid", [round(S["tclock_by_step"][str(s)][g]["ratio"]["median"], 2) for s in STEPS_T],
          "null975", round(S["tclock_step5000"][g]["null_ratio_q975_median"], 2), "mismatch", round(S["tclock_step5000"][g]["mismatch_ratio"]["median"], 2))
for src in ("stability_tclock_cross_seed_step5000", "stability_ea_30000_cross_init"):
    print(src, {g: (round(S[src][g]["g_index"]["median"], 2), round(S[src][g]["b_index"]["median"], 2), round(S[src][g]["g_sorted"]["median"], 2), round(S[src][g]["b_sorted"]["median"], 2)) for g in ("QK_pair", "VO_head", "UD", "VO")})
print("arms", {g: {a: round(S["tclock_by_arm_step5000"][a][g]["ratio"]["median"], 1) for a in ARMS} for g in MAIN})
