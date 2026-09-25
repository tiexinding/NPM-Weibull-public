"""Fig 13 v7 (260923): house style — font sizes as Figs 2-9 (canvas widened to 13.4 in so print size matches at 0.92 textwidth), italic kind symbols, reference line and sample size in the legend strip (the clipped "1.204" label removed), x label states the step-axis convention.
Fig 10 v6 (260919): Figure 2(a) encoding: colour+marker = kind, filled = row field / hollow = column field,
line style = initialization family; thin lines, one legend strip on top. Base: v5: kinds grouped by path into three columns (q/k, v/o, gate/up/down); marker = kind,
colour = initialization family, line style = axis (row solid, column dashed). Base: v4 (260916): timing chain (core / identity-axis field / pooled shape) over training, cleaned.

Plot-only copy of scripts/p5v2_figS11_timing_v1.py: same read-out (family medians over 3 seeds x 6 layers
per checkpoint from p5v2_ea_kbi_trajectory_v1.json), same layout (3 rows x 7 kinds).
Changes: in-figure footer removed (moved to the caption); legend labels name the family; the reference
line is the protocol value taken from the JSON (1.204) and is labelled once; line weights reduced.
No read-out JSON or side table is written (the v1 script owns those).

Reads only 08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json
Writes    08_paper5_draft/p5_compile/figure_v2/figures/P5v2_S11_timing_chain_v2.png
"""
import json, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
EA = json.load(open(ROOT / "08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json"))
OUT = ROOT / "08_paper5_draft/p5_compile/figure_v2/figures/P5v2_S11_timing_chain_v8.png"

COMPS = list(EA["families"]); INITS = list(EA["inits"]); SEEDS = list(EA["seeds"]); STEPS = sorted(int(s) for s in EA["steps"])
K0 = float(EA["k0_gaussian_reference"])
rows = EA["per_matrix"]
assert len(rows) == EA["n_matrices"] == len(INITS) * len(SEEDS) * len(STEPS) * len(COMPS) * 6, len(rows)
assert sorted(set(r["step"] for r in rows)) == STEPS
LABEL = {"q_proj": "$q$", "k_proj": "$k$", "v_proj": "$v$", "o_proj": "$o$", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
IDAXIS = {"q_proj": ["H_row"], "k_proj": ["H_row"], "gate_proj": ["H_row"], "up_proj": ["H_row"],
          "o_proj": ["H_col"], "down_proj": ["H_col"], "v_proj": ["H_row", "H_col"]}
C_INIT = {"gaussian": "black", "laplace": "#B2182B", "uniform": "#2166AC"}
INIT_LABEL = {"gaussian": "Gaussian", "laplace": "Laplace", "uniform": "uniform"}

def med(init, comp, step, q):
    v = [r[q] for r in rows if r["family"] == init and r["comp"] == comp and r["step"] == step]
    assert len(v) == len(SEEDS) * 6, (init, comp, step, len(v))
    return float(st.median(v))

traj = {f"{i}|{c}|{q}": [med(i, c, s, q) for s in STEPS] for i in INITS for c in COMPS for q in ("k_bi", "k_raw", "H_row", "H_col")}

def _sx(v):
    v = np.asarray(v, dtype=float)
    return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))



GROUPS = [("q/k", ["q_proj", "k_proj"]), ("v/o", ["v_proj", "o_proj"]), ("gate / up / down", ["gate_proj", "up_proj", "down_proj"])]
C_QK, C_VO, C_FFN = "#B2182B", "#2166AC", "#1B7837"
KCOL = {"q_proj": C_QK, "k_proj": C_QK, "v_proj": C_VO, "o_proj": C_VO, "gate_proj": C_FFN, "up_proj": C_FFN, "down_proj": C_FFN}
MRK = {"q_proj": "o", "k_proj": "^", "v_proj": "o", "o_proj": "^", "gate_proj": "o", "up_proj": "^", "down_proj": "s"}
LS_INIT = {"gaussian": "-", "laplace": (0, (4, 2)), "uniform": (0, (1, 1.6))}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14.5, "axes.labelsize": 13.5})
fig, axes = plt.subplots(3, 3, figsize=(13.4, 10.2), sharex=True, sharey="row", gridspec_kw=dict(wspace=0.10, hspace=0.20, top=0.815, bottom=0.07, left=0.075, right=0.985))
xt = [0, 800, 3200, 10000, 30000]; xl = ["0", "800", "3.2k", "10k", "30k"]
def line(ax, c, q, i, hollow=False):
    col = KCOL[c]
    ax.plot(_sx(STEPS), traj[f"{i}|{c}|{q}"], color=col, lw=0.9, ls=LS_INIT[i], zorder=2)
    ax.plot(_sx(STEPS), traj[f"{i}|{c}|{q}"], ls="none", marker=MRK[c], ms=5.2, mfc="white" if hollow else col, mec=col, mew=1.0, zorder=4)
for j, (gname, comps) in enumerate(GROUPS):
    for c in comps:
        for i in INITS:
            line(axes[0, j], c, "k_raw", i)
            line(axes[2, j], c, "k_bi", i)
            for axn in IDAXIS[c]:
                line(axes[1, j], c, axn, i, hollow=(axn == "H_col"))
    for r in (0, 2):
        axes[r, j].axhline(K0, color="0.6", lw=0.9, ls=(0, (4, 3)), zorder=1)
    axes[0, j].set_title(f"({'abc'[j]})  {gname}", loc="left")
    axes[2, j].set_xticks(_sx(xt)); axes[2, j].set_xticklabels(xl)
    for r in range(3):
        axes[r, j].spines[["top", "right"]].set_visible(False)
        if j > 0: axes[r, j].tick_params(labelleft=False)
        axes[r, j].grid(alpha=0.18, lw=0.5)
        axes[r, j].tick_params(labelsize=12)
axes[2, 1].set_xlabel("Training step  (linear to 800, log above)")
kb_all = [v for k, v in traj.items() if k.endswith("|k_bi")]; kr_all = [v for k, v in traj.items() if k.endswith("|k_raw")]
lo = min(min(v) for v in kb_all + kr_all) - 0.02; hi = max(max(v) for v in kb_all + kr_all) + 0.02
axes[0, 0].set_ylim(lo, hi); axes[2, 0].set_ylim(lo, hi)
h_all = [v for k, v in traj.items() if k.endswith("|H_row") or k.endswith("|H_col")]
axes[1, 0].set_ylim(0.0, max(max(v) for v in h_all) * 1.08)
axes[0, 0].set_ylabel(r"pooled  $k_{\mathrm{raw}}$")
axes[1, 0].set_ylabel(r"identity-axis width  $H^W$")
axes[2, 0].set_ylabel(r"two-sided core  $k_{\mathrm{bi}}$")
# one legend strip on top: kinds | axis | initialization
hk = [Line2D([], [], ls="none", marker=MRK[c], color=KCOL[c], ms=6, label=LABEL[c]) for c in COMPS]
hx = [Line2D([], [], ls="none", marker="o", color="0.3", ms=6, label="row field"),
      Line2D([], [], ls="none", marker="o", mfc="white", mec="0.3", mew=1.1, ms=6, label="column field")]
hi_ = [Line2D([], [], color="0.3", lw=1.2, ls=LS_INIT[i], label=INIT_LABEL[i]) for i in INITS]
hr = [Line2D([], [], color="0.6", lw=0.9, ls=(0, (4, 3)), label=r"reference $k \approx$ %.3f (top and bottom rows)" % K0),
      Line2D([], [], color="none", label=f"lines: family median of {len(SEEDS) * 6} matrices per point")]
l1 = fig.legend(handles=hk, loc="upper left", bbox_to_anchor=(0.075, 0.998), ncol=7, frameon=False, fontsize=11.5, handlelength=1.2, columnspacing=1.2, title="kind", title_fontsize=11.5)
l2 = fig.legend(handles=hx, loc="upper left", bbox_to_anchor=(0.075, 0.935), ncol=2, frameon=False, fontsize=11.5, handlelength=1.2, columnspacing=1.2, title="axis (middle row)", title_fontsize=11.5)
l3 = fig.legend(handles=hi_, loc="upper left", bbox_to_anchor=(0.34, 0.935), ncol=3, frameon=False, fontsize=11.5, handlelength=2.4, columnspacing=1.2, title="initialization", title_fontsize=11.5)
l4 = fig.legend(handles=hr, loc="upper left", bbox_to_anchor=(0.66, 0.935), ncol=1, frameon=False, fontsize=11.5, handlelength=2.4, labelspacing=0.3)
for l in (l1, l2, l3, l4): l._legend_box.align = "left"
fig.savefig(OUT, dpi=220, facecolor="white")
print("wrote", OUT.relative_to(ROOT), "K0", K0)
