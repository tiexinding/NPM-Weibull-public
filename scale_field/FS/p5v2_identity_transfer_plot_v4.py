"""Paper#5 Fig 4 v4 (260917): identity-axis transfer, three panels with per-panel legends.
Plot-only, reads 08_paper5_draft/data/p5v2_identity_transfer_v1.json.
Changes vs v2: (d) and the D1-D4 arm colours removed from the main figure (arm version = v2 image, appendix);
one colour per path (FFN green / v-o blue / q-k red), pale = individual runs, dark = median;
each panel carries its own legend; (c) annotated with the two medians (row-level vs RoPE-pair level).
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve(); DRAFT = HERE.parents[3]
OUT = HERE.parents[1] / "figures" / "P5v2_S10_identity_transfer_v4.png"
D = json.load(open(DRAFT / "data/p5v2_identity_transfer_v1.json")); assert D["schema"] == "p5v2-identity-transfer-v1"
C_QK, C_VO, C_FFN = "#B2182B", "#2166AC", "#1B7837"
PALE = {C_QK: "#E8A0A4", C_VO: "#9DC3E6", C_FFN: "#9CCf9C"}
RNG = np.random.default_rng(0)
rows = D["rows"]; t5 = [r for r in rows if r["family"] == "tclock" and r["step"] == 5000]
assert len(t5) == 72; LAYERS = sorted(set(r["layer"] for r in t5)); assert LAYERS == list(range(6))
S = D["summary"]["tclock_step5000"]
med_row, med_pair = S["q_row__k_row"]["pearson"]["median"], S["q_pair__k_pair"]["pearson"]["median"]
n_pairs, n_model = t5[0]["n_pairs"], t5[0]["n_model"]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10})
fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3), sharey=True, gridspec_kw=dict(wspace=0.10, left=0.06, right=0.99, top=0.84, bottom=0.14))
def jit(n, w): return RNG.uniform(-w, w, n)

def layer_panel(ax, key, mkey, nullkey, title, color, path_label):
    p95 = float(np.median([r[nullkey] for r in t5]))
    ax.axhspan(-p95, p95, color="0.88", zorder=1)
    for L in LAYERS:
        rr = [r for r in t5 if r["layer"] == L]
        ax.scatter(L + jit(len(rr), .18), [r[key][0] for r in rr], s=18, color=PALE[color], alpha=.9, lw=0, zorder=3)
        ax.scatter(L + jit(len(rr), .18), [r[mkey][0] for r in rr], s=14, color="0.5", alpha=.75, marker="x", zorder=2)
    ax.plot(LAYERS, [np.median([r[key][0] for r in t5 if r["layer"] == L]) for L in LAYERS], color=color, lw=2.0, zorder=4)
    ax.axhline(0, color="0.4", lw=.7); ax.set_xticks(LAYERS); ax.set_xlabel("Layer"); ax.set_ylim(-0.27, 1.36); ax.set_yticks([-0.2,0,0.2,0.4,0.6,0.8,1.0])
    ax.set_title(title, loc="left", pad=8)
    h = [Line2D([], [], marker="o", ls="", color=PALE[color], ms=5.5, label="matched profiles"),
         Line2D([], [], color=color, lw=2, label="median"),
         Line2D([], [], marker="x", ls="", color="0.5", ms=5.5, label="next-layer mismatch"),
         Patch(facecolor="0.88", label="within-layer shuffle null, 95%")]
    ax.legend(handles=h, fontsize=7.6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.2, handlelength=1.4, handletextpad=.5, labelspacing=.5)

layer_panel(axes[0], "gate_row__down_col", "mismatch_gate_row__down_col_next_layer", "null_gate_down_p95_abs",
            "(a)  FFN hidden identity\n      gate rows vs down columns", C_FFN, "FFN")
axes[0].set_ylabel("Profile correlation  $r$")
layer_panel(axes[1], "v_row__o_col", "mismatch_v_row__o_col_next_layer", "null_v_o_p95_abs",
            "(b)  Head/value identity\n      $v$ rows vs $o$ columns", C_VO, "vo")

ax = axes[2]
for L in LAYERS:
    rr = [r for r in t5 if r["layer"] == L]
    ax.scatter(L - .12 + jit(len(rr), .07), [r["q_row__k_row"][0] for r in rr], s=18, color=PALE[C_QK], alpha=.9, lw=0, zorder=3)
    ax.scatter(L + .12 + jit(len(rr), .07), [r["q_pair__k_pair"][0] for r in rr], s=20, marker="s", facecolors="none", edgecolors=C_QK, alpha=.85, zorder=3)
ax.plot(LAYERS, [np.median([r["q_row__k_row"][0] for r in t5 if r["layer"] == L]) for L in LAYERS], color=C_QK, lw=2.0, zorder=4)
ax.plot(LAYERS, [np.median([r["q_pair__k_pair"][0] for r in t5 if r["layer"] == L]) for L in LAYERS], color=C_QK, lw=1.6, ls=(0, (4, 2)), zorder=4)
ax.axhline(0, color="0.4", lw=.7); ax.set_xticks(LAYERS); ax.set_xlabel("Layer"); ax.set_ylim(-0.27, 1.36); ax.set_yticks([-0.2,0,0.2,0.4,0.6,0.8,1.0])
ax.set_title("(c)  RoPE identity in $q/k$\n      $q$ rows vs $k$ rows", loc="left", pad=8)
h = [Line2D([], [], marker="o", ls="", color=PALE[C_QK], ms=5.5, label="per row"),
     Line2D([], [], color=C_QK, lw=2, label="median, per row"),
     Line2D([], [], marker="s", ls="", mfc="none", color=C_QK, ms=5.5, label="per RoPE pair"),
     Line2D([], [], color=C_QK, lw=1.6, ls=(0, (4, 2)), label="median, per pair")]
ax.legend(handles=h, fontsize=7.6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.2, handlelength=1.4, handletextpad=.5, labelspacing=.5)

for a in axes:
    a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8.6)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white"); print("wrote", OUT)
print("caption read-outs: gate-down", S["gate_row__down_col"]["pearson"], "| v-o", S["v_row__o_col"]["pearson"],
      "| q-k row", S["q_row__k_row"]["pearson"], "| pair", S["q_pair__k_pair"]["pearson"],
      "| null", S["null_gate_down_p95_abs_median"], S["null_v_o_p95_abs_median"])
