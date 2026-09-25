"""Paper#5 Figure 2 v4 (260923): legends moved to the top of each panel with raised y limits (house style of Figs 4-8); explanatory text moved from the caption into the panels; psi notation removed from the y label.
v3 (260921): (b) fitting window drawn as two vertical dashed lines instead of a grey band (老丁). v2 (260916): a common normalized core, re-drawn for legibility.

Reads ONLY the published read-out JSONs (no checkpoint access):
  08_paper5_draft/data/p5v2_figure_readouts_v1.json   key "fig1_v3"  (panels a, c)
  08_paper5_draft/data/p5v3_core_quantile_profile_v1.json           (panel b)
Writes:
  08_paper5_draft/p5_compile/figure_v2/figures/P5v3_F1_normalized_core_v3.png

Changes vs v1 (老丁 260916):
  (a) x = model size only; the six kinds are offset within each size; colour = functional path
      (q/k red, v/o blue, FFN green), marker = member of the path (circle / triangle), filled = raw,
      open = identity-axis normalized, star = v after both axes. Thin grey connector raw->normalized,
      light whisker = block range of the normalized fit. All explanatory text moved to the caption.
  (b) time encoded by alpha only (step 0 dashed, 30k slightly heavier), same line width otherwise;
      family colours identical to (c); one compact legend each; no in-panel annotation text.
  (c) k_bi is the visual subject: coloured solid + markers, top z-order; raw k thin dashed same colour;
      Gaussian black; per-seed spread as a translucent band; reference as light dashed line labelled
      at the right end.
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
DRAFT = HERE.parents[3]                     # 08_paper5_draft
OUT = HERE.parents[1] / "figures" / "P5v3_F1_normalized_core_v5.png"
R = json.load(open(DRAFT / "data/p5v2_figure_readouts_v1.json"))["fig1_v3"]
QP = json.load(open(DRAFT / "data/p5v3_core_quantile_profile_v1.json"))

# ----------------------------- shared encodings -----------------------------
C_QK, C_VO, C_FFN = "#B2182B", "#2166AC", "#1B7837"
C_INIT = {"gaussian": "black", "laplace": C_QK, "uniform": C_VO}
INIT_LABEL = {"gaussian": "Gaussian", "laplace": "Laplace", "uniform": "uniform"}

# ----------------------------- data + asserts -----------------------------
A = R["a_public_pythia_endpoints"]
assert A["n"] == 348, A["n"]
SIZES = ["70m", "160m", "410m", "1b"]
SIZE_LABEL = {"70m": "70M", "160m": "160M", "410m": "410M", "1b": "1B"}
KINDS = ["q", "k", "v", "o", "ffn_in", "ffn_out"]
KCOL = {"q": C_QK, "k": C_QK, "v": C_VO, "o": C_VO, "ffn_in": C_FFN, "ffn_out": C_FFN}
KMRK = {"q": "o", "k": "^", "v": "o", "o": "^", "ffn_in": "o", "ffn_out": "^"}
assert set(A["per_size"]) == set(SIZES) and all(set(A["per_size"][s]) == set(KINDS) for s in SIZES)
K_REF = float(R["reference_constant_k_gauss_init"])

C = R["c_init_families"]
STEPS = C["ckpt_steps"]; INITS = list(C["family_median_k_bi_all_kinds"])
assert STEPS == [0, 800, 3200, 10000, 20000, 25000, 30000] and set(INITS) == set(C_INIT), (STEPS, INITS)
N_BELOW_GATE = int(C["n_fits_below_r2_gate"])   # retained and flagged per protocol (Appendix A); 133 in the source
K0 = float(C["k0_reference"])

P = np.array(QP["p_levels"]); BAND = QP["band"]; QSTEPS = QP["steps"]
assert BAND == [0.10, 0.90] and QSTEPS == [0, 800, 3200, 30000] and set(QP["inits"]) == set(C_INIT)
assert QP["all_final"]["n"] == 378

# ----------------------------- figure -----------------------------
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5})
fig = plt.figure(figsize=(15.4, 5.6))
# explicit narrow gap columns: col 1 = room for (b)'s y label; col 3 = (b)'s legends + (c)'s y label
gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.22, 1.0, 0.22, 1.0], wspace=0.0,
                      left=0.042, right=0.99, top=0.91, bottom=0.12)

# ---- (a) public checkpoints: size x kind
ax = fig.add_subplot(gs[0, 0])
OFF = np.linspace(-0.33, 0.33, len(KINDS))
for i, s in enumerate(SIZES):
    for j, kd in enumerate(KINDS):
        e = A["per_size"][s][kd]; x = i + OFF[j]; c = KCOL[kd]; m = KMRK[kd]
        ax.plot([x, x], [e["k_raw_median"], e["k_axis_median"]], color="0.75", lw=0.9, zorder=1)
        ax.plot([x, x], e["k_axis_range"], color=c, lw=2.6, alpha=0.18, solid_capstyle="butt", zorder=2)
        ax.scatter(x, e["k_raw_median"], marker=m, s=26, color=c, zorder=4)
        ax.scatter(x, e["k_axis_median"], marker=m, s=30, facecolor="white", edgecolor=c, lw=1.3, zorder=5)
        if kd == "v":
            ax.scatter(x, e["k_both_median"], marker="*", s=70, facecolor="white", edgecolor=c, lw=1.1, zorder=6)
    if i < len(SIZES) - 1:
        ax.axvline(i + 0.5, color="0.90", lw=0.8, zorder=0)
ax.axhline(K0, color="0.6", lw=0.9, ls=(0, (4, 3)), zorder=1)   # protocol reference 1.204, same line as (c)
ax.annotate("1.204", xy=(0.995, K0), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points", va="bottom", ha="right", fontsize=10.5, color="0.4")
ax.text(0.99, 0.01, "4 sizes, 348 matrices", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.6, color="0.45")
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels([SIZE_LABEL[s] for s in SIZES])
ax.set_xlim(-0.55, len(SIZES) - 0.45); ax.set_ylim(1.03, 1.36); ax.set_yticks([1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
ax.set_xlabel("Pythia model size"); ax.set_ylabel("Shape  $k$")
ax.set_title("(a)  Across public Pythia sizes", loc="left")
KLAB = {"q": "$q$", "k": "$k$", "v": "$v$", "o": "$o$", "ffn_in": "up", "ffn_out": "down"}
ha = [Line2D([], [], ls="none", marker=KMRK[k], color=KCOL[k], ms=5, label=KLAB[k]) for k in KINDS]
hs = [Line2D([], [], ls="none", marker="s", color="0.3", ms=5, label="raw, block median"),
      Line2D([], [], ls="none", marker="s", mfc="white", mec="0.3", mew=1.3, ms=5, label="identity-axis normalized"),
      Line2D([], [], color="0.3", lw=2.6, alpha=0.25, label="block range, normalized"),
      Line2D([], [], ls="none", marker="*", mfc="white", mec="0.3", mew=1.1, ms=8, label="$v$, both axes")]
lga = ax.legend(handles=ha, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=6, fontsize=9.9, frameon=False,
                handlelength=0.8, handletextpad=0.3, columnspacing=0.6, borderpad=0.2)
ax.add_artist(lga)
ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 0.92), ncol=2, fontsize=9.9, frameon=False,
          handlelength=1.2, handletextpad=0.4, columnspacing=0.9, borderpad=0.2)

# ---- (b) quantile profile of the core
ax = fig.add_subplot(gs[0, 2])
ALPHA = {0: 0.55, 800: 0.70, 3200: 0.85, 30000: 1.0}
LW = {0: 1.0, 800: 1.0, 3200: 1.0, 30000: 1.7}
LS = {0: (0, (1.2, 1.4)), 800: (0, (4, 2)), 3200: (0, (5, 1.5, 1, 1.5)), 30000: "-"}
for xb in BAND: ax.axvline(xb, color="0.45", lw=0.9, ls=(0, (4, 2.5)), zorder=1)
ax.axhline(0, color="0.5", lw=0.8, zorder=1)
for it in QP["inits"]:
    for t in QSTEPS:
        ax.plot(P, 100 * np.array(QP["by_init_step"][f"{it}@{t}"]["median"]),
                color=C_INIT[it], lw=LW[t], ls=LS[t], alpha=ALPHA[t], zorder=3 + (t == 30000))
ax.set_yscale("symlog", linthresh=0.5, linscale=0.55)
ax.set_ylim(-150, 900)
ax.set_yticks([-40, -10, -1, 0, 1, 10, 40]); ax.set_yticklabels(["−40", "−10", "−1", "0", "1", "10", "40"])
ax.set_xlabel("Quantile $p$"); ax.set_ylabel("Core quantile / reference $-$ 1  (%, symlog)")
ax.set_title("(b)  Quantile profile of the core", loc="left")
ax.grid(alpha=0.2, lw=0.5)
h1 = [Line2D([], [], color=C_INIT[i], lw=2.0, label=INIT_LABEL[i]) for i in QP["inits"]]
STEP_LABEL = {0: "0", 800: "800", 3200: "3.2k", 30000: "30k"}
h2 = [Line2D([], [], color="0.2", lw=LW[t], ls=LS[t], alpha=ALPHA[t], label=STEP_LABEL[t]) for t in QSTEPS]
h1 = [Line2D([], [], color="none", label="family:")] + h1
h2 = [Line2D([], [], color="none", label="step:")] + h2
lgb = ax.legend(handles=h1, fontsize=9.9, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False,
                handlelength=1.3, handletextpad=0.4, columnspacing=0.8, borderpad=0.2)
ax.add_artist(lgb)
ax.legend(handles=h2, fontsize=9.9, loc="upper left", bbox_to_anchor=(0.0, 0.92), ncol=5, frameon=False,
          handlelength=1.5, handletextpad=0.4, columnspacing=0.7, borderpad=0.2)
ax.text(0.02, 0.84, "median over 3 seeds, 6 layers, 7 kinds", transform=ax.transAxes,
        fontsize=9.6, color="0.45", ha="left", va="top")
ax.text(np.mean(BAND), 0.015, "middle-80% fitting window", transform=ax.get_xaxis_transform(),
        fontsize=9.6, color="0.45", ha="center", va="bottom")

# ---- (c) two-sided core over training
ax = fig.add_subplot(gs[0, 4])
def sx(v):  # linear to 800, log above
    v = np.asarray(v, dtype=float)
    return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))
X = sx(STEPS)
for it in INITS:
    seeds = np.array([C["seed_median_k_bi_all_kinds"][k] for k in C["seed_median_k_bi_all_kinds"] if k.startswith(it + "|")])
    assert seeds.shape == (3, len(STEPS)), seeds.shape
    ax.fill_between(X, seeds.min(0), seeds.max(0), color=C_INIT[it], alpha=0.16, lw=0, zorder=2)
    ax.plot(X, C["family_median_k_raw_all_kinds"][it], color=C_INIT[it], lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.plot(X, C["family_median_k_bi_all_kinds"][it], color=C_INIT[it], lw=2.2, marker="o", ms=4.2,
            mec="white", mew=0.7, zorder=6)
ax.axhline(K0, color="0.6", lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.text(sx(30000) + 0.03, K0 - 0.012, f"{K0:.3f}", fontsize=9.9, color="0.45", va="center", ha="left")
xt = [0, 800, 3200, 10000, 30000]
ax.set_xticks(sx(xt)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
ax.set_xlim(sx(0) - 0.05, sx(30000) + 0.22)
ylo = min(min(v) for v in C["family_median_k_raw_all_kinds"].values())
yhi = max(max(max(v) for v in C["family_median_k_bi_all_kinds"].values()),
          max(max(v) for v in C["family_median_k_raw_all_kinds"].values()))
ax.set_ylim(ylo - 0.02, yhi + 0.085)
ax.set_xlabel("Training step  (linear to 800, log above)"); ax.set_ylabel(r"Core shape  $k_{\mathrm{bi}}$")
ax.set_title("(c)  Core convergence across initializations", loc="left")
hc = [Line2D([], [], color=C_INIT[i], lw=2.2, marker="o", ms=4, mec="white", label=INIT_LABEL[i]) for i in ["gaussian", "laplace", "uniform"]]
hc2 = [Line2D([], [], color="0.4", lw=2.2, marker="o", ms=4, mec="white", label=r"$k_{\mathrm{bi}}$, median of 126 matrices"),
       Patch(facecolor="0.4", alpha=0.25, label="range of 3 seed medians"),
       Line2D([], [], color="0.4", lw=1.1, ls=(0, (3, 2)), label="raw $k$")]
lgc = ax.legend(handles=hc, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=9.9, frameon=False,
          handlelength=1.8, handletextpad=0.4, columnspacing=0.8, borderpad=0.2)
ax.add_artist(lgc)
ax.legend(handles=hc2, loc="upper left", bbox_to_anchor=(0.0, 0.92), ncol=2, fontsize=9.9, frameon=False,
          handlelength=1.8, handletextpad=0.4, columnspacing=0.8, borderpad=0.2)

for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=11.5)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
# read-outs used by the caption
norm = [A["per_size"][s][k]["k_axis_median"] for s in SIZES for k in ("q", "k", "o", "ffn_in", "ffn_out")]
both = [A["per_size"][s]["v"]["k_both_median"] for s in SIZES]
print("normalized medians (q,k,o,ffn): %.3f-%.3f" % (min(norm), max(norm)))
print("v both-axes medians: %.3f-%.3f" % (min(both), max(both)))
print("30k median profile max |dev| in band: %.2f%%" % (100 * QP["all_final"]["max_abs_dev_band"]))
print("k0 reference", K0, "K_REF const", K_REF)
