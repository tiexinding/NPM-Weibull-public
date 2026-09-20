"""Paper#5 Figure 6 v3 (260917): v2 + every panel carries its own legend in a headroom band at the top
(y-limit raised, ticks kept to the data range; 老丁 legend-headroom rule). Base: v2 (260916): how the scale fields evolve and the pooled shape follows.

Reads ONLY stored JSON (no checkpoint access, no recomputation):
  08_paper5_draft/data/p5v3_fig4_field_evolution_v1.json   rows (medians per kind per step, already gated)
  08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json      cross-check of steps / matrix count
Writes:
  08_paper5_draft/p5_compile/figure_v2/figures/P5v3_F4_field_evolution_v3.png

Changes vs v1 (老丁 260916 审查清单):
  * three-line in-figure note and the script-sha footer removed (caption / Appendix A).
  * panel titles shortened to neutral labels; in-panel annotations ("▼ = peak", "back on its own ...",
    "through-origin slope ...", "each curve runs from ...") removed (caption).
  * step axis: linear to 800, logarithmic above (same convention as Figure 2(c)), no "step 0 at left edge" hack.
  * kind encoding: colour = functional path (q/k red, v/o blue, FFN green), marker = member of the path
    (circle / triangle / square); line style = regime read off panel (a). One legend lists every kind with
    its own colour+marker; a second legend gives the three line styles.
  * lines thinner (1.3), markers small.
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
OUT = HERE.parents[1] / "figures" / "P5v3_F4_field_evolution_v3.png"
J = json.load(open(DRAFT / "data/p5v3_fig4_field_evolution_v1.json"))
SRC = json.load(open(DRAFT / "data/p5v2_ea_kbi_trajectory_v1.json"))

# ----------------------------- asserts -----------------------------
assert J["schema"] == "p5v3-fig4-field-evolution-v1" and J["family"] == "gaussian", (J["schema"], J["family"])
STEPS = J["steps"]
assert STEPS == SRC["steps"] == [0, 800, 3200, 10000, 20000, 25000, 30000], STEPS
assert SRC["n_matrices"] == 2646 and SRC["fig1b_crosscheck_max_abs_dev"] == 0.0
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
assert set(J["rows"]) == set(KINDS), set(J["rows"])
for k in KINDS:
    R = J["rows"][k]
    assert len(R["H"]) == len(R["k_raw"]) == len(R["k_ident"]) == len(R["dmix"]) == len(STEPS), k
    assert min(R["n"]) >= 12 and R["n_dropped"] == 0, (k, R["n"], R["n_dropped"])
    assert R["group"] in ("accumulate", "partial", "fall back"), R["group"]
SIXPI = float(J["six_over_pi2"]); assert abs(SIXPI - 6 / np.pi ** 2) < 1e-12

SH = {k: k.replace("_proj", "") for k in KINDS}
C_QK, C_VO, C_FFN = "#B2182B", "#2166AC", "#1B7837"
COL = {"q": C_QK, "k": C_QK, "v": C_VO, "o": C_VO, "gate": C_FFN, "up": C_FFN, "down": C_FFN}
MRK = {"q": "o", "k": "^", "v": "o", "o": "^", "gate": "o", "up": "^", "down": "s"}
LS = {"accumulate": "-", "partial": (0, (4, 2)), "fall back": (0, (1.2, 1.4))}
GL = {"accumulate": "keeps widening", "partial": "recedes partly", "fall back": "recedes"}

def kind_handles():
    """Same seven entries in every panel: colour = path, marker = member, line style = trend from (a).
    Column-wise fill with ncol=4 groups them by path: (q,k) (v,o) (gate,up) (down)."""
    return [Line2D([], [], ls=LS[J["rows"][k]["group"]], marker=MRK[SH[k]], color=COL[SH[k]], lw=1.3, ms=4.5,
                   mec="white", mew=0.4, label=LBL[SH[k]]) for k in KINDS]

def two_legends(ax, extras, extra_title=None):
    l1 = ax.legend(handles=kind_handles(), loc="upper left", ncol=4, fontsize=8, frameon=False,
                   handletextpad=0.4, handlelength=2.4, columnspacing=1.0, labelspacing=0.3, borderpad=0.2,
                   bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(l1)
    ax.legend(handles=extras, loc="upper left", ncol=len(extras), fontsize=8, frameon=False, title=extra_title,
              title_fontsize=8, handletextpad=0.4, handlelength=2.4, columnspacing=1.2, borderpad=0.2,
              bbox_to_anchor=(0.0, 0.875))
LBL = {"q": "$q$", "k": "$k$", "v": "$v$", "o": "$o$", "gate": "gate", "up": "up", "down": "down"}

def sx(v):  # linear to 800, log above (Figure 2(c) convention)
    v = np.asarray(v, dtype=float)
    return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))
X = sx(STEPS)
XT = [0, 800, 3200, 10000, 30000]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10})
fig = plt.figure(figsize=(14.6, 4.6))
gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 0.20, 1.0, 0.20, 1.0], wspace=0.0,
                      left=0.045, right=0.985, top=0.91, bottom=0.13)

# ---- (a) field width over training
ax = fig.add_subplot(gs[0, 0])
for k in KINDS:
    s = SH[k]; R = J["rows"][k]
    ax.plot(X, R["H"], ls=LS[R["group"]], marker=MRK[s], ms=3.6, lw=1.3, color=COL[s], mec="white", mew=0.4, zorder=3)
    ip = STEPS.index(R["H_peak_step"])
    ax.plot([X[ip]], [R["H_peak"]], "v", ms=7, color=COL[s], mec="k", mew=0.5, zorder=5)
ax.set_xticks(sx(XT)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
ax.set_xlim(sx(0) - 0.04, sx(30000) + 0.06)
ax.set_xlabel("Training step"); ax.set_ylabel(r"Field width  $H^W$")
ax.set_title("(a)  Field width over training", loc="left")
ax.grid(alpha=0.2, lw=0.5)
ax.set_ylim(0.02, 0.345); ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25])
two_legends(ax, [Line2D([], [], ls="none", marker="v", mfc="0.6", mec="k", mew=0.5, ms=6, label="peak of the curve")])

# ---- (b) shape departure
ax = fig.add_subplot(gs[0, 2])
ax.axhline(0, color="0.4", lw=0.9, zorder=1)
for k in KINDS:
    s = SH[k]; R = J["rows"][k]
    dep = [kr - ki for kr, ki in zip(R["k_raw"], R["k_ident"])]
    ax.plot(X, dep, ls=LS[R["group"]], marker=MRK[s], ms=3.6, lw=1.3, color=COL[s], mec="white", mew=0.4, zorder=3)
ax.set_xticks(sx(XT)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
ax.set_xlim(sx(0) - 0.04, sx(30000) + 0.06)
ax.set_xlabel("Training step"); ax.set_ylabel(r"$k_{\mathrm{raw}} - k_{\mathrm{ident}}$")
ax.set_title("(b)  Pooled-shape departure", loc="left")
ax.grid(alpha=0.2, lw=0.5)
ax.set_ylim(-0.0325, 0.0135); ax.set_yticks([0.0, -0.01, -0.02, -0.03])
two_legends(ax, [Line2D([], [], color="0.3", ls=LS[g], lw=1.3, label=GL[g]) for g in ("accumulate", "partial", "fall back")],
            extra_title=None)

# ---- (c) bridge along the trajectory
ax = fig.add_subplot(gs[0, 4])
hmax = max(max(J["rows"][k]["H"]) for k in KINDS) ** 2 * 1.06
xs = np.linspace(0, hmax, 50)
ax.plot(xs, SIXPI * xs, color="0.5", lw=1.1, ls=(0, (5, 3)), zorder=1)
for k in KINDS:
    s = SH[k]; R = J["rows"][k]
    h2 = np.array(R["H"]) ** 2
    ax.plot(h2, R["dmix"], ls=LS[R["group"]], marker=MRK[s], ms=3.4, lw=1.3, color=COL[s], mec="white", mew=0.4, zorder=3)
    ax.annotate("", xy=(h2[-1], R["dmix"][-1]), xytext=(h2[-2], R["dmix"][-2]),
                arrowprops=dict(arrowstyle="-|>", color=COL[s], lw=1.2), zorder=4)
ax.set_xlabel(r"$(H^W)^2$"); ax.set_ylabel(r"$\Delta_k^{\mathrm{mix}} = k_{\mathrm{raw}}^{-2} - k_{\mathrm{ident}}^{-2}$")
ax.set_title("(c)  Bridge along the trajectory", loc="left")
ax.grid(alpha=0.2, lw=0.5)
ax.set_ylim(-0.002, 0.054); ax.set_yticks([0.0, 0.01, 0.02, 0.03])
two_legends(ax, [Line2D([], [], color="0.5", lw=1.1, ls=(0, (5, 3)), label=r"slope $6/\pi^2$ (reference)"),
                 Line2D([], [], color="0.3", lw=1.2, marker=">", ms=5, ls="-", label="arrowhead = step 30,000")])

for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", OUT)

# ----------------------------- read-outs for the caption -----------------------------
for k in KINDS:
    R = J["rows"][k]
    print(f"{SH[k]:5s} {R['group']:10s} H0 {R['H'][0]:.3f} H800 {R['H'][1]:.3f} peak@{R['H_peak_step']:>5d} {R['H_peak']:.3f} "
          f"final {R['H_final']:.3f} ratio {R['peak_over_final']:.2f} dep_final {R['k_raw_final']-R['k_ident_final']:+.4f} "
          f"alpha {R['alpha_through_origin']:.3f} R2_0 {R['R2_0']:.3f}")
al = [J["rows"][k]["alpha_through_origin"] for k in KINDS]; r2 = [J["rows"][k]["R2_0"] for k in KINDS]
print("alpha range %.2f-%.2f, R2_0 min %.3f" % (min(al), max(al), min(r2)))
