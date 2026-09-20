"""Paper#5 v2 Figure 1: shape is inherited, scale is learned.

  (c) [260911 order] TCLOCK-D, 1,440 matrix fits: raw-fit k and row-normalized refit k_norm against the
      row-scale heterogeneity H^W = sd_i(log r_i), coloured by component family; the
      Gaussian-init reference k = 1.20 is drawn as a horizontal line
  (b) E-A runs: 3 initialization families (Gaussian/Laplace/uniform) x 3 seeds, llama-70M, 30k steps:
      two-sided core k_bi (median over 7 kinds x 6 layers) and raw k against training step [260911 late: replaced the q/k k_norm panel]
      [260911: replaced the single IVN 30k-run panel (kept as deprecated_*ivn30kB*.png), which had
      replaced the Pythia size-run box panel on 260910]
  (a) [260911 order] public Pythia 70m/160m/410m/1b endpoints, 348 matrices: terminal raw k and the
      identity-axis-normalized refit per kind (rows for q/k/v/ffn_in, columns for o/ffn_out);
      v additionally after row+column normalization  [refit added 260910]

Sources (read-only):
  06_hrow/kmix_bridge_v1_1.json                        partA_audit_rows
  08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json   (scripts/p5v2_ea_kbi_trajectory_v1.py)
  08_paper5_draft/data/p5_public_pythia_knorm_v1.json      rows (scripts/p5v2_public_knorm_refit_v1.py)
Outputs:
  08_paper5_draft/figures/P5v2_F1_shape_inherited.png
  08_paper5_draft/data/p5v2_figure_readouts_v1.json   (key "fig1")
"""
import json
import statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "08_paper5_draft/figures/P5v2_F1_shape_inherited.png"
READ = ROOT / "08_paper5_draft/data/p5v2_figure_readouts_v1.json"

K_GAUSS_INIT = 1.20           # reference constant: Weibull shape of a Gaussian init family
SEL = ["q_proj", "k_proj"]
TRA = ["v_proj", "gate_proj", "up_proj"]
FAM = SEL + TRA
LABEL = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "gate_proj": "gate", "up_proj": "up"}
C_SEL, C_TRA = "#B2182B", "#2166AC"
COL = {f: (C_SEL if f in SEL else C_TRA) for f in FAM}
MRK = {"q_proj": "o", "k_proj": "s", "v_proj": "^", "gate_proj": "D", "up_proj": "v"}
C_RAW, C_NORM = "#404040", "#D95F02"


def _rng(v):
    return [float(min(v)), float(max(v))]


# =============================== (a) TCLOCK ===============================
A = json.load(open(ROOT / "06_hrow/kmix_bridge_v1_1.json"))
rows = A["partA_audit_rows"]
assert len(rows) == 1440, len(rows)
assert set(r["kind"] for r in rows) == set(FAM)
assert min(min(r["r2fit_raw"], r["r2fit_norm"]) for r in rows) >= 0.99, "fit gate"
assert len(set(r["arm"] for r in rows)) == 4 and len(set(r["seed"] for r in rows)) == 3
assert len(set(r["step"] for r in rows)) == 4

fig1a = {"n": len(rows), "per_family": {}}
for f in FAM:
    s = [r for r in rows if r["kind"] == f]
    fig1a["per_family"][f] = {
        "n": len(s),
        "k_raw_range": _rng([r["k_raw"] for r in s]),
        "k_raw_median": float(st.median(r["k_raw"] for r in s)),
        "k_norm_range": _rng([r["k_norm"] for r in s]),
        "k_norm_median": float(st.median(r["k_norm"] for r in s)),
        "hw_range": _rng([r["hw"] for r in s]),
        "hw_median": float(st.median(r["hw"] for r in s)),
        "corr_k_raw_vs_hw": float(np.corrcoef([r["hw"] for r in s], [r["k_raw"] for r in s])[0, 1]),
        "corr_k_norm_vs_hw": float(np.corrcoef([r["hw"] for r in s], [r["k_norm"] for r in s])[0, 1]),
    }
fig1a["all"] = {
    "k_raw_range": _rng([r["k_raw"] for r in rows]),
    "k_norm_range": _rng([r["k_norm"] for r in rows]),
    "k_norm_median": float(st.median(r["k_norm"] for r in rows)),
    "k_norm_dev_from_1p20_max_pct": float(100 * max(abs(r["k_norm"] - K_GAUSS_INIT) for r in rows) / K_GAUSS_INIT),
    "hw_range": _rng([r["hw"] for r in rows]),
    "corr_k_raw_vs_hw": float(np.corrcoef([r["hw"] for r in rows], [r["k_raw"] for r in rows])[0, 1]),
    "corr_k_norm_vs_hw": float(np.corrcoef([r["hw"] for r in rows], [r["k_norm"] for r in rows])[0, 1]),
}

# =============================== (b) one 30k run: k_norm vs step ===============================
B = json.load(open(ROOT / "08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json"))
INITS = list(B["inits"]); SEEDS = [str(x) for x in B["seeds"]]; STEPS_B = list(B["steps"]); KINDS_B = list(B["families"])
PM = B["per_matrix"]
assert len(PM) == 3 * 3 * 7 * 7 * 6 == 2646 and len(STEPS_B) == 7 and len(KINDS_B) == 7, (len(PM), STEPS_B, KINDS_B)
K_REF_B = float(B["k0_gaussian_reference"])
def _med(init, step, key, seed=None):
    v = [r[key] for r in PM if r["family"] == init and r["step"] == step and (seed is None or str(r["seed"]) == seed)]
    assert len(v) == (42 if seed is not None else 126), (init, step, seed, len(v))
    return float(st.median(v))
fig1b = {"source": "E-A runs, llama-70M, 3 init families x 3 seeds, 7 kinds x 6 layers, 30k steps (04_ea_runs/ckpt_key); two-sided core k_bi per matrix",
         "ckpt_steps": STEPS_B, "kinds": KINDS_B, "k0_reference": K_REF_B, "n_fits_below_r2_gate": B["n_fits_below_r2_gate"],
         "family_median_k_bi_all_kinds": {i: [_med(i, t, "k_bi") for t in STEPS_B] for i in INITS},
         "seed_median_k_bi_all_kinds": {f"{i}|s{sd}": [_med(i, t, "k_bi", sd) for t in STEPS_B] for i in INITS for sd in SEEDS},
         "family_median_k_raw_all_kinds": {i: [_med(i, t, "k_raw") for t in STEPS_B] for i in INITS},
         "step0_family_median_k_bi": {i: _med(i, 0, "k_bi") for i in INITS},
         "merge_steps_k_bi_per_kind": B["merge_steps_k_bi"],
         "step30000_per_matrix_range_k_bi": {i: _rng([r["k_bi"] for r in PM if r["family"] == i and r["step"] == 30000]) for i in INITS},
         "gaussian_k_bi_range_all_steps": _rng([r["k_bi"] for r in PM if r["family"] == "gaussian"])}

# =============================== (c) public Pythia endpoints, identity-axis refit ===============================
P = json.load(open(ROOT / "08_paper5_draft/data/p5_public_pythia_knorm_v1.json"))
erows = P["rows"]
KINDS_C = ["q", "k", "v", "o", "ffn_in", "ffn_out"]
SIZES_C = ["70m", "160m", "410m", "1b"]
assert len(erows) == 348, len(erows)  # 36 + 72 + 144 + 96
assert set(r["size"] for r in erows) == set(SIZES_C) and set(r["kind"] for r in erows) == set(KINDS_C)
assert P["phase1b_kf_absdiff_median_max"][1] < 1e-6, "k_raw must reproduce the phase1b endpoint record"
below_gate = [{"size": r["size"], "block": r["block"], "kind": r["kind"], "r2_raw": r["r2_raw"], "k_raw": r["k_raw"]}
              for r in erows if r["r2_raw"] < 0.99]
assert len(below_gate) == 2 and all(b["kind"] == "q" for b in below_gate), below_gate
assert min(min(r["r2_axis"], r["r2_both"]) for r in erows) >= 0.99, "normalized fits pass the gate"
fig1c = {"n": len(erows), "n_r2_raw_below_0.99": len(below_gate), "below_gate_matrices": below_gate, "per_size": {}}
for s in SIZES_C:
    e = {}
    for kd in KINDS_C:
        sub = [r for r in erows if r["size"] == s and r["kind"] == kd]
        e[kd] = {"n": len(sub), "identity_axis": sub[0]["identity_axis"],
                 "k_raw_median": float(st.median(r["k_raw"] for r in sub)), "k_raw_range": _rng([r["k_raw"] for r in sub]),
                 "k_axis_median": float(st.median(r["k_axis"] for r in sub)), "k_axis_range": _rng([r["k_axis"] for r in sub]),
                 "k_both_median": float(st.median(r["k_both"] for r in sub)), "k_both_range": _rng([r["k_both"] for r in sub]),
                 "H_row_median": float(st.median(r["H_row"] for r in sub)), "H_col_median": float(st.median(r["H_col"] for r in sub))}
    fig1c["per_size"][s] = e
for s in SIZES_C:  # Selection raw below Transmission raw; axis-normalized medians near the reference
    assert max(fig1c["per_size"][s][k]["k_raw_median"] for k in ("q", "k")) < min(fig1c["per_size"][s][k]["k_raw_median"] for k in ("o", "ffn_in", "ffn_out"))
    assert all(abs(fig1c["per_size"][s][k]["k_axis_median"] - K_GAUSS_INIT) < 0.012 for k in ("q", "k", "o", "ffn_in", "ffn_out")), s
    assert abs(fig1c["per_size"][s]["v"]["k_both_median"] - K_GAUSS_INIT) < 0.012, s

# =============================== figure ===============================
fig = plt.figure(figsize=(13.4, 4.15))
gs = fig.add_gridspec(1, 3, width_ratios=[1.04, 1.04, 1.22], wspace=0.28,
                      left=0.05, right=0.99, top=0.87, bottom=0.17)

# ---- (c) grid [was (a)]
ax = fig.add_subplot(gs[0, 2])
for f in FAM:
    s = [r for r in rows if r["kind"] == f]
    x = np.array([r["hw"] for r in s])
    ax.scatter(x, [r["k_raw"] for r in s], s=11, marker=MRK[f], facecolor=COL[f],
               edgecolor="none", alpha=0.45, zorder=3)
    ax.scatter(x, [r["k_norm"] for r in s], s=12, marker=MRK[f], facecolor="none",
               edgecolor=COL[f], linewidth=0.6, alpha=0.5, zorder=2)
ax.axhline(K_GAUSS_INIT, color="0.35", lw=1.2, ls=(0, (5, 3)), zorder=1)
ax.text(0.985, 0.955, "n = %d fits, 4 arms x 3 seeds x 4 steps\ndashed: Gaussian-init reference k = %.2f" % (len(rows), K_GAUSS_INIT),
        transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color="0.3")
ax.set_xlabel(r"row-scale heterogeneity  $H^W = \mathrm{sd}_i(\log r_i)$")
ax.set_ylabel("Weibull shape  $k$")
ax.set_xlim(0, fig1a["all"]["hw_range"][1] * 1.04)
ylo = fig1a["all"]["k_raw_range"][0]
ax.set_ylim(ylo - 0.03, fig1a["all"]["k_norm_range"][1] + 0.055)
ax.set_title("(c) Raw and row-normalized shape against $H^W$, 1,440 fits", loc="left",
             fontsize=10.5, pad=7)
h_fill = [Line2D([], [], ls="none", marker=MRK[f], mfc=COL[f], mec="none", ms=5.4,
                 label=LABEL[f]) for f in FAM]
lg1 = ax.legend(handles=h_fill, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=8.3,
                frameon=False, ncol=5, columnspacing=0.8, handletextpad=0.25,
                title=r"filled: raw fit $k_{\mathrm{raw}}$        open: row-normalized refit $k_{\mathrm{norm}}$",
                title_fontsize=8.3, borderpad=0.1)
lg1._legend_box.align = "left"

# ---- (b) init families, two-sided core
ax = fig.add_subplot(gs[0, 1])
C_INIT = {"gaussian": "0.25", "laplace": C_SEL, "uniform": C_TRA}
def _sx(v):  # linear below 800, log above (rendered on a plain axis)
    v = np.asarray(v, dtype=float)
    return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))
for i in INITS:
    for sd in SEEDS:
        ax.plot(_sx(STEPS_B), fig1b["seed_median_k_bi_all_kinds"][f"{i}|s{sd}"], color=C_INIT[i], lw=0.7, alpha=0.55, zorder=3)
    ax.plot(_sx(STEPS_B), fig1b["family_median_k_bi_all_kinds"][i], color=C_INIT[i], lw=2.2, zorder=5, marker="o", ms=3.4, mec="white", mew=0.6)
    ax.plot(_sx(STEPS_B), fig1b["family_median_k_raw_all_kinds"][i], color=C_INIT[i], lw=1.3, ls=(0, (3, 2)), zorder=4)
ax.axhline(K_GAUSS_INIT, color="0.35", lw=1.2, ls=(0, (5, 3)), zorder=1)
xt = [0, 800, 3200, 10000, 30000]
ax.set_xticks(_sx(xt)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"], fontsize=8.6)
ax.set_xlim(_sx(0) - 0.05, _sx(30000) + 0.05)
ax.set_xlabel("training step  (linear to 800, log above)")
ax.set_ylabel("Weibull shape  $k$  (median over 7 kinds x 6 layers)")
ylo = min(min(min(v) for v in fig1b["family_median_k_raw_all_kinds"].values()), min(min(v) for v in fig1b["family_median_k_bi_all_kinds"].values()))
yhi = max(max(v) for v in fig1b["family_median_k_bi_all_kinds"].values())
ax.set_ylim(ylo - 0.02, yhi + 0.03)
ax.set_title("(b) Two-sided core, three initializations, 30,000 steps", loc="left", fontsize=10.5, pad=7)
hb = [Line2D([], [], color=C_INIT["gaussian"], lw=2.2, label="Gaussian init"),
      Line2D([], [], color=C_INIT["laplace"], lw=2.2, label="Laplace init"),
      Line2D([], [], color=C_INIT["uniform"], lw=2.2, label="uniform init"),
      Line2D([], [], color="0.5", lw=2.2, label=r"$k_{\mathrm{bi}}$ (thick: family median; thin: seeds)"),
      Line2D([], [], color="0.5", lw=1.3, ls=(0, (3, 2)), label="raw $k$ (family median)")]
ax.legend(handles=hb, loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=7.6, frameon=False, handlelength=1.8, handletextpad=0.4, borderpad=0.1)
ax.text(0.985, 0.955, "llama-70M, one data setting\n3 init families x 3 seeds, all 42 matrices",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.0, color="0.3")

# ---- (a) public [was (c)]
ax = fig.add_subplot(gs[0, 0])
LABC = {"q": "q", "k": "k", "v": "v", "o": "o", "ffn_in": "ffn in", "ffn_out": "ffn out"}
CCOL = {"q": C_SEL, "k": C_SEL, "v": C_TRA, "o": C_TRA, "ffn_in": C_TRA, "ffn_out": C_TRA}
C_BOTH = "#1B7837"
xt, xl = [], []
for i, s in enumerate(SIZES_C):
    for j, kd in enumerate(KINDS_C):
        x = i * 7 + j
        e = fig1c["per_size"][s][kd]
        ax.plot([x, x], [e["k_raw_median"], e["k_axis_median"]], color="#CCCCCC", lw=1.0, zorder=1)
        ax.plot([x, x], e["k_axis_range"], color=CCOL[kd], lw=0.7, alpha=0.45, zorder=2)
        ax.scatter(x, e["k_raw_median"], marker="o", s=20, color=CCOL[kd], zorder=3)
        ax.scatter(x, e["k_axis_median"], marker="s", s=22, facecolor="white", edgecolor=CCOL[kd], lw=1.3, zorder=4)
        if kd == "v":
            ax.scatter(x, e["k_both_median"], marker="D", s=20, facecolor="white", edgecolor=C_BOTH, lw=1.3, zorder=5)
        xt.append(x); xl.append(LABC[kd])
    ax.text(i * 7 + 2.5, 1.04, "Pythia %s" % s.upper(), ha="center", fontsize=8.8, color="0.25")
    if i < len(SIZES_C) - 1:
        ax.axvline(i * 7 + 6, color="0.88", lw=0.8, zorder=0)
ax.axhline(K_GAUSS_INIT, color="0.35", lw=1.2, ls=(0, (5, 3)), zorder=1)
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7.2, rotation=90)
ax.set_xlim(-1, len(SIZES_C) * 7 - 1)
ax.set_ylim(1.03, 1.30)
ax.set_ylabel("terminal Weibull shape  $k$  (public checkpoints)")
ax.set_title("(a) Public checkpoints at four sizes", loc="left", fontsize=10.5, pad=7)
hc = [Line2D([], [], ls="none", marker="o", color="0.3", ms=5, label="raw $k$ (median over blocks)"),
      Line2D([], [], ls="none", marker="s", mfc="white", mec="0.3", mew=1.3, ms=5, label="identity-axis normalized (thin line: min-max)"),
      Line2D([], [], ls="none", marker="D", mfc="white", mec=C_BOTH, mew=1.3, ms=5, label="row+column normalized ($v$ only)")]
ax.legend(handles=hc, loc="upper left", bbox_to_anchor=(-0.01, 1.01), fontsize=7.6, frameon=False,
          handlelength=1.0, handletextpad=0.35, borderpad=0.1,
          title="%d matrices; red = q/k, blue = v/o/FFN" % len(erows), title_fontsize=7.6)

for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=9)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)

# =============================== readouts ===============================
R = json.load(open(READ)) if READ.exists() else {}
R["fig1"] = {"script": Path(__file__).name, "output": str(OUT.relative_to(ROOT)),
             "reference_constant_k_gauss_init": K_GAUSS_INIT,
             "a_public_pythia_endpoints": fig1c, "b_init_families": fig1b, "c_tclock": fig1a, "panel_order_note": "260911: panels reordered public -> init families -> grid"}
json.dump(R, open(READ, "w"), indent=1)
print("readouts ->", READ)
print(json.dumps({"a_all": fig1a["all"]}, indent=1))
for f in FAM:
    e = fig1a["per_family"][f]
    print(f"{f:10s} k_raw {e['k_raw_range'][0]:.3f}-{e['k_raw_range'][1]:.3f} "
          f"k_norm {e['k_norm_range'][0]:.3f}-{e['k_norm_range'][1]:.3f} (med {e['k_norm_median']:.3f}) "
          f"hw med {e['hw_median']:.3f} max {e['hw_range'][1]:.3f} corr raw {e['corr_k_raw_vs_hw']:+.2f} norm {e['corr_k_norm_vs_hw']:+.2f}")
print("fig1b step0 k_bi", fig1b["step0_family_median_k_bi"], "30k range", fig1b["step30000_per_matrix_range_k_bi"], "gauss all-steps", fig1b["gaussian_k_bi_range_all_steps"])
print("fig1b family medians k_bi", {i: [round(v, 4) for v in fig1b["family_median_k_bi_all_kinds"][i]] for i in INITS})
for s in SIZES_C:
    print(s, {kd: "%.3f->%.3f" % (fig1c["per_size"][s][kd]["k_raw_median"], fig1c["per_size"][s][kd]["k_axis_median"]) for kd in KINDS_C})
