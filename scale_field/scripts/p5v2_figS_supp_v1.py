"""Paper#5 v2 supplementary figures S1-S3 (260910; zero GPU; all inputs archived).

  S1  Cross-run reproduction of the q/k frequency profile on 9 E-A runs
      (3 init families x 3 seeds); v pseudo-frequency profile = negative control.
      Source: 06_hrow/t123_structure_v1.json  (freq_profiles, t3)
  S2  Axis-specific normalization rescue on the 4 IVN arms: q/k are returned to
      the initialization value by ROW normalization only, W_o by COLUMN only.
      Source: 05_ivn_runs/ivn_prereg_readouts_v1.json (row_rescue)
  S3  KMIX three-term decomposition on 144 real TCLOCK q/k matrices:
      D_actual = (6/pi^2) H^2 + D_high + D_pair; pairing term share by data arm.
      Source: 06_hrow/kmix_bridge_v1_2.json (partD_decomposition)

Outputs:
  08_paper5_draft/figures/P5v2_S1_profile_cross_run.png
  08_paper5_draft/figures/P5v2_S2_axis_rescue.png
  08_paper5_draft/figures/P5v2_S3_kmix_decomposition.png
  08_paper5_draft/data/p5v2_supp_readouts_v1.json

Style follows scripts/p5v2_fig{1,2,3}_v1.py (red = Selection, blue = Transmission,
grey = raw, orange = row-normalized; dashed reference = Gaussian-init k = 1.204).
Evidence grades: S1 observational (same architecture, same corpus); S2 single-seed
paired intervention screen; S3 descriptive decomposition (pre-registered Part D).
"""
import json
import statistics as st
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "08_paper5_draft/figures"
READ = ROOT / "08_paper5_draft/data/p5v2_supp_readouts_v1.json"
C_SEL, C_TRA = "#B2182B", "#2166AC"
C_RAW, C_NORM = "#404040", "#D95F02"
K0 = 1.204  # Gaussian-init reference under the middle-80% protocol
readouts = {}

# ----------------------------------------------------------------- S1
T = json.load(open(ROOT / "06_hrow/t123_structure_v1.json"))
FP, T3 = T["freq_profiles"], T["t3"]
RUNS = sorted(FP.keys())
assert len(RUNS) == 9 and all(len(FP[r][k]) == 32 for r in RUNS for k in ("q", "k", "v"))
FAM_C = {"gaussian": "#B2182B", "laplace": "#E08214", "uniform": "#5AAE61"}
SEED_LS = {"s1": "-", "s2": "--", "s3": ":"}


def cross_run_median(kind):
    vals = [np.corrcoef(FP[a][kind], FP[b][kind])[0, 1] for a, b in combinations(RUNS, 2)]
    return float(np.median(vals)), (float(min(vals)), float(max(vals)))


s1 = {}
for kind in ("q", "k", "v"):
    med, rng = cross_run_median(kind)
    ref = T3[kind]["freq_profile_cross_run_corr_median"]
    assert abs(med - ref) < 0.01, (kind, med, ref)
    s1[kind] = {"cross_run_pearson_median": med, "range": rng}
assert s1["q"]["cross_run_pearson_median"] > 0.9 and s1["k"]["cross_run_pearson_median"] > 0.9
assert s1["v"]["cross_run_pearson_median"] < 0.3
readouts["S1"] = {"runs": RUNS, **s1,
                  "source": "06_hrow/t123_structure_v1.json freq_profiles (pair-averaged, layer+head controlled, 0->30k)"}

fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9), sharey=True)
titles = {"q": "(a) q rows: 9 runs, one shape", "k": "(b) k rows: 9 runs, one shape",
          "v": "(c) v rows (pseudo-frequency index): negative control"}
for ax, kind in zip(axes, ("q", "k", "v")):
    for r in RUNS:
        fam, seed = r.split("_")[1], r.split("_")[2]
        ax.plot(range(32), FP[r][kind], color=FAM_C[fam], ls=SEED_LS[seed], lw=1.4, alpha=0.9)
    ax.axhline(0, color="#BDBDBD", lw=0.8)
    ax.set_title(titles[kind], loc="left", fontsize=11)
    ax.set_xlabel("rotary frequency pair index  (0 = highest frequency)")
    m, (lo, hi) = s1[kind]["cross_run_pearson_median"], s1[kind]["range"]
    ax.text(0.03, 0.05, f"cross-run Pearson median {m:.2f}  (range {lo:.2f}–{hi:.2f})",
            transform=ax.transAxes, fontsize=9, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("accumulated row-scale change 0→30k\n(log r, median-centered, layer+head controlled)")
handles = [Line2D([], [], color=c, lw=2, label=f"{f} init") for f, c in FAM_C.items()] + \
          [Line2D([], [], color="#404040", ls=ls, lw=1.4, label=f"seed {s[1]}") for s, ls in SEED_LS.items()]
axes[0].legend(handles=handles, fontsize=8, ncol=2, frameon=False, loc="upper left")
fig.text(0.99, 0.01, "E-A runs, llama-70M, 30k steps, λ_wd = 0.1; same architecture and corpus in all nine",
         ha="right", fontsize=8, color="#777777")
fig.tight_layout()
fig.savefig(FIG / "P5v2_S1_profile_cross_run.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ----------------------------------------------------------------- S2
RR = json.load(open(ROOT / "05_ivn_runs/ivn_prereg_readouts_v1.json"))["row_rescue"]
ARMS = ["base", "flatv", "nomom", "ropeperm"]
ARM_LAB = {"base": "base (AdamW)", "flatv": "flatv (row-flattened v_t)",
           "nomom": "nomom (β₁ = 0)", "ropeperm": "ropeperm (RoPE map permuted)"}
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
KL = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate",
      "up_proj": "up", "down_proj": "down"}
assert set(RR) == set(ARMS) and all(set(RR[a]) == set(KINDS) for a in ARMS)
s2 = {}
for a in ARMS:
    for kd in ("q_proj", "k_proj"):
        assert RR[a][kd]["k_after_row"] > 1.200 > RR[a][kd]["k_after_col"], (a, kd)
    assert RR[a]["o_proj"]["k_after_col"] > 1.200 > RR[a]["o_proj"]["k_after_row"], a
    s2[a] = {kd: RR[a][kd] for kd in KINDS}
readouts["S2"] = {"arms": s2, "reference_k0": K0,
                  "source": "05_ivn_runs/ivn_prereg_readouts_v1.json row_rescue (terminal window, 6-layer median)"}

fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7), sharey=True)
x = np.arange(len(KINDS))
for ax, a in zip(axes, ARMS):
    ax.axhline(K0, color="#7F7F7F", ls="--", lw=1)
    for i, kd in enumerate(KINDS):
        col = C_SEL if kd in ("q_proj", "k_proj") else C_TRA
        d = RR[a][kd]
        ax.plot([i - 0.22, i - 0.22], [d["k_before"], d["k_after_row"]], color="#CCCCCC", lw=1, zorder=1)
        ax.plot([i + 0.22, i + 0.22], [d["k_before"], d["k_after_col"]], color="#CCCCCC", lw=1, zorder=1)
        ax.scatter(i, d["k_before"], marker="o", s=42, color=C_RAW, zorder=3)
        ax.scatter(i - 0.22, d["k_after_row"], marker="s", s=42, facecolor="white", edgecolor=C_NORM, lw=1.6, zorder=3)
        ax.scatter(i + 0.22, d["k_after_col"], marker="^", s=46, facecolor="white", edgecolor=col, lw=1.6, zorder=3)
    ax.set_xticks(x, [KL[k] for k in KINDS])
    ax.set_title(ARM_LAB[a], loc="left", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    for lab in ax.get_xticklabels():
        lab.set_color(C_SEL if lab.get_text() in ("q", "k") else "#222222")
axes[0].set_ylabel("Weibull shape k (terminal window)")
axes[0].set_ylim(1.17, 1.215)
handles = [Line2D([], [], marker="o", color=C_RAW, ls="", ms=7, label="raw k"),
           Line2D([], [], marker="s", mfc="white", mec=C_NORM, mew=1.6, ls="", ms=7, label="after row normalization"),
           Line2D([], [], marker="^", mfc="white", mec="#555555", mew=1.6, ls="", ms=7, label="after column normalization"),
           Line2D([], [], color="#7F7F7F", ls="--", label="Gaussian-init reference 1.204")]
axes[0].legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
fig.text(0.01, 0.01, "q/k identity axis = rows: rescued by row normalization only.  o identity axis = columns: rescued by column normalization only.  "
         "Global RMS preserved in both operations. Single-seed paired screen.", fontsize=8, color="#555555")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "P5v2_S2_axis_rescue.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ----------------------------------------------------------------- S3
KM = json.load(open(ROOT / "06_hrow/kmix_bridge_v1_2.json"))["partD_decomposition"]
SUM, PM = KM["summary"], KM["per_matrix"]
DARMS = ["D1", "D2", "D3", "D4"]
DLAB = {"D1": "D1\nstructured\nρ=1", "D2": "D2\nshuffled\nρ=1", "D3": "D3\nshuffled\nρ=64", "D4": "D4\nstructured\nρ=64"}
assert len(PM) == 144 and all(sum(1 for r in PM if r["arm"] == a) == 36 for a in DARMS)
base_term = 6 / np.pi ** 2
s3 = {}
for a in DARMS:
    rows = [r for r in PM if r["arm"] == a]
    for r in rows:  # identity closes per matrix
        assert abs(r["d_actual"] - (base_term * r["hw"] ** 2 + r["d_high"] + r["d_pair"])) < 1e-9
    pos = sum(1 for r in rows if r["d_pair"] > 0)
    assert abs(pos / 36 - SUM[a]["pair_pos_frac"]) < 1e-9
    share = SUM[a]["share_med"]
    s3[a] = {"share_med": share, "pair_pos": f"{pos}/36", "d_pair_med": SUM[a]["d_pair_med"],
             "d_actual_med": SUM[a]["d_actual_med"]}
tot_pos = sum(1 for r in PM if r["d_pair"] > 0)
assert tot_pos == 132, tot_pos
readouts["S3"] = {"arms": s3, "pair_positive_total": f"{tot_pos}/144",
                  "source": "06_hrow/kmix_bridge_v1_2.json partD_decomposition (q/k, 6 layers x 3 seeds x 4 arms @5000, 10 stable permutations each)"}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
w, xs = 0.25, np.arange(4)
C_BASE, C_HIGH, C_PAIR = "#9E9E9E", "#BDBDBD", C_SEL
for j, (key, col, lab) in enumerate([("base", C_BASE, "base mixture (6/π²)·(H^W)²"),
                                     ("high", C_HIGH, "higher-moment term (skew of h)"),
                                     ("pair", C_PAIR, "scale–shape pairing term")]):
    vals = [SUM[a]["share_med"][key] for a in DARMS]
    ax.bar(xs + (j - 1) * w, vals, w - 0.03, color=col, label=lab)
    for xi, v in zip(xs + (j - 1) * w, vals):
        ax.text(xi, v + (0.015 if v >= 0 else -0.05), f"{v:.2f}", ha="center", fontsize=8, color="#333333")
ax.axhline(0, color="#7F7F7F", lw=0.8)
ax.set_xticks(xs, [DLAB[a] for a in DARMS], fontsize=9)
ax.set_ylabel("median share of Δ_k^mix per matrix")
ax.set_title("(a) Three-term decomposition by data arm (q/k, step 5,000)", loc="left", fontsize=11)
ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=1)
ax.set_ylim(-0.2, 1.3)
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
rng = np.random.default_rng(0)
for i, a in enumerate(DARMS):
    rows = [r for r in PM if r["arm"] == a]
    y = np.array([r["d_pair"] for r in rows])
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=16, color=C_SEL, alpha=0.7, edgecolor="none")
    ax.text(i, 0.0185, s3[a]["pair_pos"] + " > 0", ha="center", fontsize=8.5, color="#333333")
ax.axhline(0, color="#7F7F7F", lw=0.8)
ax.set_xticks(range(4), [DLAB[a] for a in DARMS], fontsize=9)
ax.set_ylabel("pairing term Δ_pair (per matrix)")
ax.set_title("(b) The pairing term is positive in 132/144 matrices", loc="left", fontsize=11)
ax.set_ylim(-0.004, 0.0195)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.01, "Δ_pair = Δ_k^mix(real pairing) − median over 10 re-assignments of the same row scales to the same normalized rows; "
         "shares are per-matrix medians and do not add to 1 exactly.", fontsize=8, color="#555555")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIG / "P5v2_S3_kmix_decomposition.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

json.dump(readouts, open(READ, "w"), indent=1)
print("S1", {k: round(v["cross_run_pearson_median"], 3) for k, v in s1.items()})
print("S2 ok; S3 pair>0", tot_pos, "/144; shares", {a: round(s3[a]["share_med"]["pair"], 2) for a in DARMS})
