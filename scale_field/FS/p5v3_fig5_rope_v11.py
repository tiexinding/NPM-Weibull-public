"""v10 (260923 老丁): Fig 5 labels unified — q/k coloured red/purple in every panel (a: base solid, permuted dashed); same quantity named the same in (a) and (c) ("Row-scale profile change"); axis labels capitalised as in the other figures; frequency direction written into the x labels instead of separate high/low annotations. Output F5_rope_v8 (Fig 11/12 unchanged, same file names).
v9 (260923 老丁): Fig 5 (b)(c) colour separation — k purple (#7B3294) instead of light red, distinct markers per kind in (b) (q circle, k triangle, v diamond); in (c) colour = kind and model size = line style + marker (70M dotted/triangle, 160M dashed/square, 410M solid/circle) instead of alpha shades. Outputs F5_rope_v7 / SB_rope_readout_v5 / SB_nope_v6.
v8 (260923): house style — fonts as Figs 2-4 (title 10.5, label 10, ticks 8.6); Fig 5: Delta r_id definition in the y label, pairs per head in the x labels, null and control named in full; Fig 11: kind labels italic, o marked n/a in the width panel; Fig 12: legends added to (b) and (c), bar values printed in (c), null percentile named. Arm palette (blue/orange) kept per 260920 decision. Outputs F5_rope_v6 / SB_rope_readout_v4 / SB_nope_v5.
v7 (260920): readout (a) kind labels as x ticks, seed labels and legend at top, y range extended for headroom. v6 (260920): readout fig (b) split into shape/width axes with seed note; NoPE (b) title renamed; outputs SB_rope_readout_v2 / SB_nope_v4; Fig 5 unchanged.
Paper#5 RoPE figures v5 (260917, per 老丁 split by evidence grade). Read-only JSON.
Fig 5 (main, 1x3): (a) permuted run indexed by coordinate vs by frequency; (b) identity advantage per layer at three
paired seeds with v control; (c) frequency profile at three Pythia sizes (static transport).
Fig B-readout (appendix, 1x2): (a) pooled C_freq/C_coord per seed; (b) change of pooled k, k_row, H^W under permutation.
Fig B-nope (appendix, 1x3): (a) H^W of q/k over training, RoPE vs NoPE; (b) pair-indexed profile at 30k;
(c) q/k pair correlation and pair gain-to-balance ratio, with v/o and up/down as path controls.
"""
import json, statistics as st
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parents[4]; REPO = ROOT.parents[1]
FIG = ROOT / "08_paper5_draft/p5_compile/figure_v2/figures"
C_Q, C_K, C_V = "#B2182B", "#7B3294", "#808080"; C_BASE, C_PERM, C_NOPE = "#404040", "#D95F02", "#D95F02"
# appendix palette (Fig 11/12): paper blue for the base run, soft orange for the contrast arm, light blue for the secondary read-out
A_BASE, A_ALT, A_LIGHT, A_V = "#2166AC", "#E08214", "#92C5DE", "#2166AC"
KS = ["q_proj", "k_proj", "v_proj"]; LAB = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}
PA = json.load(open(ROOT / "08_paper5_draft/data/p5v2_fig2a_profiles_v1.json")); assert PA["schema"] == "p5v2-fig2a-profiles-v1"; NF = PA["n_freq_pairs"]
J = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json")); assert J["pi_selfcheck_maxdev"] < 1e-6
MS = json.load(open(ROOT / "10_p5batch_cloud/results/j3_multiseed_v1.json")); SEEDS = sorted(int(s) for s in MS["seeds"]); assert len(SEEDS) == 3
NO = json.load(open(ROOT / "10_p5batch_cloud/results/nope_analysis_v1.json")); LAST = str(NO["common_steps"][-1]); assert LAST == "30000"; NPAIR = NO["head_dim"] // 2
HW = json.load(open(ROOT / "08_paper5_draft/data/p5v2_fig2c_hw_sd_readouts_v1.json")); assert HW["schema"] == "p5v2-fig2c-hw-sd-v1"
TERM = J["steps_terminal"]; kterm = {}
for arm in ("base", "ropeperm"):
    rows = json.load(open(ROOT / f"05_ivn_runs/ivn_{arm}/analysis_v2.json"))["rows"]; assert min(r["R2"] for r in rows if r["step"] in TERM) > 0.99
    kterm[arm] = {kd: st.median([r["k"] for r in rows if r["kind"] == kd and r["step"] in TERM]) for kd in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj")}
SIZES = ["70m", "160m", "410m"]; prof = {}
for s in SIZES:
    d = json.load(open(REPO / f"50_data/05_paper4/runs_size/p4grid{s}_f000_r01_s0/p5_extract_v1.json")); assert d["schema"] == "p5-extract-pythia-v1" and d["n_r2fit_below_0.99"] == 0
    steps = d["steps"]; iv = [f"{a}-{b}" for a, b in zip(steps[:-1], steps[1:])]
    prof[s] = {kd: np.sum([np.asarray(d["freq_profile"][kd][i], float) for i in iv], 0) for kd in ("q", "k")}
NFP = len(prof["70m"]["q"])
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5, "legend.fontsize": 10.5})
def clean(fig):
    for a in fig.axes: a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=11.5)

# ================= Fig 5 main =================
fig = plt.figure(figsize=(14.0, 5.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.25, 1.0], wspace=0.28, left=0.05, right=0.99, top=0.88, bottom=0.17)
sg = gs[0, 0].subgridspec(1, 2, wspace=0.08); axc = [fig.add_subplot(sg[0, i]) for i in range(2)]; x32 = np.arange(NF)
for i, (key, lab) in enumerate((("ropeperm_by_coordinate", "by matrix coordinate"), ("ropeperm_by_frequency", "by assigned frequency"))):
    a = axc[i]; a.axhline(0, color="0.5", lw=.7)
    a.plot(x32, PA["pooled"]["q_proj"]["base"], color=C_Q, lw=1.6); a.plot(x32, PA["pooled"]["k_proj"]["base"], color=C_K, lw=1.6)
    a.plot(x32, PA["pooled"]["q_proj"][key], color=C_Q, lw=1.3, ls=(0, (3, 1.6)), alpha=0.9); a.plot(x32, PA["pooled"]["k_proj"][key], color=C_K, lw=1.3, ls=(0, (3, 1.6)), alpha=0.9)
    a.text(0.03, 0.97, lab, transform=a.transAxes, fontsize=10.6, color="0.3", va="top", ha="left"); pass
yl = [v for kd in ("q_proj", "k_proj") for key in PA["pooled"][kd] for v in PA["pooled"][kd][key]]
for a in axc: a.set_ylim(min(yl) - .02, max(yl) + .15); a.set_yticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10])
axc[1].set_yticklabels([]); axc[0].set_ylabel("Row-scale profile change")
axc[0].set_xlabel(f"Rotary pair ({NF} per head), high $\\to$ low freq."); axc[0].xaxis.set_label_coords(1.04, -0.085)
axc[0].set_title("(a)  Profile follows frequency, not coordinate", loc="left")
axc[0].legend(handles=[Line2D([], [], color="0.3", lw=1.6, label="base run"), Line2D([], [], color="0.3", lw=1.3, ls=(0, (3, 1.6)), label="permuted run")], frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.90), handlelength=1.4, borderaxespad=0.2)
axc[1].legend(handles=[Line2D([], [], color=C_Q, lw=2.0, label="$q$"), Line2D([], [], color=C_K, lw=2.0, label="$k$")], frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.90), handlelength=1.4, borderaxespad=0.2)
# (b) x = layer within each seed block
ax = fig.add_subplot(gs[0, 1]); NL = 6; all_dc = []; BW = 7.5; off = {"q_proj": -0.22, "k_proj": 0.0, "v_proj": 0.22}
for si, s in enumerate(SEEDS):
    j3 = MS["seeds"][str(s)]["reduced"]["J3"]
    for kd in KS:
        xs = np.array([si * BW + L + off[kd] for L in range(NL)]); pl, nlp = j3[kd]["per_layer"], j3[kd]["null_layer_p95"]
        ys = [(pl[L] if isinstance(pl, list) else pl[str(L)])["dC"] for L in range(NL)]; nl = [nlp[L] if isinstance(nlp, list) else nlp[str(L)] for L in range(NL)]; all_dc += ys
        ax.plot(xs, ys, {"q_proj": "o", "k_proj": "^", "v_proj": "D"}[kd], color={"q_proj": C_Q, "k_proj": C_K, "v_proj": C_V}[kd], ms={"q_proj": 5.4, "k_proj": 6.0, "v_proj": 4.6}[kd], mec="white", mew=.5, zorder=4); ax.plot(xs, nl, "_", color="0.55", ms=6, mew=1.1, zorder=3)
    if si: ax.axvline(si * BW - 1.0, color="0.85", lw=.8)
    ax.text(si * BW + 2.5, -1.12, f"seed {s}", ha="center", va="center", fontsize=11, color="0.3")
ax.set_xticks([si * BW + L for si in range(3) for L in range(NL)]); ax.set_xticklabels([str(L) for si in range(3) for L in range(NL)]); ax.set_xlabel("Layer, within each paired seed")
ax.axhline(0, color="0.3", lw=.8); ax.set_ylim(-1.28, 1.72); ax.set_yticks([-0.5, 0, 0.5, 1.0]); ax.set_ylabel(r"Frequency advantage  $\Delta r_{\rm id} = r_{\rm freq} - r_{\rm coord}$")
ax.set_title("(b)  Replication across three paired seeds", loc="left")
ax.legend(handles=[Line2D([], [], ls="none", marker=m, color=c, ms=5.5, label=l) for c, m, l in ((C_Q, "o", "$q$"), (C_K, "^", "$k$"), (C_V, "D", "$v$, negative control"))] + [Line2D([], [], ls="none", marker="_", color="0.55", ms=9, mew=1.3, label="layer-wise permutation null, 95%")],
          frameon=False, loc="upper left", ncol=2, columnspacing=1.0, handletextpad=.3)
# (c) Pythia
ax = fig.add_subplot(gs[0, 2]); SZ_LS = {"70m": (0, (1.2, 1.6)), "160m": (0, (4, 2)), "410m": "-"}; SZ_MK = {"70m": "^", "160m": "s", "410m": "o"}; SZ_LW = {"70m": 1.2, "160m": 1.2, "410m": 1.6}
for s in SIZES:
    for kd in ("q", "k"):
        hc = prof[s][kd] - np.median(prof[s][kd]); ax.plot(range(NFP), hc / np.max(np.abs(hc)), color=C_Q if kd == "q" else C_K, ls=SZ_LS[s], marker=SZ_MK[s], ms=3.8, lw=SZ_LW[s], mfc="white", mew=1.0)
ax.axhline(0, color="0.5", lw=.7); ax.set_xlabel("Pair index, high $\\to$ low freq."); ax.set_ylabel(r"Row-scale profile change / max$|\cdot|$"); ax.set_title("(c)  Transport across Pythia sizes", loc="left"); ax.set_ylim(-1.15, 1.75); ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax.legend(handles=[Line2D([], [], color="0.3", ls=SZ_LS[s], marker=SZ_MK[s], mfc="white", ms=3.8, lw=SZ_LW[s], label=s.upper()) for s in SIZES] + [Line2D([], [], color=C_Q, lw=2.0, label="$q$"), Line2D([], [], color=C_K, lw=2.0, label="$k$")], frameon=False, loc="upper left", ncol=2, columnspacing=1.0, handlelength=2.2)

clean(fig); fig.savefig(FIG / "P5v3_F5_rope_v9.png", dpi=220, bbox_inches="tight", facecolor="white"); print("wrote P5v3_F5_rope_v9.png")

# ================= Fig B readout =================
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"wspace": 0.32, "width_ratios": [1.3, 1.0]})
ax = axs[0]; vals = []; xt = []; xl = []
for si, s in enumerate(SEEDS):
    j3 = MS["seeds"][str(s)]["reduced"]["J3"]
    for ki, kd in enumerate(KS):
        x = si * 3.6 + ki; vals += [j3[kd]["C_freq"], j3[kd]["C_coord"]]
        ax.bar(x - .19, j3[kd]["C_freq"], .38, color={"q_proj": C_Q, "k_proj": C_K, "v_proj": A_V}[kd]); ax.bar(x + .19, j3[kd]["C_coord"], .38, color="0.80")
        xt.append(x); xl.append(f"${LAB[kd]}$")
    ax.text(si * 3.6 + 1, 1.12, f"seed {s}", ha="center", fontsize=11.5, color="0.3")
for si in range(1, len(SEEDS)): ax.axvline(si * 3.6 - 0.8, ymin=0, ymax=0.78, color="0.6", lw=0.8, ls=(0, (4, 3)), zorder=0)
ax.set_ylim(min(-.3, min(vals) - .08), 1.85); ax.set_yticks([-0.5, 0, 0.5, 1.0]); ax.set_xticks(xt); ax.set_xticklabels(xl); ax.axhline(0, color="0.3", lw=.8); ax.set_ylabel("profile similarity, permuted vs base"); ax.set_title("(a)  Frequency arrangement is stable across seeds", loc="left")
ax.legend(handles=[Patch(facecolor="0.45", label=r"$r_{\rm freq}$, by frequency (kind colour)"), Patch(facecolor="0.80", label=r"$r_{\rm coord}$, by coordinate")], frameon=False, loc="upper left", ncol=1, handlelength=1.0, handletextpad=0.4, labelspacing=0.3, borderpad=0.1, fontsize=11, bbox_to_anchor=(0.0, 1.0))
K6 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]; L6 = ["$q$", "$k$", "$v$", "$o$", "gate", "up"]; idx = np.arange(6); w = .36
dk = [100 * (kterm["ropeperm"][k] - kterm["base"][k]) / kterm["base"][k] for k in K6]
dkr = [HW["per_family"][k]["pooled18"]["k_norm"]["pct_change"] if k in HW["per_family"] else np.nan for k in K6]
dh = [HW["per_family"][k]["pooled18"]["hw_sd"]["pct_change"] if k in HW["per_family"] else np.nan for k in K6]
axs[1].set_axis_off()
sub = axs[1].get_subplotspec().subgridspec(2, 1, hspace=0.12, height_ratios=[1, 1])
axk = fig.add_subplot(sub[0]); axh = fig.add_subplot(sub[1], sharex=axk)
axk.bar(idx - w / 2, dk, w, color=A_BASE, label="pooled shape $k$"); axk.bar(idx + w / 2, dkr, w, color=A_LIGHT, label="$k_{\\rm row}$")
axk.axhline(0, color="0.3", lw=.8); axk.set_ylim(-0.6, 0.6); axk.set_yticks([-0.5, 0, 0.5]); axk.set_ylabel("shape change (%)")
axk.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.2); axk.tick_params(labelbottom=False)
axk.set_title("(b)  Pooled shape nearly unchanged, width not (seed 1)", loc="left")
axh.bar(idx, dh, .6, color=A_ALT, label="$H^W_{\\rm row}$"); axh.axhline(0, color="0.3", lw=.8)
for i_, v_ in enumerate(dh):
    if np.isnan(v_): axh.text(i_, 0.5, "n/a", ha="center", va="bottom", fontsize=10, color="0.45")
axh.set_xticks(idx); axh.set_xticklabels(L6); axh.set_ylabel("width change (%)"); axh.legend(frameon=False, loc="upper left", handlelength=1.2)
clean(fig); fig.savefig(FIG / "P5v3_SB_rope_readout_v6.png", dpi=220, bbox_inches="tight", facecolor="white"); print("wrote P5v3_SB_rope_readout_v6.png")

# ================= Fig B NoPE =================
fig, axs = plt.subplots(1, 4, figsize=(16.0, 4.6), gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1.25, 0.7, 1.0]})
H = NO["H_identity_axis_by_step"]; common = [int(s) for s in NO["common_steps"]]
def _sx(v): v = np.asarray(v, float); return np.where(v <= 800, v / 800., 1. + np.log10(np.maximum(v, 800) / 800.))
ax = axs[0]
for arm, c in (("base", A_BASE), ("nope", A_ALT)):
    for k, ls in (("q_proj", "-"), ("k_proj", (0, (3, 2)))): ax.plot(_sx(common), [H[arm][k][str(s)] for s in common], color=c, ls=ls, lw=1.5, marker="o", ms=3)
ax.set_xticks(_sx([0, 800, 3200, 10000, 30000])); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"]); ax.set_xlabel("Training step (log above 800)"); ax.set_ylabel("$H^W_{\\rm row}$ of $q$, $k$"); ax.set_title("(a)  $q/k$ row field without RoPE", loc="left")
ax.legend(handles=[Line2D([], [], color=A_BASE, lw=1.5, label="RoPE (base)"), Line2D([], [], color=A_ALT, lw=1.5, label="NoPE"), Line2D([], [], color="0.55", lw=1.5, label="$q$"), Line2D([], [], color="0.55", lw=1.5, ls=(0, (3, 2)), label="$k$")], frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
ax = axs[1]; qk = NO["qk_structure"]
for arm, c in (("base", A_BASE), ("nope", A_ALT)):
    for nm, ls in (("q", "-"), ("k", (0, (4, 2)))): ax.plot(range(NPAIR), np.median([e[f"{nm}_pair_static"] for e in qk[arm][LAST]], 0), color=c, ls=ls, lw=1.4, marker="o", ms=2.4)
ax.axhline(0, color="0.5", lw=.7); ax.set_xlabel("Rotary pair index (label only for NoPE)"); ax.set_ylabel("row-scale profile at 30k"); ax.set_title("(b)  Pair-indexed profile, 30k", loc="left")
yb = ax.get_ylim(); ax.set_ylim(yb[0], yb[1] + 0.35 * (yb[1] - yb[0]))
ax.legend(handles=[Line2D([], [], color=A_BASE, lw=1.4, label="RoPE (base)"), Line2D([], [], color=A_ALT, lw=1.4, label="NoPE"), Line2D([], [], color="0.55", lw=1.4, label="$q$"), Line2D([], [], color="0.55", lw=1.4, ls=(0, (4, 2)), label="$k$")], frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
crit = NO["criteria"]; qkc = crit["q_k_corr_30k"]; gb = crit["gain_balance_ratio_30k"]
ax = axs[2]; w = .36
ax.bar([-w / 2], [qkc["base"]["pair"]], w, color=A_BASE); ax.bar([w / 2], [qkc["nope"]["pair"]], w, color=A_ALT)
for xv, a_ in ((-w / 2, "base"), (w / 2, "nope")): ax.text(xv, qkc[a_]["pair"] + 0.02, f"{qkc[a_]['pair']:.2f}", ha="center", va="bottom", fontsize=10.3, color="0.3")
ax.set_xticks([0]); ax.set_xticklabels(["$q/k$ by RoPE pair"]); ax.set_xlim(-0.7, 0.7); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]);
ax.legend(handles=[Patch(facecolor=A_BASE, label="RoPE (base)"), Patch(facecolor=A_ALT, label="NoPE")], frameon=False, loc="upper left", ncol=1); ax.set_ylabel("$q$--$k$ profile correlation at 30k"); ax.set_title("(c)  Pair coupling", loc="left")
ax = axs[3]; P3 = ["QK", "VO", "UD"]; x = np.arange(3)
ax.bar(x - w / 2, [gb["base"][p]["ratio"] for p in P3], w, color=A_BASE); ax.bar(x + w / 2, [gb["nope"][p]["ratio"] for p in P3], w, color=A_ALT)
ax.scatter(x - w / 2, [gb["base"][p]["null_q975"] for p in P3], marker="_", color="0.45", s=110, zorder=5); ax.scatter(x + w / 2, [gb["nope"][p]["null_q975"] for p in P3], marker="_", color="0.45", s=110, zorder=5)
ax.set_yscale("log"); ax.set_ylim(0.8, ax.get_ylim()[1] * 6); ax.set_xticks(x); ax.set_xticklabels(["$q/k$ (pair)", "$v/o$", "up/down"]); ax.set_ylabel("var($g$) / var($b$) at 30k"); ax.set_title("(d)  Gain-to-balance ratio", loc="left")
ax.legend(handles=[Patch(facecolor=A_BASE, label="RoPE (base)"), Patch(facecolor=A_ALT, label="NoPE"), Line2D([], [], ls="none", marker="_", color="0.45", ms=10, mew=1.4, label="re-pairing null, 97.5%")], frameon=False, loc="upper left", ncol=2, columnspacing=0.8)
clean(fig); fig.savefig(FIG / "P5v3_SB_nope_v7.png", dpi=220, bbox_inches="tight", facecolor="white"); print("wrote P5v3_SB_nope_v7.png")

# ================= caption read-outs =================
print("C_coord/C_freq seed1:", {LAB[k]: (round(J["J3"][k]["C_coord"], 3), round(J["J3"][k]["C_freq"], 3)) for k in KS})
for s in SEEDS:
    j = MS["seeds"][str(s)]["reduced"]["J3"]; print("seed", s, {LAB[k]: (round(j[k]["dC"], 2), j[k]["n_layers_dC_over_layer_null_p95"]) for k in KS})
print("Pythia argmax (centred)", {s: {k: int(np.argmax(prof[s][k] - np.median(prof[s][k]))) for k in ("q", "k")} for s in SIZES}, "pairwise r q", [round(np.corrcoef(prof[a]["q"], prof[b]["q"])[0, 1], 3) for a, b in (("70m", "160m"), ("70m", "410m"), ("160m", "410m"))], "k", [round(np.corrcoef(prof[a]["k"], prof[b]["k"])[0, 1], 3) for a, b in (("70m", "160m"), ("70m", "410m"), ("160m", "410m"))])
print("dk% max", round(max(abs(v) for v in dk), 2), "dkrow% max", round(np.nanmax(np.abs(dkr)), 2), "dH%", dict(zip(L6, [None if np.isnan(v) else round(v, 1) for v in dh])))
print("NoPE H30k", crit["qk_H_row_30k"], "pair corr", {a: round(qkc[a]["pair"], 2) for a in qkc}, "row corr", {a: round(qkc[a]["row"], 2) for a in qkc}, "gb QK", {a: (round(gb[a]["QK"]["ratio"], 1), round(gb[a]["QK"]["null_q975"], 2)) for a in gb}, "VO/UD nope", {p: round(gb["nope"][p]["ratio"], 1) for p in ("VO", "UD")}, "base", {p: round(gb["base"][p]["ratio"], 1) for p in ("VO", "UD")})
print("pseudo-pair corr", round(crit["pseudo_pair_vs_true_pair_corr_30k"]["q_static"]["median_r"], 2), round(crit["pseudo_pair_vs_true_pair_corr_30k"]["k_static"]["median_r"], 2), "null", round(crit["pseudo_pair_vs_true_pair_corr_30k"]["q_static"]["null_abs_p95_median"], 2), "head var share", {a: {k: round(v, 2) for k, v in crit["head_var_share_30k"][a].items()} for a in crit["head_var_share_30k"]}, "loss", crit["loss_30k"])
