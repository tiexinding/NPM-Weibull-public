"""Paper#5 Figure 3 v3 (260916 late): common core + scale fields -> pooled shape through the mixture bridge.

v3 (老丁 §4.2 主线): panel (c) is now the two-axis prediction scatter (absorbed from the former
two-axis figure, data p5v2_twoaxis_bridge_v1.json); the per-kind alpha_P panel moves to the appendix
figure p5v3_figB_bridge_coefficients_v1.py. Symbols: one-axis coefficient alpha_P, two-axis alpha_r/alpha_c.

Reads ONLY stored JSON (no checkpoint access):
  06_hrow/kmix_bridge_v1_1.json                         partA, partA_audit_rows (1,440 grid fits)
  08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json   k0_gaussian_reference (protocol reference 1.204)
  08_paper5_draft/data/p5v2_twoaxis_bridge_v1.json      per_matrix (2,016 two-axis fits), fits.pooled_all7
Writes:
  08_paper5_draft/p5_compile/figure_v2/figures/P5v3_F2_scale_field_bridge_v4.png

Changes vs v1 (老丁 260916 审查清单):
  * 2x2 layout instead of four panels in one row (aspect 4.4:1 -> ~1.35:1).
  * (a) legend lists every kind with its own colour+marker; filled/open glyphs in a second legend;
        in-panel "n = 1440 ... reference k = 1.20" text removed; reference line is the protocol value
        1.204 (read from JSON), labelled at its right end.
  * (b) in-panel alpha/R^2/n text removed (caption); lines get their own legend entries.
  * (c) "marker: pooled fit / bar: range" note removed (caption); per-row R^2 labels kept as data labels.
  * (d) in-panel "max fold = arm D3 ..." note removed (caption).
  * Component palette: q/k red (#B2182B), v blue (#2166AC), gate/up green (#1B7837); path members
    distinguished by marker (circle / triangle).
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
DRAFT = HERE.parents[3]                       # 08_paper5_draft
SM = DRAFT.parent                             # selfavg_mechanism
OUT = HERE.parents[1] / "figures" / "P5v3_F2_scale_field_bridge_v4.png"

A = json.load(open(SM / "06_hrow/kmix_bridge_v1_1.json"))
K_REF = float(json.load(open(DRAFT / "data/p5v2_ea_kbi_trajectory_v1.json"))["k0_gaussian_reference"])
D2 = json.load(open(DRAFT / "data/p5v2_twoaxis_bridge_v1.json"))
assert D2["schema"] == "p5v2-twoaxis-bridge-v1", D2["schema"]
assert abs(K_REF - 1.204) < 1e-9, K_REF        # protocol reference used in Figure 2
ALPHA_WEIBULL = 6 / np.pi ** 2                 # exact-Weibull coefficient (analytic constant)

SEL = ["q_proj", "k_proj"]
TRA = ["v_proj", "gate_proj", "up_proj"]
FAM = SEL + TRA
LABEL = {"q_proj": "$q$", "k_proj": "$k$", "v_proj": "$v$", "gate_proj": "gate", "up_proj": "up"}
C_QK, C_VO, C_FFN = "#B2182B", "#2166AC", "#1B7837"
COL = {"q_proj": C_QK, "k_proj": C_QK, "v_proj": C_VO, "gate_proj": C_FFN, "up_proj": C_FFN}
MRK = {"q_proj": "o", "k_proj": "^", "v_proj": "o", "gate_proj": "o", "up_proj": "^"}
# seven-kind encodings for the two-axis panel (same palette as Figure 2)
FAM7 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAB7 = {**LABEL, "o_proj": "$o$", "down_proj": "down"}
COL7 = {**COL, "o_proj": C_VO, "down_proj": C_FFN}
MRK7 = {**MRK, "o_proj": "^", "down_proj": "s"}

# ----------------------------- data + asserts -----------------------------
rows = A["partA_audit_rows"]
assert len(rows) == 1440, len(rows)
assert set(r["kind"] for r in rows) == set(FAM)
assert min(min(r["r2fit_raw"], r["r2fit_norm"]) for r in rows) > 0.99, "fit gate"
assert len(set(r["arm"] for r in rows)) == 4 and len(set(r["seed"] for r in rows)) == 3
assert len(set(r["step"] for r in rows)) == 4
assert A["partA"]["J_A_v11"] == "supported", A["partA"]["J_A_v11"]
assert all(A["partA"][f]["runs_same_sign"] == 12 for f in FAM)
for r in rows:
    r["d"] = r["k_raw"] ** -2 - r["k_norm"] ** -2
    r["h2"] = r["hw"] ** 2
qk = A["partA"]["qk_pooled"]
assert qk["n"] == 576, qk["n"]
a_tra = float(np.mean([A["partA"][f]["alpha"] for f in TRA]))
for f in FAM:
    assert len(A["partA"][f]["pred_mae_LORO"]["per_group"]) == 12 and len(A["partA"][f]["pred_mae_LOAO"]["per_group"]) == 4, f
loao_argmax = {f: max(A["partA"][f]["pred_mae_LOAO"]["per_group"].items(), key=lambda kv: kv[1])[0] for f in FAM}
loro_argmax = {f: max(A["partA"][f]["pred_mae_LORO"]["per_group"].items(), key=lambda kv: kv[1])[0] for f in FAM}
assert all(a in loro_argmax[f] for f, a in loao_argmax.items()), "LORO max fold should lie in the LOAO max arm"

rows2 = D2["per_matrix"]
assert len(rows2) == D2["n_matrices"] == 2016, len(rows2)
assert D2["n_fits_below_gate"] == 0 and min(min(r["r2_raw"], r["r2_bi"]) for r in rows2) >= D2["r2_gate"], "two-axis fit gate"
assert list(D2["families"]) == FAM7, D2["families"]
M2 = D2["fits"]["pooled_all7"]["M2"]; M1 = D2["fits"]["pooled_all7"]["M1"]
assert M2["coef_names"] == ["a_r", "a_c"], M2["coef_names"]
alpha_r, alpha_c = M2["coef"]

hw_all = np.array([r["hw"] for r in rows]); kr_all = np.array([r["k_raw"] for r in rows]); kn_all = np.array([r["k_norm"] for r in rows])

# ----------------------------- figure -----------------------------
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 10})
fig = plt.figure(figsize=(11.6, 6.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.92],
                      wspace=0.22, hspace=0.34, left=0.065, right=0.985, top=0.95, bottom=0.085)

# ---- (a) raw and row-normalized shape against H^W
ax = fig.add_subplot(gs[0, 0])
for f in FAM:
    s = [r for r in rows if r["kind"] == f]
    x = np.array([r["hw"] for r in s])
    ax.scatter(x, [r["k_raw"] for r in s], s=12, marker=MRK[f], facecolor=COL[f], edgecolor="none", alpha=0.45, zorder=3)
    ax.scatter(x, [r["k_norm"] for r in s], s=13, marker=MRK[f], facecolor="none", edgecolor=COL[f], linewidth=0.6, alpha=0.55, zorder=2)
ax.axhline(K_REF, color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1)
xmax_a = float(hw_all.max()) * 1.04
ax.set_xlim(0, xmax_a)
ax.text(xmax_a, K_REF, f" {K_REF:.3f}", fontsize=7.6, color="0.45", va="center", ha="left", clip_on=False)
ax.set_ylim(float(kr_all.min()) - 0.03, float(kn_all.max()) + 0.03)
ax.set_xlabel(r"Field width  $H^W$")
ax.set_ylabel("Shape  $k$")
ax.set_title("(a)  Shape against field width, 1,440 fits", loc="left")
h_kind = [Line2D([], [], ls="none", marker=MRK[f], mfc=COL[f], mec=COL[f], ms=5.4, label=LABEL[f]) for f in FAM]
h_state = [Line2D([], [], ls="none", marker="s", mfc="0.3", mec="0.3", ms=5, label=r"raw $k_{\mathrm{raw}}$"),
           Line2D([], [], ls="none", marker="s", mfc="white", mec="0.3", mew=1.2, ms=5, label=r"row-normalized $k_{\mathrm{row}}$")]
lg = ax.legend(handles=h_kind, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=5, fontsize=8.2, frameon=False,
               columnspacing=0.9, handletextpad=0.3, borderpad=0.2)
ax.add_artist(lg)
ax.legend(handles=h_state, loc="lower left", bbox_to_anchor=(0.0, 0.085), ncol=2, fontsize=8.2, frameon=False,
          columnspacing=1.2, handletextpad=0.4, borderpad=0.2)

# ---- (b) bridge scatter
ax = fig.add_subplot(gs[0, 1])
for f in FAM:
    x = np.array([r["h2"] for r in rows if r["kind"] == f])
    y = np.array([r["d"] for r in rows if r["kind"] == f])
    ax.scatter(x, y, s=12, marker=MRK[f], facecolor="none", linewidth=0.7, edgecolor=COL[f], alpha=0.55, zorder=2)
h2max = max(r["h2"] for r in rows) * 1.03
xs = np.linspace(0, h2max, 100)
ax.plot(xs, qk["alpha"] * xs, color=C_QK, lw=1.8, zorder=4)
ax.plot(xs, a_tra * xs, color="0.25", lw=1.8, zorder=4)
ax.plot(xs, ALPHA_WEIBULL * xs, color="0.5", lw=1.1, ls=(0, (5, 3)), zorder=3)
ax.set_xlim(0, h2max)
ax.set_ylim(-0.012, max(r["d"] for r in rows) * 1.05)
ax.set_xlabel(r"$(H^W)^2$")
ax.set_ylabel(r"$\Delta_k^{\mathrm{mix}} = k_{\mathrm{raw}}^{-2} - k_{\mathrm{row}}^{-2}$")
ax.set_title(r"(b)  Mixture term against $(H^W)^2$", loc="left")
h_pts = [Line2D([], [], ls="none", marker=MRK[f], mfc="none", mec=COL[f], mew=0.9, ms=5.6, label=LABEL[f]) for f in FAM]
h_lines = [Line2D([], [], color=C_QK, lw=1.8, label="$q$/$k$ pooled fit"),
           Line2D([], [], color="0.25", lw=1.8, label="$v$/gate/up mean"),
           Line2D([], [], color="0.5", lw=1.1, ls=(0, (5, 3)), label=r"$6/\pi^2$")]
lg = ax.legend(handles=h_pts, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=8.2, frameon=False,
               handletextpad=0.3, labelspacing=0.3, borderpad=0.2)
ax.add_artist(lg)
ax.legend(handles=h_lines, loc="upper left", bbox_to_anchor=(0.17, 1.0), ncol=1, fontsize=8.2, frameon=False,
          handlelength=1.8, handletextpad=0.5, labelspacing=0.3, borderpad=0.2)

# ---- (c) two-axis prediction (absorbed from the former two-axis figure)
ax = fig.add_subplot(gs[1, 0])
y2 = np.array([r["k_raw"] ** -2 - r["k_bi"] ** -2 for r in rows2])
p2 = np.array([alpha_r * r["H_row"] ** 2 + alpha_c * r["H_col"] ** 2 for r in rows2])
kind2 = np.array([r["kind"] for r in rows2])
for f in FAM7:
    m = kind2 == f
    ax.scatter(p2[m], y2[m], s=13, marker=MRK7[f], color=COL7[f], alpha=0.65, lw=0.4, edgecolor="white", zorder=3)
lim = float(max(y2.max(), p2.max())) * 1.05
ax.plot([0, lim], [0, lim], color="0.45", lw=1.0, ls=(0, (5, 3)), zorder=2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel(r"Two-axis prediction  $\alpha_r (H^W_{\mathrm{row}})^2 + \alpha_c (H^W_{\mathrm{col}})^2$")
ax.set_ylabel(r"$k_{\mathrm{raw}}^{-2} - k_{\mathrm{bi}}^{-2}$")
ax.set_title("(c)  Two-axis mixture term, 2,016 fits", loc="left")
h_c = [Line2D([], [], ls="none", marker=MRK7[f], color=COL7[f], ms=5.5, label=LAB7[f]) for f in FAM7]
h_c.append(Line2D([], [], color="0.45", lw=1.0, ls=(0, (5, 3)), label="identity"))
ax.legend(handles=h_c, loc="lower right", ncol=2, fontsize=8.2, frameon=False, handlelength=1.2,
          handletextpad=0.4, columnspacing=0.9, borderpad=0.2)

# ---- (d) held-out relative MAE, recomputed per matrix from the audit rows (protocol reproduced and asserted)
rows_all = A["partA_audit_rows"]
def _fit_alpha(rs):                       # through-origin fit of y = k_raw^-2 - k_norm^-2 on x = hw^2
    x = np.array([r["hw"] ** 2 for r in rs]); y = np.array([r["k_raw"] ** -2 - r["k_norm"] ** -2 for r in rs])
    return float((x * y).sum() / (x * x).sum())
def _pred(rs, alpha):
    return np.array([(r["k_norm"] ** -2 + alpha * r["hw"] ** 2) ** -0.5 for r in rs]), np.array([r["k_raw"] for r in rs])
REL = {}   # kind -> dict(ins, loro_med, loro_max, loao_med, loao_max) in percent
ABS = {}
for f in FAM:
    rs = [r for r in rows_all if r["kind"] == f]
    assert len(rs) == 288, (f, len(rs))
    alpha_all = _fit_alpha(rs)
    assert abs(alpha_all - A["partA"][f]["alpha"]) < 1e-9, (f, alpha_all, A["partA"][f]["alpha"])
    ph, y = _pred(rs, alpha_all)
    ins_abs = float(np.mean(np.abs(ph - y))); assert abs(ins_abs - A["partA"][f]["gof_mae_insample"]) < 1e-9, f
    ins_rel = float(np.mean(np.abs(ph - y) / y)) * 100
    def _folds(keyfn, stored):
        groups = sorted(set(keyfn(r) for r in rs)); rel, ab = {}, {}
        for g in groups:
            tr = [r for r in rs if keyfn(r) != g]; te = [r for r in rs if keyfn(r) == g]
            al = _fit_alpha(tr); ph, y = _pred(te, al)
            ab[g] = float(np.mean(np.abs(ph - y))); rel[g] = float(np.mean(np.abs(ph - y) / y)) * 100
        # reproduce the stored absolute per-group MAE exactly
        for g, v in ab.items():
            key = str(g) if isinstance(g, str) else str(tuple(g))
            assert abs(v - stored["per_group"][key]) < 1e-9, (f, g, v, stored["per_group"][key])
        return rel, ab
    loro_rel, loro_abs = _folds(lambda r: (r["arm"], r["seed"]), A["partA"][f]["pred_mae_LORO"])
    loao_rel, loao_abs = _folds(lambda r: r["arm"], A["partA"][f]["pred_mae_LOAO"])
    assert len(loro_rel) == 12 and len(loao_rel) == 4
    REL[f] = dict(ins=ins_rel, loro_med=float(np.median(list(loro_rel.values()))), loro_max=max(loro_rel.values()),
                  loao_med=float(np.median(list(loao_rel.values()))), loao_max=max(loao_rel.values()),
                  loro_worst=max(loro_rel, key=loro_rel.get), loao_worst=max(loao_rel, key=loao_rel.get))
    ABS[f] = dict(ins=ins_abs, loro_med=float(np.median(list(loro_abs.values()))), loro_max=max(loro_abs.values()),
                  loao_med=float(np.median(list(loao_abs.values()))), loao_max=max(loao_abs.values()))
ax = fig.add_subplot(gs[1, 1])
xx = np.arange(len(FAM)); w = 0.26
ax.bar(xx - w, [REL[f]["ins"] for f in FAM], w * 0.9, color="0.75", zorder=3)
ax.bar(xx, [REL[f]["loro_med"] for f in FAM], w * 0.9, color=[COL[f] for f in FAM], alpha=0.5, zorder=3)
ax.bar(xx + w, [REL[f]["loao_med"] for f in FAM], w * 0.9, color=[COL[f] for f in FAM], zorder=3)
ax.scatter(xx, [REL[f]["loro_max"] for f in FAM], marker="_", s=120, color="0.2", zorder=5, lw=1.4)
ax.scatter(xx + w, [REL[f]["loao_max"] for f in FAM], marker="_", s=120, color="0.2", zorder=5, lw=1.4)
ax.set_xticks(xx); ax.set_xticklabels([LABEL[f] for f in FAM], fontsize=9.6)
for t, f in zip(ax.get_xticklabels(), FAM):
    t.set_color(COL[f])
ax.set_ylim(0, max(REL[f]["loao_max"] for f in FAM) * 1.12)
ax.set_ylabel(r"Relative MAE of predicted $k_{\mathrm{raw}}$ (%)")
ax.set_title("(d)  Held-out prediction error", loc="left")
h_d = [Patch(facecolor="0.75", label="in-sample"),
       Patch(facecolor="0.4", alpha=0.5, label="leave-one-run-out"),
       Patch(facecolor="0.4", label="leave-one-arm-out"),
       Line2D([], [], ls="none", marker="_", color="0.2", ms=10, mew=1.4, label="worst fold")]
ax.legend(handles=h_d, loc="upper left", fontsize=8.0, frameon=False, handlelength=1.4, handletextpad=0.5,
          labelspacing=0.3, borderpad=0.2)
ax.grid(axis="y", alpha=0.2, lw=0.5)

for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=8.8)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", OUT)

# ----------------------------- read-outs for the caption -----------------------------
print("hw range %.3f-%.3f; k_raw range %.3f-%.3f; corr(hw,k_raw) %+.2f; corr(hw,k_row) %+.2f"
      % (hw_all.min(), hw_all.max(), kr_all.min(), kr_all.max(),
         np.corrcoef(hw_all, kr_all)[0, 1], np.corrcoef(hw_all, kn_all)[0, 1]))
for f in ["q_proj", "k_proj", "gate_proj", "up_proj"]:
    s = [r["k_norm"] for r in rows if r["kind"] == f]
    print(f"  k_row {LABEL[f]:6s} {min(s):.3f}-{max(s):.3f}")
print("qk pooled alpha %.3f R0^2 %.3f n %d; transmission mean alpha %.3f" % (qk["alpha"], qk["R0_sq"], qk["n"], a_tra))
print("per-kind alpha:", {LABEL[f]: round(A["partA"][f]["alpha"], 3) for f in FAM})
print("LOAO worst arm:", loao_argmax)
print("two-axis pooled: alpha_r %.3f alpha_c %.3f R0^2(M2) %.3f R0^2(M1 row-only) %.3f n %d" % (alpha_r, alpha_c, M2["R0_sq"], M1["R0_sq"], len(rows2)))

print("REL (%):", {f: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in REL[f].items()} for f in FAM})
print("ABS:", {f: {k: round(v, 5) for k, v in ABS[f].items()} for f in FAM})
