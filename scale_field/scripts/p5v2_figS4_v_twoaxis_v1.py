"""Paper#5 v2 supplementary figure S4 (260910): v_proj is a two-axis scale field.

Question: after row normalization the k_norm of q/k returns to the initialization value but
v_proj stays at 1.185-1.194 (Figure 1b). Why?  Answer measured here: v_proj carries a
column-side (residual-stream channel) scale field of the same size as its row-side field;
row normalization removes only the row mixture. Row+column normalization returns v to 1.20.

Panels
  (a) v_proj per layer, IVN base @30k and TCLOCK D1_s1 / D3_s1 @5k: k under raw / row-only
      (old protocol) / row+column (new) normalization.
  (b) v_proj along the IVN base run (8 checkpoints): median and min-max over 6 layers of
      k_raw, k_row (old) and k_both (new).
  (c) H_row vs H_col per family, IVN base @30k: v is the only family with H_col ~ H_row.
  Readouts also hold the column-profile correlations (v_col vs q/k/gate/up columns, o/down
  rows, input-LayerNorm gain; cross-layer and 5k->30k persistence).

Protocol: weibull_fit_1024 middle-80% (as kmix_bridge_v1_1); row-norm = each row to matrix
RMS; col-norm = same on the transpose; both = alternate row/column until both row and column RMS are flat to 0.1% (≤60 iterations).
Sources: 05_ivn_runs/ivn_base/ckpt/step*.pt, 05_ivn_runs/ivn_base/full_model_final.pt,
         04_tclock_runs/wt_ckpts/tclock_D{1,3}_s1/step5000.pt
Outputs: 08_paper5_draft/figures/P5v2_S4_v_two_axis.png
         08_paper5_draft/data/p5v2_S4_v_twoaxis_readouts_v1.json
Grade: descriptive, single run (IVN) + two single-seed TCLOCK arms. Backup figure, not in main text.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "08_paper5_draft/figures/P5v2_S4_v_two_axis.png"
READ = ROOT / "08_paper5_draft/data/p5v2_S4_v_twoaxis_readouts_v1.json"
HB, HR = 1024, (-12.0, 2.0)
K0 = 1.204
C_SEL, C_TRA, C_RAW, C_NORM = "#B2182B", "#2166AC", "#404040", "#D95F02"
C_BOTH = "#1B7837"
FAM = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
LAB = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "gate_proj": "gate", "up_proj": "up"}
STEPS = [800, 3200, 5000, 10000, 15000, 20000, 25000, 30000]


def fit_hist(hist):
    hist = hist.astype(np.float64); total = hist.sum()
    edges = np.linspace(HR[0], HR[1], HB + 1)
    F = ((np.cumsum(hist) / total) + ((np.cumsum(hist) - hist) / total)) / 2.0
    bc = (edges[:-1] + edges[1:]) / 2.0 * np.log(10.0)
    m = (F >= 0.1) & (F <= 0.9) & (hist > 0)
    y = np.log(-np.log(1 - np.clip(F[m], 1e-12, 1 - 1e-12)))
    X = np.vstack([bc[m], np.ones(m.sum())]).T
    w = hist[m]; WX = X * w[:, None]
    beta, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)
    res = y - X @ beta; ybar = (w * y).sum() / w.sum()
    return float(beta[0]), float(1 - (w * res ** 2).sum() / (w * (y - ybar) ** 2).sum())


def kfit(x):
    return fit_hist(np.histogram(np.log10(np.maximum(np.abs(x.ravel()), 1e-13)), bins=HB, range=HR)[0])


def rownorm(w):
    r = np.sqrt((w ** 2).mean(1, keepdims=True)); z = w / np.maximum(r, 1e-30)
    return z * np.sqrt((w ** 2).mean()) / np.sqrt((z ** 2).mean())


def colnorm(w):
    return rownorm(w.T).T


def bothnorm(w, it=60, tol=1e-5):
    """Alternate row / column normalization until both row and column RMS are flat (Sinkhorn-like)."""
    scale = np.sqrt((w ** 2).mean())
    for _ in range(it):
        w2 = colnorm(rownorm(w))
        if np.abs(w2 - w).max() < tol * scale:
            w = w2
            break
        w = w2
    rr = np.sqrt((w ** 2).mean(1)); cc = np.sqrt((w ** 2).mean(0))
    assert rr.std() / rr.mean() < 1e-3 and cc.std() / cc.mean() < 1e-3, ("not converged", rr.std() / rr.mean(), cc.std() / cc.mean())
    return w


def Hs(w):
    return (float(np.log(np.sqrt((w ** 2).mean(1))).std()), float(np.log(np.sqrt((w ** 2).mean(0))).std()))


def analyse(w):
    kr, r2r = kfit(w); kw, r2w = kfit(rownorm(w)); kc, r2c = kfit(colnorm(w)); kb, r2b = kfit(bothnorm(w))
    assert min(r2r, r2w, r2c, r2b) > 0.99, "fit gate"
    hr, hc = Hs(w)
    assert abs(np.sqrt((bothnorm(w) ** 2).mean()) - np.sqrt((w ** 2).mean())) < 1e-9, "global RMS preserved"
    return {"k_raw": kr, "k_row": kw, "k_col": kc, "k_both": kb, "H_row": hr, "H_col": hc}


def load(p):
    return torch.load(p, map_location="cpu", weights_only=True)


def layer_of(nm):
    return int(re.search(r"layers\.(\d+)", nm).group(1))


R = {"protocol": "weibull_fit_1024 middle-80%; row/col/both normalization preserve global RMS", "k0": K0}

# ---------- (b) IVN base trajectory, all families
traj = {f: {} for f in FAM}
for s in STEPS:
    st = load(ROOT / f"05_ivn_runs/ivn_base/ckpt/step{s}.pt")
    for nm, w in st.items():
        mm = re.search(r"\.(\w+_proj)\.weight$", nm)
        if not mm or mm.group(1) not in FAM:
            continue
        traj[mm.group(1)][f"{s}|L{layer_of(nm)}"] = analyse(w.double().numpy())
assert all(len(traj[f]) == 6 * len(STEPS) for f in FAM)
R["ivn_base_trajectory"] = traj

# ---------- (a) per-layer v @ IVN 30k and TCLOCK D1/D3 @5k
groups = {"IVN base @30k": {f"L{L}": traj["v_proj"][f"30000|L{L}"] for L in range(6)}}
for arm in ("D1", "D3"):
    st = load(ROOT / f"04_tclock_runs/wt_ckpts/tclock_{arm}_s1/step5000.pt")
    groups[f"TCLOCK {arm} s1 @5k"] = {f"L{layer_of(nm)}": analyse(w.double().numpy())
                                      for nm, w in st.items() if nm.endswith("v_proj.weight")}
    assert len(groups[f"TCLOCK {arm} s1 @5k"]) == 6
R["v_per_layer"] = groups
for gname, g in groups.items():
    vals_b = [d["k_both"] for d in g.values()]; vals_r = [d["k_row"] for d in g.values()]
    assert min(vals_b) > 1.185 and max(vals_b) < 1.225, (gname, vals_b)
    assert np.median(vals_b) > np.median(vals_r), gname

# ---------- column-profile correlations (IVN final)
full = load(ROOT / "05_ivn_runs/ivn_base/full_model_final.pt")
sd = full["model"] if isinstance(full, dict) and "model" in full else full
def lr(w, ax): return np.log(np.sqrt((w.double().numpy() ** 2).mean(ax)))
corr = {}
prev = None; adj = []
for L in range(6):
    g = lambda n: sd[f"model.layers.{L}.{n}.weight"]
    vc = lr(g("self_attn.v_proj"), 0)
    corr[f"L{L}"] = {
        "v_col_vs_q_col": float(spearmanr(vc, lr(g("self_attn.q_proj"), 0))[0]),
        "v_col_vs_k_col": float(spearmanr(vc, lr(g("self_attn.k_proj"), 0))[0]),
        "v_col_vs_gate_col": float(spearmanr(vc, lr(g("mlp.gate_proj"), 0))[0]),
        "v_col_vs_up_col": float(spearmanr(vc, lr(g("mlp.up_proj"), 0))[0]),
        "v_col_vs_o_row": float(spearmanr(vc, lr(g("self_attn.o_proj"), 1))[0]),
        "v_col_vs_down_row": float(spearmanr(vc, lr(g("mlp.down_proj"), 1))[0]),
        "v_col_vs_input_ln_gain": float(spearmanr(vc, np.log(np.abs(sd[f"model.layers.{L}.input_layernorm.weight"].double().numpy()) + 1e-12))[0]),
    }
    if prev is not None:
        adj.append(float(spearmanr(prev, vc)[0]))
    prev = vc
s5 = load(ROOT / "05_ivn_runs/ivn_base/ckpt/step5000.pt"); s30 = load(ROOT / "05_ivn_runs/ivn_base/ckpt/step30000.pt")
persist = [float(spearmanr(lr(s5[f"model.layers.{L}.self_attn.v_proj.weight"], 0),
                           lr(s30[f"model.layers.{L}.self_attn.v_proj.weight"], 0))[0]) for L in range(6)]
R["v_col_profile_correlations_spearman"] = {"per_layer": corr, "adjacent_layer_v_col": adj, "v_col_5k_vs_30k": persist,
    "reading": "v_col anti-correlates with k_col (and q_col), correlates positively with up_col in layers 1-4, weakly with the LN gain; profile shared across layers 1-5 and persistent 5k->30k. Descriptive, single run."}

# ---------- figure
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9), gridspec_kw={"width_ratios": [1.25, 1.05, 0.9]})
# (a)
ax = axes[0]
x0 = 0
xt, xl = [], []
for gname, g in groups.items():
    for L in range(6):
        d = g[f"L{L}"]; x = x0 + L
        ax.plot([x, x], [d["k_row"], d["k_both"]], color="#CCCCCC", lw=1, zorder=1)
        ax.scatter(x, d["k_raw"], marker="o", s=26, color=C_RAW, zorder=3)
        ax.scatter(x, d["k_row"], marker="s", s=30, facecolor="white", edgecolor=C_NORM, lw=1.5, zorder=3)
        ax.scatter(x, d["k_both"], marker="D", s=26, facecolor="white", edgecolor=C_BOTH, lw=1.5, zorder=4)
        xt.append(x); xl.append(str(L))
    ax.text(x0 + 2.5, 1.232, gname, ha="center", fontsize=8.6, color="#333333")
    x0 += 7
    if x0 < 20:
        ax.axvline(x0 - 1, color="#DDDDDD", lw=0.8)
ax.axhline(K0, color="0.35", lw=1.2, ls=(0, (5, 3)), zorder=1)
ax.set_xticks(xt, xl, fontsize=8); ax.set_xlabel("layer")
ax.set_ylabel("Weibull shape k of v_proj")
ax.set_ylim(0.98, 1.24)
ax.set_title("(a) v_proj: row-only vs row+column normalization", loc="left", fontsize=10.5)
h = [Line2D([], [], marker="o", color=C_RAW, ls="", ms=6, label="raw k"),
     Line2D([], [], marker="s", mfc="white", mec=C_NORM, mew=1.5, ls="", ms=6, label="row-normalized (old)"),
     Line2D([], [], marker="D", mfc="white", mec=C_BOTH, mew=1.5, ls="", ms=6, label="row+column-normalized (new)"),
     Line2D([], [], color="0.35", ls=(0, (5, 3)), label="Gaussian-init 1.204")]
ax.legend(handles=h, fontsize=7.8, frameon=False, loc="lower left", ncol=2)
# (b)
ax = axes[1]
xs = np.array(STEPS, dtype=float)
for key, c, ls, lab in (("k_raw", C_RAW, (0, (3, 2)), "raw k"), ("k_row", C_NORM, "-", "row-normalized (old)"), ("k_both", C_BOTH, "-", "row+column (new)")):
    arr = np.array([[traj["v_proj"][f"{s}|L{L}"][key] for L in range(6)] for s in STEPS])
    ax.fill_between(xs, arr.min(1), arr.max(1), color=c, alpha=0.15, lw=0)
    ax.plot(xs, np.median(arr, 1), color=c, lw=2, ls=ls, marker="o", ms=3.5, mec="white", label=lab)
ax.axhline(K0, color="0.35", lw=1.2, ls=(0, (5, 3)))
ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000], ["0", "5k", "10k", "15k", "20k", "25k", "30k"], fontsize=8.6)
ax.set_xlim(-600, 31000); ax.set_xlabel("training step"); ax.set_ylabel("Weibull shape k of v_proj")
ax.set_title("(b) v_proj along the IVN base run (6 layers)", loc="left", fontsize=10.5)
ax.legend(fontsize=7.8, frameon=False, loc="lower left", title="line: median; band: min-max over layers", title_fontsize=7.8)
# (c)
ax = axes[2]
MK = {"q_proj": "o", "k_proj": "s", "v_proj": "^", "gate_proj": "D", "up_proj": "v"}
for f in FAM:
    c = C_SEL if f in ("q_proj", "k_proj") else C_TRA
    hr = [traj[f][f"30000|L{L}"]["H_row"] for L in range(6)]; hc = [traj[f][f"30000|L{L}"]["H_col"] for L in range(6)]
    ax.scatter(hr, hc, marker=MK[f], s=34, facecolor=c if f != "v_proj" else "white", edgecolor=c, lw=1.4, label=LAB[f], zorder=3)
lim = 0.28
ax.plot([0, lim], [0, lim], color="#BBBBBB", lw=0.9, ls="--"); ax.text(lim * 0.97, lim * 0.9, "H_col = H_row", fontsize=7.5, color="#777777", ha="right")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("row-side heterogeneity  H_row = sd_i(log row RMS)"); ax.set_ylabel("column-side  H_col = sd_j(log column RMS)")
ax.set_title("(c) Only v has a column field as large as its row field", loc="left", fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.text(0.01, 0.005, "IVN base: llama-70M, structured data, single seed, step 30,000 unless noted. Descriptive; backup figure.", fontsize=8, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG, dpi=200, bbox_inches="tight", facecolor="white")
json.dump(R, open(READ, "w"), indent=1)
v30 = [traj["v_proj"][f"30000|L{L}"] for L in range(6)]
print("v @30k  k_row %.3f-%.3f  k_both %.3f-%.3f  H_row %.2f-%.2f  H_col %.2f-%.2f" % (
    min(d["k_row"] for d in v30), max(d["k_row"] for d in v30), min(d["k_both"] for d in v30), max(d["k_both"] for d in v30),
    min(d["H_row"] for d in v30), max(d["H_row"] for d in v30), min(d["H_col"] for d in v30), max(d["H_col"] for d in v30)))
print("adjacent-layer v_col rho", [round(a, 2) for a in adj], "persist 5k->30k", [round(p, 2) for p in persist])
print("wrote", FIG)
