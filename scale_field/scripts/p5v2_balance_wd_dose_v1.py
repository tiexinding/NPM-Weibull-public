"""Paper#5 v2 Figure S15: is the path balance mode set by weight decay? (zero GPU)

For matched units on a path (up row i / down column i; v row / o column; q/k rows of the
same RoPE pair) the rescaling (e^eps, e^-eps) is an exact gauge freedom of the network
output.  Decoupled weight decay (-eta*lambda*W) is the only term that breaks it and pushes
the two norms toward equality, so the balance mode b = h1 - h2 should have mean ~0, small
variance that falls with lambda_wd, and should converge in time; with lambda_wd = 0 it can
keep its initial spread and drift.

Doses (all seed 1, identical batch stream, eta 1e-3, bs 24 x 512, 30k steps):
  lambda 0.0   05_opt_runs/opt_a1_{gaussian,laplace}_s1
  lambda 0.03  05_opt_runs/opt_a3w003_{gaussian,laplace}_s1
  lambda 0.1   04_ea_runs/ckpt_key/ea_{gaussian,laplace}_s1      (E-A, same protocol)
  lambda 0.3   05_opt_runs/opt_a3w030_{gaussian,laplace}_s1
checkpoints 0/800/3200/10000/20000/25000/30000, 42 analysis matrices each.

Two versions of the balance are reported:
  b_c   centred (median-subtracted per side, as in S13)     -> var(b_c), ratio var(g)/var(b_c)
  d     uncentred ln RMS_1(i) - ln RMS_2(i)                  -> mean(d), sd(d): the log norm ratio
Cross-check: lambda 0.1 rows (E-A gaussian_s1 / laplace_s1) must reproduce S13 to 1e-9.

Outputs:
  08_paper5_draft/data/p5v2_balance_wd_dose_v1.json
  08_paper5_draft/figures/P5v2_S15_balance_wd_dose.png
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "08_paper5_draft/data/p5v2_balance_wd_dose_v1.json"
OUT_FIG = ROOT / "08_paper5_draft/figures/P5v2_S15_balance_wd_dose.png"
S13 = json.load(open(ROOT / "08_paper5_draft/data/p5v2_path_gain_balance_v1.json"))
IVN = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
HEADS, HEAD_DIM = int(IVN["heads"]), int(IVN["head_dim"])
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

INITS = ["gaussian", "laplace"]
STEPS = [0, 800, 3200, 10000, 20000, 25000, 30000]
DOSE_DIR = {0.0: "05_opt_runs/opt_a1_{i}_s1/ckpt", 0.03: "05_opt_runs/opt_a3w003_{i}_s1/ckpt",
            0.1: "04_ea_runs/ckpt_key/ea_{i}_s1/ckpt", 0.3: "05_opt_runs/opt_a3w030_{i}_s1/ckpt"}
DOSES = sorted(DOSE_DIR)
MAIN = ["QK_pair", "VO", "UD"]
GCOL = {"QK_pair": "#B2182B", "VO": "#2166AC", "UD": "#1b9e77"}
DCOL = {0.0: "#404040", 0.03: "#7570b3", 0.1: "#d95f02", 0.3: "#1b9e77"}

# ---- dose metadata: read lambda from run_meta, never hard-code
def meta_lwd(dose, init):
    d = ROOT / DOSE_DIR[dose].format(i=init)
    mp = d.parent / "run_meta.json"
    if not mp.exists():  # E-A: ckpt_key/ea_*/ckpt holds the tensors, run_meta sits in 04_ea_runs/ea_*/
        mp = ROOT / f"04_ea_runs/ea_{init}_s1/run_meta.json"
    m = json.load(open(mp))
    return float(m["lwd"]), m["batch_hashes_first5"][:5] if isinstance(m["batch_hashes_first5"], list) else eval(m["batch_hashes_first5"])[:5]


CK = []
for dose in DOSES:
    for init in INITS:
        lwd, bh = meta_lwd(dose, init)
        assert abs(lwd - dose) < 1e-12, (dose, init, lwd)
        for st in STEPS:
            CK.append((dose, init, st, ROOT / DOSE_DIR[dose].format(i=init) / f"step{st}.pt", bh))
assert len(CK) == len(DOSES) * len(INITS) * len(STEPS) == 56, len(CK)
missing = [str(p) for *_, p, _ in CK if not p.exists()]
assert not missing, missing
# same batch stream across doses (first-5 hashes)
bh0 = CK[0][4]
for c in CK:
    assert c[4] == bh0, ("batch stream differs", c[:3])


def lnrms(x, axis):
    return np.log(np.sqrt((x ** 2).mean(axis)))


def cen(v):
    return v - np.median(v)


def load_layers(p):
    st = torch.load(p, map_location="cpu", weights_only=True)
    st = st.get("model", st) if isinstance(st, dict) and "model" in st else st
    lay = {}
    for nm, w in st.items():
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", nm)
        if m:
            lay.setdefault(int(m.group(1)), {})[m.group(2)] = w.double().numpy()
    return lay


def pair_profile(h_row):
    half = HEAD_DIM // 2
    m = h_row.reshape(HEADS, HEAD_DIM)
    return np.median((m[:, :half] + m[:, half:]) / 2, 0)


def raw_profiles(m):
    """uncentred ln RMS on each side of the three paths"""
    q_r, k_r = lnrms(m["q_proj"], 1), lnrms(m["k_proj"], 1)
    v_r, o_c = lnrms(m["v_proj"], 1), lnrms(m["o_proj"], 0)
    u_r, d_c = lnrms(m["up_proj"], 1), lnrms(m["down_proj"], 0)
    assert q_r.shape == k_r.shape == v_r.shape == o_c.shape and u_r.shape == d_c.shape
    return {"QK_pair": (pair_profile(q_r), pair_profile(k_r)), "VO": (v_r, o_c), "UD": (u_r, d_c)}


def stats(h1, h2):
    """h1,h2 uncentred. centred g/b as in S13; uncentred d = h1-h2"""
    c1, c2 = cen(h1), cen(h2)
    g, b = c1 + c2, c1 - c2
    d = h1 - h2
    return {"var_g": float(g.var()), "var_b": float(b.var()), "ratio": float(g.var() / b.var()),
            "corr": float(np.corrcoef(c1, c2)[0, 1]),
            "mean_d": float(d.mean()), "sd_d": float(d.std()), "q05_d": float(np.quantile(d, .05)), "q95_d": float(np.quantile(d, .95)),
            "mean_h1": float(h1.mean()), "mean_h2": float(h2.mean())}


S13ROWS = {(r["run"], r["step"], r["layer"]): r for r in S13["rows"] if r["family"] == "ea"}

rows, bvec = [], {}
for dose, init, st, p, _ in CK:
    lay = load_layers(p); Ls = sorted(lay); assert len(Ls) == 6, (p, Ls)
    for L in Ls:
        P = raw_profiles(lay[L])
        e = {"dose": dose, "init": init, "step": st, "layer": L, "s13_crosschecked": False}
        for gname in MAIN:
            h1, h2 = P[gname]; e[gname] = stats(h1, h2)
            bvec[(dose, init, st, L, gname)] = (cen(h1) - cen(h2), h1 - h2)
        if dose == 0.1:
            s = S13ROWS[(f"{init}_s1", st, L)]
            for gname in MAIN:
                for k in ("var_g", "var_b", "corr"):
                    assert abs(e[gname][k] - s[gname][k]) < 1e-9, (init, st, L, gname, k, e[gname][k], s[gname][k])
            e["s13_crosschecked"] = True
        rows.append(e)
    print(dose, init, st, "done", flush=True)
assert len(rows) == 56 * 6, len(rows)
assert sum(1 for r in rows if r["s13_crosschecked"]) == 2 * 7 * 6


def med(v):
    v = np.array(v, float); return {"n": int(v.size), "median": float(np.median(v)), "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)), "min": float(v.min()), "max": float(v.max())}


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


# ---- summaries
by_dose_step = {}
for dose in DOSES:
    for st in STEPS:
        rs = sel(dose=dose, step=st)
        by_dose_step[f"{dose}|{st}"] = {g: {k: med([r[g][k] for r in rs]) for k in ("var_g", "var_b", "ratio", "corr", "mean_d", "sd_d")} for g in MAIN}
by_dose_30k = {str(dose): by_dose_step[f"{dose}|30000"] for dose in DOSES}

# balance autocorrelation across steps: corr of b vector at step s with b at 30k (same layer), median over layers x inits
autocorr = {}
for dose in DOSES:
    for g in MAIN:
        for st in STEPS:
            cs = []
            for init in INITS:
                for L in range(6):
                    b_s, _ = bvec[(dose, init, st, L, g)]; b_f, _ = bvec[(dose, init, 30000, L, g)]
                    cs.append(float(np.corrcoef(b_s, b_f)[0, 1]))
            autocorr[f"{dose}|{g}|{st}"] = med(cs)
# also: b at step 0 vs b at 30k (does the initial balance persist?)
persist = {f"{dose}|{g}": autocorr[f"{dose}|{g}|0"]["median"] for dose in DOSES for g in MAIN}

# monotonicity of var(b) in lambda at 30k (Spearman over 4 doses of the median)
from scipy.stats import spearmanr
mono = {}
for g in MAIN:
    v = [by_dose_30k[str(d)][g]["var_b"]["median"] for d in DOSES]
    vg = [by_dose_30k[str(d)][g]["var_g"]["median"] for d in DOSES]
    mono[g] = {"var_b_30k_by_dose": dict(zip(map(str, DOSES), v)), "spearman_var_b_vs_lambda": float(spearmanr(DOSES, v)[0]),
               "var_g_30k_by_dose": dict(zip(map(str, DOSES), vg)), "spearman_var_g_vs_lambda": float(spearmanr(DOSES, vg)[0]),
               "ratio_30k_by_dose": {str(d): by_dose_30k[str(d)][g]["ratio"]["median"] for d in DOSES}}

example = {}
for dose in (0.0, 0.3):
    b_c, d = bvec[(dose, "gaussian", 30000, 3, "UD")]
    example[str(dose)] = {"b_centred": b_c.tolist(), "d_uncentred": d.tolist(), "layer": 3, "init": "gaussian", "step": 30000, "path": "UD"}

out = {"schema": "p5v2_balance_wd_dose_v1", "script": Path(__file__).name, "script_sha256": SHA,
       "protocol": "raw W; h = ln RMS along the identity axis; centred g/b use median-subtracted h per side (S13 protocol); "
                   "d = uncentred ln RMS_1 - ln RMS_2 per matched unit (log norm ratio); QK by RoPE pair (rotate_half, median over heads); "
                   "doses read from run_meta lwd; all runs seed 1 with identical batch stream",
       "doses": DOSES, "inits": INITS, "steps": STEPS, "n_ckpt": len(CK), "n_rows": len(rows),
       "dose_sources": {str(d): DOSE_DIR[d] for d in DOSES},
       "summary_by_dose_step": by_dose_step, "summary_30k_by_dose": by_dose_30k,
       "balance_autocorr_with_30k": autocorr, "initial_balance_persistence_corr_b0_b30k": persist,
       "monotonicity_30k": mono, "example_UD_L3_gaussian_30k": example, "rows": rows}
json.dump(out, open(OUT_JSON, "w"), indent=1)
print("json ->", OUT_JSON)

# ---- figure
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8})
fig = plt.figure(figsize=(13.4, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0], wspace=0.45)
xd = np.arange(len(DOSES))
# (a) var(b) and var(g) vs lambda at 30k
ax = fig.add_subplot(gs[0, 0])
for g in MAIN:
    vb = [by_dose_30k[str(d)][g]["var_b"] for d in DOSES]; vg = [by_dose_30k[str(d)][g]["var_g"] for d in DOSES]
    ax.errorbar(xd, [v["median"] for v in vb], yerr=[[v["median"] - v["q25"] for v in vb], [v["q75"] - v["median"] for v in vb]],
                color=GCOL[g], lw=1.8, marker="o", ms=4, capsize=2, label=None)
    ax.errorbar(xd + 0.08, [v["median"] for v in vg], yerr=[[v["median"] - v["q25"] for v in vg], [v["q75"] - v["median"] for v in vg]],
                color=GCOL[g], lw=1.4, ls=(0, (3, 2)), marker="s", ms=3.5, mfc="white", capsize=2)
ax.set_yscale("log"); ax.set_xticks(xd); ax.set_xticklabels([str(d) for d in DOSES])
ax.set_xlabel(r"weight decay $\lambda_{\mathrm{wd}}$"); ax.set_ylabel("variance over units, step 30,000")
ax.set_title(r"(a) var($b$) and var($g$) against $\lambda_{\mathrm{wd}}$", loc="left")
h = [Line2D([], [], color=GCOL[g], lw=1.8, marker="o", ms=4, label={"QK_pair": "q/k by RoPE pair", "VO": "v rows / o columns", "UD": "up rows / down columns"}[g]) for g in MAIN] + \
    [Line2D([], [], color="0.4", lw=1.8, marker="o", ms=4, label="balance $b$ (solid)"), Line2D([], [], color="0.4", lw=1.4, ls=(0, (3, 2)), marker="s", ms=3.5, mfc="white", label="gain $g$ (dashed)")]
ax.legend(handles=h, loc="lower left", frameon=False, handlelength=1.8)
ax.text(0.98, 0.97, "median over 2 inits x 6 layers;\nbar: IQR", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.3")


def _sx(v):
    v = np.asarray(v, float); return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))


# (b) mean(d) and var(b) over steps per dose (UD path; VO and QK in json)
ax = fig.add_subplot(gs[0, 1])
ax2 = ax.twinx()
for dose in DOSES:
    vb = [by_dose_step[f"{dose}|{st}"]["UD"]["var_b"]["median"] for st in STEPS]
    md = [by_dose_step[f"{dose}|{st}"]["UD"]["mean_d"]["median"] for st in STEPS]
    ax.plot(_sx(STEPS), vb, color=DCOL[dose], lw=1.9, marker="o", ms=3.5, label=f"$\\lambda$ = {dose}")
    ax2.plot(_sx(STEPS), md, color=DCOL[dose], lw=1.2, ls=(0, (3, 2)), marker="s", ms=3, mfc="white")
ax.set_yscale("log"); xt = [0, 800, 3200, 10000, 30000]; ax.set_xticks(_sx(xt)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
ax.set_xlabel("training step  (linear to 800, log above)"); ax.set_ylabel("var($b$), up rows / down columns (solid)")
ax2.set_ylabel(r"mean of $\ln\|W_{\mathrm{up},i}\| - \ln\|W_{\mathrm{down},\cdot i}\|$ (dashed)")
ax2.axhline(0, color="0.6", lw=0.8)
ax.set_title("(b) Balance over training, per dose (up/down path)", loc="left")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, ncol=1, handlelength=1.6, fontsize=7.5)
# (c) histograms of d at 30k, lambda 0 vs 0.3
ax = fig.add_subplot(gs[0, 2])
allv = np.concatenate([np.array(example["0.0"]["d_uncentred"]), np.array(example["0.3"]["d_uncentred"])])
bins = np.linspace(allv.min(), allv.max(), 45)
for dose in (0.0, 0.3):
    d = np.array(example[str(dose)]["d_uncentred"])
    ax.hist(d, bins=bins, color=DCOL[dose], alpha=0.55, label=f"$\\lambda$ = {dose}: mean {d.mean():+.3f}, sd {d.std():.3f}")
ax.axvline(0, color="0.4", lw=0.9)
ax.set_xlabel(r"$\ln\|W_{\mathrm{up},i}\| - \ln\|W_{\mathrm{down},\cdot i}\|$  (1,376 hidden units)")
ax.set_ylabel("count"); ax.set_title("(c) Log norm ratio per hidden unit, layer 3, 30k", loc="left")
ax.legend(loc="upper right", frameon=False)
for a in fig.axes:
    a.spines[["top", "right"]].set_visible(False) if a is not ax2 and a not in (fig.axes[1],) else None
fig.axes[1].spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
print("fig ->", OUT_FIG)
for g in MAIN:
    print(g, "var_b 30k by dose", {k: round(v, 5) for k, v in mono[g]["var_b_30k_by_dose"].items()}, "spearman", round(mono[g]["spearman_var_b_vs_lambda"], 2),
          "| ratio", {k: round(v, 1) for k, v in mono[g]["ratio_30k_by_dose"].items()})
    print("   persistence corr(b0,b30k) by dose", {str(d): round(persist[f"{d}|{g}"], 2) for d in DOSES})
    print("   mean_d 30k by dose", {str(d): round(by_dose_30k[str(d)][g]["mean_d"]["median"], 4) for d in DOSES},
          "sd_d", {str(d): round(by_dose_30k[str(d)][g]["sd_d"]["median"], 4) for d in DOSES})
