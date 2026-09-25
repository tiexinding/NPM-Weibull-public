"""Paper#5 v2 Figure 1(b) candidate (260911): k_norm trajectories of three initialization families.

Question: is the row-normalized within-row shape kept from initialization (Gaussian) or does it
approach a common band early in training regardless of the initial marginal?  Here the E-A runs
(llama-70M, {gaussian, laplace, uniform} x seeds {1,2,3}, 30k steps) are refitted at the seven
retained checkpoints with the same protocol as Figure 1 / S4:
  weibull_fit_1024 middle-80%, row-normalization preserving global RMS (v additionally row+column).
Raw pooled k at 15 time points comes from each run's analysis.json (same fitting protocol at training time).

Outputs: 08_paper5_draft/figures/P5v2_F1b_init_families.png
         08_paper5_draft/data/p5v2_fig1b_init_families_readouts_v1.json
Grade: descriptive; three families x three seeds, one architecture / one data setting.
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
DRAFT = ROOT / "08_paper5_draft"
FIG = DRAFT / "figures/P5v2_F1b_init_families.png"
READ = DRAFT / "data/p5v2_fig1b_init_families_readouts_v1.json"
K0 = json.load(open(DRAFT / "data/p5v2_S4_v_twoaxis_readouts_v1.json"))["k0"]  # Gaussian protocol reference, not hardcoded here
HB, HR = 1024, (-12.0, 2.0)
R2_GATE = 0.99
FAMS = ["gaussian", "laplace", "uniform"]
SEEDS = [1, 2, 3]
STEPS = [0, 800, 3200, 10000, 20000, 25000, 30000]
COMP = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
GROUPS = {"q/k": ["q_proj", "k_proj"], "v": ["v_proj"], "gate/up": ["gate_proj", "up_proj"]}
C_FAM = {"gaussian": "#404040", "laplace": "#B2182B", "uniform": "#2166AC"}
MERGE_TOL = 0.01


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
    scale = np.sqrt((w ** 2).mean())
    for _ in range(it):
        w2 = colnorm(rownorm(w))
        if np.abs(w2 - w).max() < tol * scale:
            w = w2
            break
        w = w2
    rr = np.sqrt((w ** 2).mean(1)); cc = np.sqrt((w ** 2).mean(0))
    assert rr.std() / rr.mean() < 1e-3 and cc.std() / cc.mean() < 1e-3, "not converged"
    return w


def layer_of(nm):
    return int(re.search(r"layers\.(\d+)", nm).group(1))


# ---------- per-matrix refits at the retained checkpoints
per_matrix = []
n_low_r2 = 0
for fam in FAMS:
    for s in SEEDS:
        for step in STEPS:
            p = ROOT / f"04_ea_runs/ckpt_key/ea_{fam}_s{s}/ckpt/step{step}.pt"
            st = torch.load(p, map_location="cpu", weights_only=True)
            n = 0
            for nm, w in st.items():
                mm = re.search(r"\.(\w+_proj)\.weight$", nm)
                if not mm or mm.group(1) not in COMP:
                    continue
                w = w.double().numpy()
                kr, r2r = kfit(w); kn, r2n = kfit(rownorm(w))
                rec = {"family": fam, "seed": s, "step": step, "comp": mm.group(1), "layer": layer_of(nm),
                       "k_raw": kr, "r2_raw": r2r, "k_norm": kn, "r2_norm": r2n,
                       "H_row": float(np.log(np.sqrt((w ** 2).mean(1))).std())}
                if mm.group(1) == "v_proj":
                    kb, r2b = kfit(bothnorm(w)); rec["k_both"] = kb; rec["r2_both"] = r2b
                n_low_r2 += int(min(rec[k] for k in rec if k.startswith("r2_")) < R2_GATE)
                per_matrix.append(rec); n += 1
            assert n == 6 * len(COMP), (p, n)
assert len(per_matrix) == len(FAMS) * len(SEEDS) * len(STEPS) * 6 * len(COMP), len(per_matrix)

# ---------- raw pooled k trajectories (15 time points) from analysis.json
raw_traj = {}
raw_steps = None
for fam in FAMS:
    for s in SEEDS:
        rows = json.load(open(ROOT / f"04_ea_runs/ea_{fam}_s{s}/analysis.json"))["rows"]
        steps = sorted({r["step"] for r in rows})
        raw_steps = steps if raw_steps is None else raw_steps
        assert steps == raw_steps and set(STEPS) <= set(steps)
        for g, comps in GROUPS.items():
            raw_traj[f"{fam}|s{s}|{g}"] = [float(np.median([r["k"] for r in rows if r["step"] == st and r["kind"] in comps])) for st in steps]
            assert all(sum(1 for r in rows if r["step"] == st and r["kind"] in comps) == 6 * len(comps) for st in steps)

# ---------- family/run medians, merge step
def med(fam, s, step, g, key):
    comps = GROUPS[g]
    v = [r[key] for r in per_matrix if r["family"] == fam and r["seed"] == s and r["step"] == step and r["comp"] in comps]
    assert len(v) == 6 * len(comps)
    return float(np.median(v))

run_med = {f"{fam}|s{s}|{g}|{key}": [med(fam, s, st, g, key) for st in STEPS]
           for fam in FAMS for s in SEEDS for g in GROUPS for key in ("k_norm", "k_raw")}
run_med.update({f"{fam}|s{s}|v|k_both": [med(fam, s, st, "v", "k_both") for st in STEPS] for fam in FAMS for s in SEEDS})
fam_med = {}
for fam in FAMS:
    for g in GROUPS:
        for key in (["k_norm", "k_raw"] + (["k_both"] if g == "v" else [])):
            fam_med[f"{fam}|{g}|{key}"] = [float(np.median([run_med[f"{fam}|s{s}|{g}|{key}"][i] for s in SEEDS])) for i in range(len(STEPS))]
# refit raw family median over the 15 analysis steps
raw_fam_med = {f"{fam}|{g}": [float(np.median([raw_traj[f"{fam}|s{s}|{g}"][i] for s in SEEDS])) for i in range(len(raw_steps))]
               for fam in FAMS for g in GROUPS}

merge = {}
for g in GROUPS:
    for fam in ("laplace", "uniform"):
        gaus = fam_med[f"gaussian|{g}|k_norm"]; oth = fam_med[f"{fam}|{g}|k_norm"]
        hit = [STEPS[i] for i in range(len(STEPS)) if abs(oth[i] - gaus[i]) < MERGE_TOL]
        merge[f"{fam}|{g}|k_norm_first_step_within_{MERGE_TOL}_of_gaussian"] = hit[0] if hit else None
        gaus = raw_fam_med[f"gaussian|{g}"]; oth = raw_fam_med[f"{fam}|{g}"]
        hit = [raw_steps[i] for i in range(len(raw_steps)) if abs(oth[i] - gaus[i]) < MERGE_TOL]
        merge[f"{fam}|{g}|k_raw_first_step_within_{MERGE_TOL}_of_gaussian"] = hit[0] if hit else None

def rng(fam, g, key, step):
    v = [r[key] for r in per_matrix if r["family"] == fam and r["step"] == step and r["comp"] in GROUPS[g]]
    return [float(min(v)), float(max(v))]

R = {"protocol": "weibull_fit_1024 middle-80%; row normalization preserves global RMS; v also row+column (alternating, flat to 0.1%)",
     "k0_gaussian_reference": K0, "r2_gate": R2_GATE, "n_fits_below_r2_gate": n_low_r2,
     "runs": [f"ea_{f}_s{s}" for f in FAMS for s in SEEDS], "ckpt_steps": STEPS, "analysis_steps": raw_steps,
     "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
     "step0_family_median_k_norm": {f"{fam}|{g}": fam_med[f"{fam}|{g}|k_norm"][0] for fam in FAMS for g in GROUPS},
     "step0_family_median_k_raw": {f"{fam}|{g}": fam_med[f"{fam}|{g}|k_raw"][0] for fam in FAMS for g in GROUPS},
     "step30000_per_matrix_range_k_norm": {f"{fam}|{g}": rng(fam, g, "k_norm", STEPS[-1]) for fam in FAMS for g in GROUPS},
     "step30000_per_matrix_range_v_k_both": {fam: rng(fam, "v", "k_both", STEPS[-1]) for fam in FAMS},
     "merge_steps": merge, "run_medians": run_med, "family_medians": fam_med,
     "raw_k_trajectories_analysis_json": {"steps": raw_steps, "run": raw_traj, "family_median": raw_fam_med},
     "per_matrix": per_matrix}
json.dump(R, open(READ, "w"), indent=1)

# ---------- figure: one panel per group; k_norm solid (seed thin, family median thick), raw k dashed (15 pts)
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5})
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.9), sharey=True)
xs = np.array(STEPS, dtype=float); xr = np.array(raw_steps, dtype=float)
for ax, g in zip(axes, GROUPS):
    for fam in FAMS:
        c = C_FAM[fam]
        for s in SEEDS:
            ax.plot(xs, run_med[f"{fam}|s{s}|{g}|k_norm"], color=c, lw=0.8, alpha=0.45)
        ax.plot(xs, fam_med[f"{fam}|{g}|k_norm"], color=c, lw=2.2, marker="o", ms=3.5, mec="white")
        ax.plot(xr, raw_fam_med[f"{fam}|{g}"], color=c, lw=1.4, ls=(0, (3, 2)))
    ax.axhline(K0, color="0.35", lw=1.0, ls=(0, (5, 3)), zorder=0)
    ax.set_xscale("symlog", linthresh=800, linscale=0.6)
    ax.set_xticks([0, 800, 3200, 10000, 30000], ["0", "800", "3.2k", "10k", "30k"])
    ax.set_xlim(-60, 33000)
    ax.set_xlabel("training step (symlog, linear below 800)")
    ax.set_title(f"({'abc'[list(GROUPS).index(g)]}) {g}", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Weibull shape k (median over layers, matrices)")
h = [Line2D([], [], color=C_FAM[f], lw=2.2, label=f"{f} init") for f in FAMS] + [
    Line2D([], [], color="0.3", lw=2.2, label="row-normalized k_norm (thick: family median; thin: seeds)"),
    Line2D([], [], color="0.3", lw=1.4, ls=(0, (3, 2)), label="raw pooled k (family median, 15 checkpoints)"),
    Line2D([], [], color="0.35", lw=1.0, ls=(0, (5, 3)), label=f"Gaussian protocol reference {K0}")]
axes[2].legend(handles=h, fontsize=7.6, frameon=False, loc="lower right")
fig.text(0.01, 0.005, "E-A runs: llama-70M, one data setting, 3 initialization families x 3 seeds, 30k steps.", fontsize=8, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG, dpi=200, bbox_inches="tight", facecolor="white")

print("fits below R2 gate:", n_low_r2, "of", len(per_matrix))
for k, v in R["step0_family_median_k_norm"].items(): print("step0 k_norm", k, round(v, 3), " raw", round(R["step0_family_median_k_raw"][k], 3))
for k, v in merge.items(): print("merge", k, v)
for k, v in R["step30000_per_matrix_range_k_norm"].items(): print("30k range k_norm", k, [round(x, 3) for x in v])
for k, v in R["step30000_per_matrix_range_v_k_both"].items(): print("30k range v k_both", k, [round(x, 3) for x in v])
print("wrote", FIG)
