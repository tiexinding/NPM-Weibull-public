"""Paper#5: two-axis-normalized core shape k_bi(t) over training for three initialization
families (E-A runs), all seven projection families (zero GPU).

Per matrix (9 runs x 7 checkpoints x 7 families x 6 layers = 2,646 fits):
  k_raw, k_row, k_col, k_bi, H_row, H_col  -- protocol verbatim from p5v2_twoaxis_bridge_v1.py
  (weibull_fit_1024 middle-80%; row / column / alternating row-column RMS normalization at
  fixed global RMS; alternating scheme converged to 0.1%).
Cross-check: k_raw and k_row of q/k/v/gate/up must reproduce
  08_paper5_draft/data/p5v2_fig1b_init_families_readouts_v1.json per matrix (bitwise).

Sources (read-only): 04_ea_runs/ckpt_key/ea_{gaussian,laplace,uniform}_s{1,2,3}/ckpt/step{...}.pt
Outputs: 08_paper5_draft/data/p5v2_ea_kbi_trajectory_v1.json
         08_paper5_draft/figures/P5v2_S9_ea_kbi_trajectory.png
"""
import hashlib
import json
import re
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "08_paper5_draft"
CK = ROOT / "04_ea_runs/ckpt_key"
OUT_JSON = DRAFT / "data/p5v2_ea_kbi_trajectory_v1.json"
OUT_FIG = DRAFT / "figures/P5v2_S9_ea_kbi_trajectory.png"
FIG1B = DRAFT / "data/p5v2_fig1b_init_families_readouts_v1.json"

HB, HR = 1024, (-12.0, 2.0)
R2_GATE = 0.99
INITS = ["gaussian", "laplace", "uniform"]
SEEDS = [1, 2, 3]
STEPS = [0, 800, 3200, 10000, 20000, 25000, 30000]
FAM = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAB = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
B1 = json.load(open(FIG1B))
K0 = float(B1["k0_gaussian_reference"])          # Gaussian protocol reference (from readouts, not hardcoded)
MERGE_TOL = 0.01                                    # same definition as fig1b merge_steps


# ------------------------------------------------------------------ protocol (verbatim p5v2_twoaxis_bridge_v1)
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


def bothnorm_tracked(w, it=200, tol=1e-6):
    scale = np.sqrt((w ** 2).mean())
    a = np.ones(w.shape[0]); b = np.ones(w.shape[1]); z = w.copy()
    for _ in range(it):
        r = np.sqrt((z ** 2).mean(1)); z = z / r[:, None]; a = a * r
        c = np.sqrt((z ** 2).mean(0)); z = z / c[None, :]; b = b * c
        rr = np.sqrt((z ** 2).mean(1)); cc = np.sqrt((z ** 2).mean(0))
        if rr.std() / rr.mean() < tol and cc.std() / cc.mean() < tol:
            break
    g = scale / np.sqrt((z ** 2).mean()); z = z * g; a = a / np.sqrt(g); b = b / np.sqrt(g)
    rr = np.sqrt((z ** 2).mean(1)); cc = np.sqrt((z ** 2).mean(0))
    assert rr.std() / rr.mean() < 1e-3 and cc.std() / cc.mean() < 1e-3, ("not converged", rr.std() / rr.mean(), cc.std() / cc.mean())
    assert np.abs(a[:, None] * z * b[None, :] - w).max() < 1e-9 * scale, "W = diag(a) Z diag(b) violated"
    return z, a, b


def analyse(w):
    kr, r2r = kfit(w); kw, r2w = kfit(rownorm(w)); kc, r2c = kfit(colnorm(w))
    z, a, b = bothnorm_tracked(w); kb, r2b = kfit(z)
    hr = float(np.log(np.sqrt((w ** 2).mean(1))).std()); hc = float(np.log(np.sqrt((w ** 2).mean(0))).std())
    return {"k_raw": kr, "k_row": kw, "k_col": kc, "k_bi": kb,
            "r2_raw": r2r, "r2_row": r2w, "r2_col": r2c, "r2_bi": r2b, "H_row": hr, "H_col": hc}


def do_ckpt(args):
    init, sd, step = args
    st = torch.load(CK / f"ea_{init}_s{sd}/ckpt/step{step}.pt", map_location="cpu", weights_only=True)
    rows = []
    for nm, w in st.items():
        mm = re.search(r"\.(\w+_proj)\.weight$", nm)
        if not mm or mm.group(1) not in FAM:
            continue
        L = int(re.search(r"layers\.(\d+)", nm).group(1))
        d = analyse(w.double().numpy())
        d.update({"family": init, "seed": sd, "step": step, "comp": mm.group(1), "layer": L, "mat": nm, "shape": list(w.shape)})
        rows.append(d)
    assert len(rows) == 42, (args, len(rows))
    return rows


if __name__ == "__main__":
    jobs = list(product(INITS, SEEDS, STEPS))
    assert len(jobs) == 63
    with Pool(8) as pool:
        per_matrix = [r for rows in pool.map(do_ckpt, jobs) for r in rows]
    assert len(per_matrix) == 63 * 42, len(per_matrix)

    # ---- cross-check against fig1b readouts (k_raw, k_row of the five row-identity families)
    ref = {(r["family"], r["seed"], r["step"], r["comp"], r["layer"]): r for r in B1["per_matrix"]}
    dev = 0.0; n_ck = 0
    for r in per_matrix:
        key = (r["family"], r["seed"], r["step"], r["comp"], r["layer"])
        if key in ref:
            dev = max(dev, abs(r["k_raw"] - ref[key]["k_raw"]), abs(r["k_row"] - ref[key]["k_norm"])); n_ck += 1
    assert n_ck == len(B1["per_matrix"]) == 1890, n_ck
    assert dev < 1e-9, ("fig1b cross-check failed", dev)

    # ---- fit-gate bookkeeping (recorded, not filtered)
    low = [{"family": r["family"], "seed": r["seed"], "step": r["step"], "comp": r["comp"], "layer": r["layer"],
            "min_r2": min(r["r2_raw"], r["r2_row"], r["r2_col"], r["r2_bi"])}
           for r in per_matrix if min(r["r2_raw"], r["r2_row"], r["r2_col"], r["r2_bi"]) < R2_GATE]

    def med(init, step, comp, key, seed=None):
        v = [r[key] for r in per_matrix if r["family"] == init and r["step"] == step and r["comp"] == comp
             and (seed is None or r["seed"] == seed)]
        assert len(v) == (6 if seed else 18), (init, step, comp, seed, len(v))
        return float(np.median(v))

    fam_med = {f"{init}|{c}|{key}": [med(init, s, c, key) for s in STEPS]
               for init in INITS for c in FAM for key in ("k_raw", "k_row", "k_col", "k_bi", "H_row", "H_col")}
    run_med = {f"{init}|s{sd}|{c}|k_bi": [med(init, s, c, "k_bi", sd) for s in STEPS]
               for init in INITS for sd in SEEDS for c in FAM}
    merge = {}
    for c in FAM:
        for init in ("laplace", "uniform"):
            g = fam_med[f"gaussian|{c}|k_bi"]; o = fam_med[f"{init}|{c}|k_bi"]
            hit = [STEPS[i] for i in range(len(STEPS)) if abs(o[i] - g[i]) < MERGE_TOL]
            merge[f"{init}|{c}|k_bi_first_step_within_{MERGE_TOL}_of_gaussian"] = hit[0] if hit else None

    def rng(init, c, key, step):
        v = [r[key] for r in per_matrix if r["family"] == init and r["comp"] == c and r["step"] == step]
        return [float(min(v)), float(max(v))]

    gauss_kbi_all = [r["k_bi"] for r in per_matrix if r["family"] == "gaussian"]
    R = {"schema": "p5v2-ea-kbi-trajectory-v1", "script": Path(__file__).name,
         "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
         "protocol": "weibull_fit_1024 middle-80%; row / column / alternating row-column RMS normalization at fixed global RMS (alternating converged to 0.1%); verbatim p5v2_twoaxis_bridge_v1.py",
         "k0_gaussian_reference": K0, "r2_gate": R2_GATE, "merge_tol": MERGE_TOL,
         "inits": INITS, "seeds": SEEDS, "steps": STEPS, "families": FAM,
         "n_matrices": len(per_matrix), "fig1b_crosscheck_max_abs_dev": dev, "fig1b_crosscheck_n": n_ck,
         "n_fits_below_r2_gate": len(low), "fits_below_r2_gate": low,
         "family_medians": fam_med, "run_medians_k_bi": run_med, "merge_steps_k_bi": merge,
         "step0_family_median_k_bi": {f"{init}|{c}": fam_med[f"{init}|{c}|k_bi"][0] for init in INITS for c in FAM},
         "step0_family_median_k_raw": {f"{init}|{c}": fam_med[f"{init}|{c}|k_raw"][0] for init in INITS for c in FAM},
         "step30000_per_matrix_range_k_bi": {f"{init}|{c}": rng(init, c, "k_bi", STEPS[-1]) for init in INITS for c in FAM},
         "gaussian_k_bi_range_all_steps_all_families": [float(min(gauss_kbi_all)), float(max(gauss_kbi_all))],
         "per_matrix": per_matrix}
    OUT_JSON.parent.mkdir(exist_ok=True, parents=True)
    json.dump(R, open(OUT_JSON, "w"), indent=1)

    # ---- figure: seven panels, k_bi family median (thick) + seeds (thin), style of p5v2_fig1_v1
    C_SEL, C_TRA = "#B2182B", "#2166AC"
    C_INIT = {"gaussian": "0.25", "laplace": C_SEL, "uniform": C_TRA}

    def _sx(v):
        v = np.asarray(v, dtype=float)
        return np.where(v <= 800, v / 800.0, 1.0 + np.log10(np.maximum(v, 800) / 800.0))

    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.4), sharex=True, sharey=True)
    axes = axes.ravel()
    letters = "abcdefg"
    for pi, c in enumerate(FAM):
        ax = axes[pi]
        for init in INITS:
            for sd in SEEDS:
                ax.plot(_sx(STEPS), run_med[f"{init}|s{sd}|{c}|k_bi"], color=C_INIT[init], lw=0.7, alpha=0.55, zorder=3)
            ax.plot(_sx(STEPS), fam_med[f"{init}|{c}|k_bi"], color=C_INIT[init], lw=2.2, zorder=5,
                    marker="o", ms=3.2, mec="white", mew=0.6)
        ax.axhline(K0, color="0.35", lw=1.1, ls=(0, (5, 3)), zorder=1)
        ax.set_title(f"({letters[pi]}) {LAB[c]}: two-axis-normalized shape $k_{{\\mathrm{{bi}}}}$", loc="left", fontsize=10, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.6)
    xt = [0, 800, 3200, 10000, 30000]
    for ax in axes[:7]:
        ax.set_xticks(_sx(xt)); ax.set_xticklabels(["0", "800", "3.2k", "10k", "30k"])
    axes[0].set_xlim(_sx(0) - 0.05, _sx(30000) + 0.05)
    lo = min(min(v) for k, v in fam_med.items() if k.endswith("|k_bi")); hi = max(max(v) for k, v in fam_med.items() if k.endswith("|k_bi"))
    axes[0].set_ylim(lo - 0.02, hi + 0.03)
    for i in (4, 5, 6):
        axes[i].set_xlabel("training step  (linear to 800, log above)", fontsize=9)
    for i in (0, 4):
        axes[i].set_ylabel("Weibull shape  $k_{\\mathrm{bi}}$  (median over layers)", fontsize=9)
    ax = axes[7]; ax.axis("off")
    hb = [Line2D([], [], color=C_INIT["gaussian"], lw=2.2, label="Gaussian init"),
          Line2D([], [], color=C_INIT["laplace"], lw=2.2, label="Laplace init"),
          Line2D([], [], color=C_INIT["uniform"], lw=2.2, label="uniform init"),
          Line2D([], [], color="0.5", lw=2.2, label="thick: family median over 3 seeds x 6 layers"),
          Line2D([], [], color="0.5", lw=0.7, alpha=0.7, label="thin: per-seed median over 6 layers"),
          Line2D([], [], color="0.35", lw=1.1, ls=(0, (5, 3)), label="Gaussian protocol reference %.3f" % K0)]
    ax.legend(handles=hb, loc="center left", fontsize=8.6, frameon=False, handlelength=2.0)
    ax.text(0.0, 0.08, "E-A runs: llama-70M, one data setting,\n3 initialization families x 3 seeds, 30k steps;\n%d fits, %d below the R$^2$ gate (recorded, not filtered)" % (len(per_matrix), len(low)),
            transform=ax.transAxes, fontsize=8.2, color="0.3", va="bottom")
    fig.subplots_adjust(wspace=0.12, hspace=0.32, left=0.06, right=0.99, top=0.94, bottom=0.09)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT_FIG, "and", OUT_JSON)

    # ---- console report
    print("fig1b cross-check n=%d max|dev|=%.2e" % (n_ck, dev))
    print("fits below gate:", len(low), sorted(set((l["family"], l["step"], l["comp"]) for l in low))[:20])
    print("gaussian k_bi range all steps/families:", R["gaussian_k_bi_range_all_steps_all_families"])
    for c in FAM:
        for init in INITS:
            m = fam_med[f"{init}|{c}|k_bi"]
            print(f"{LAB[c]:5s} {init:8s} step0 {m[0]:.3f} 800 {m[1]:.3f} 3200 {m[2]:.3f} 30k {m[-1]:.3f} "
                  f"merge {merge.get(f'{init}|{c}|k_bi_first_step_within_{MERGE_TOL}_of_gaussian', '-')} "
                  f"30k range {[round(x, 3) for x in R['step30000_per_matrix_range_k_bi'][f'{init}|{c}']]} "
                  f"| step0 k_raw {fam_med[f'{init}|{c}|k_raw'][0]:.3f}")
