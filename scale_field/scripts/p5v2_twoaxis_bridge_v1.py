"""Paper#5 two-axis mixture-bridge check (zero GPU).

Object: W_ij = a_i * b_j * Z_ij  (log|W| = log a_i + log b_j + log|Z_ij|), with a, b the
row/column scale fields obtained by alternating row/column RMS normalization (Sinkhorn-like;
global RMS preserved) and Z the two-axis-normalized core.

Per matrix (TCLOCK grid, 12 runs x 4 steps x 7 families x 6 layers = 2,016 matrices):
  k_raw, k_row, k_col, k_bi (Weibull shape of |W|, row-normalized, column-normalized,
  two-axis-normalized; weibull_fit_1024 middle-80% exactly as kmix_bridge_v1_1),
  H_row = sd_i(ln RMS_i), H_col = sd_j(ln RMS_j), variance decomposition of ln|W|.
Models (through the origin, target D = k_raw^-2 - k_bi^-2):
  M1: a_r H_row^2 ; M2: a_r H_row^2 + a_c H_col^2 ; M3: M2 + a_rc H_row H_col
  baseline: partA target k_raw^-2 - k_row^-2 on H_row^2 (must reproduce partA alpha).
Leave-one-run-out (12 folds) / leave-one-arm-out (4 folds) MAE of predicted k_raw, as loo_mae
in kmix_bridge_v1_1.py. Transpose self-check: H_row<->H_col swap, k_raw/k_bi invariant,
prediction invariance <=> a_r ~ a_c. Second dataset: public Pythia 348 matrices (M2 only).

Sources (read-only): 04_tclock_runs/wt_ckpts/tclock_D{1..4}_s{1,2,3}/step{800,1600,3200,5000}.pt
                     06_hrow/kmix_bridge_v1_1.json (partA_audit_rows, cross-check)
                     08_paper5_draft/data/p5_public_pythia_knorm_v1.json
Outputs: 08_paper5_draft/data/p5v2_twoaxis_bridge_v1.json, figures/P5v2_S8_twoaxis_bridge.png
"""
import hashlib
import json
import re
import sys
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
WT = ROOT / "04_tclock_runs/wt_ckpts"
OUT_JSON = ROOT / "08_paper5_draft/data/p5v2_twoaxis_bridge_v1.json"
OUT_FIG = ROOT / "08_paper5_draft/figures/P5v2_S8_twoaxis_bridge.png"
PARTA = ROOT / "06_hrow/kmix_bridge_v1_1.json"
PUBLIC = ROOT / "08_paper5_draft/data/p5_public_pythia_knorm_v1.json"

HB, HR = 1024, (-12.0, 2.0)
ARMS, SEEDS, STEPS = ["D1", "D2", "D3", "D4"], [1, 2, 3], [800, 1600, 3200, 5000]
FAM = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FAM_A = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]          # partA families
LAB = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
R2_GATE = 0.99


# ------------------------------------------------------------------ protocol (verbatim kmix_bridge_v1_1)
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
    """Alternating row/column normalization; returns Z and the diagonal scales a (rows), b (cols)
    such that W = diag(a) Z diag(b) exactly (up to float round-off)."""
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
    # exact log decomposition on the entries: ln|W| = ln a_i + ln b_j + ln|Z_ij|
    lw = np.log(np.maximum(np.abs(w), 1e-13)); la = np.log(a); lb = np.log(b); lz = np.log(np.maximum(np.abs(z), 1e-13))
    A = np.broadcast_to(la[:, None], w.shape); B = np.broadcast_to(lb[None, :], w.shape)
    var_w = float(lw.var()); var_a = float(la.var()); var_b = float(lb.var()); var_z = float(lz.var())
    cov_az = float(((A - A.mean()) * (lz - lz.mean())).mean()); cov_bz = float(((B - B.mean()) * (lz - lz.mean())).mean())
    cov_ab = float(((A - A.mean()) * (B - B.mean())).mean())      # zero by construction (separable grid)
    return {"k_raw": kr, "k_row": kw, "k_col": kc, "k_bi": kb,
            "r2_raw": r2r, "r2_row": r2w, "r2_col": r2c, "r2_bi": r2b,
            "H_row": hr, "H_col": hc, "sd_lna": float(la.std()), "sd_lnb": float(lb.std()),
            "var_lnW": var_w, "var_lna": var_a, "var_lnb": var_b, "var_lnZ": var_z,
            "cov_terms_2x": 2 * (cov_az + cov_bz + cov_ab),
            "resid_share": float((var_w - var_a - var_b - var_z) / var_w)}


def do_ckpt(args):
    arm, sd, step = args
    st = torch.load(WT / f"tclock_{arm}_s{sd}/step{step}.pt", map_location="cpu", weights_only=True)
    rows = []
    for nm, w in st.items():
        mm = re.search(r"\.(\w+_proj)\.weight$", nm)
        if not mm or mm.group(1) not in FAM:
            continue
        L = int(re.search(r"layers\.(\d+)", nm).group(1))
        d = analyse(w.double().numpy())
        d.update({"arm": arm, "seed": sd, "step": step, "kind": mm.group(1), "layer": L, "mat": nm, "shape": list(w.shape)})
        rows.append(d)
    assert len(rows) == 42, (args, len(rows))
    return rows


# ------------------------------------------------------------------ models
def design(rows, model, swap=False):
    hr = np.array([r["H_col" if swap else "H_row"] for r in rows]); hc = np.array([r["H_row" if swap else "H_col"] for r in rows])
    cols = {"M1": [hr ** 2], "M2": [hr ** 2, hc ** 2], "M3": [hr ** 2, hc ** 2, hr * hc]}[model]
    return np.vstack(cols).T


def target(rows, base="k_bi"):
    return np.array([r["k_raw"] ** -2 - r[base] ** -2 for r in rows])


def origin_fit(rows, model, base="k_bi"):
    X = design(rows, model); y = target(rows, base)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r0 = float(1 - ((y - pred) ** 2).sum() / (y ** 2).sum())
    kb = np.array([r[base] for r in rows]); kr = np.array([r["k_raw"] for r in rows])
    mae = float(np.mean(np.abs((kb ** -2 + pred) ** -0.5 - kr)))
    return beta, r0, mae


def loo_mae(rows, model, key, base="k_bi"):
    groups = sorted(set(key(r) for r in rows)); errs = {}
    for g in groups:
        tr = [r for r in rows if key(r) != g]; te = [r for r in rows if key(r) == g]
        beta = origin_fit(tr, model, base)[0]
        pred = design(te, model) @ beta
        kb = np.array([r[base] for r in te]); kr = np.array([r["k_raw"] for r in te])
        errs[str(g)] = float(np.mean(np.abs((kb ** -2 + pred) ** -0.5 - kr)))
    v = np.array(list(errs.values()))
    return {"per_group": errs, "median": float(np.median(v)), "max": float(v.max()),
            "max_group": max(errs, key=errs.get)}


def fit_block(rows, label):
    out = {"n": len(rows)}
    for m in ("M1", "M2", "M3"):
        beta, r0, mae = origin_fit(rows, m)
        run_betas = [origin_fit([r for r in rows if (r["arm"], r["seed"]) == rr], m)[0]
                     for rr in sorted(set((r["arm"], r["seed"]) for r in rows))]
        rb = np.array(run_betas)
        # transpose self-check: swap H_row/H_col, keep k_raw/k_bi, same coefficients
        Xs = design(rows, m, swap=True); y = target(rows)
        pred = design(rows, m) @ beta; pred_T = Xs @ beta
        kb = np.array([r["k_bi"] for r in rows]); kr = np.array([r["k_raw"] for r in rows])
        mae_T = float(np.mean(np.abs((kb ** -2 + pred_T) ** -0.5 - kr)))
        out[m] = {"coef": [float(x) for x in beta], "coef_names": {"M1": ["a_r"], "M2": ["a_r", "a_c"], "M3": ["a_r", "a_c", "a_rc"]}[m],
                  "R0_sq": r0, "mae_insample": mae,
                  "run_coef_range": [[float(x) for x in rb.min(0)], [float(x) for x in rb.max(0)]],
                  "runs_same_sign_per_coef": [int((np.sign(rb[:, i]) == np.sign(beta[i])).sum()) for i in range(len(beta))],
                  "pred_mae_LORO": loo_mae(rows, m, lambda r: (r["arm"], r["seed"])),
                  "pred_mae_LOAO": loo_mae(rows, m, lambda r: r["arm"]),
                  "transpose_check": {"mae_insample_transposed_fields": mae_T,
                                      "max_abs_pred_change": float(np.abs(pred - pred_T).max()),
                                      "a_r_over_a_c": (float(beta[0] / beta[1]) if m != "M1" and beta[1] != 0 else None)}}
    return out


if __name__ == "__main__":
    jobs = list(product(ARMS, SEEDS, STEPS))
    assert len(jobs) == 48
    with Pool(8) as pool:
        rows = [r for rr in pool.map(do_ckpt, jobs) for r in rr]
    assert len(rows) == 48 * 42, len(rows)

    # ---- cross-check against partA (q/k/v/gate/up: k_raw, k_norm=k_row, hw=H_row)
    A = json.load(open(PARTA))["partA_audit_rows"]
    assert len(A) == 1440
    idx = {(r["arm"], r["seed"], r["step"], r["mat"]): r for r in rows}
    dev = []
    for a in A:
        r = idx[(a["arm"], a["seed"], a["step"], a["mat"])]
        dev.append(max(abs(r["k_raw"] - a["k_raw"]), abs(r["k_row"] - a["k_norm"]), abs(r["H_row"] - a["hw"])))
    max_dev = float(max(dev))
    assert max_dev < 1e-6, ("partA mismatch", max_dev)

    # ---- fit gate bookkeeping (recorded, not filtered)
    below = [{"mat": r["mat"], "arm": r["arm"], "seed": r["seed"], "step": r["step"],
              "r2": [r["r2_raw"], r["r2_row"], r["r2_col"], r["r2_bi"]]} for r in rows
             if min(r["r2_raw"], r["r2_row"], r["r2_col"], r["r2_bi"]) < R2_GATE]

    # ---- baseline: partA target & model on partA families (must reproduce alpha_P)
    qk = [r for r in rows if r["kind"] in ("q_proj", "k_proj")]
    base_alpha, base_r0, base_mae = origin_fit(qk, "M1", base="k_row")
    pa = json.load(open(PARTA))["partA"]["qk_pooled"]
    assert abs(base_alpha[0] - pa["alpha"]) < 1e-9 and abs(base_r0 - pa["R0_sq"]) < 1e-9, (base_alpha, pa["alpha"])

    # ---- k_bi range per family/step
    kbi_range = {f: {str(s): [float(min(r["k_bi"] for r in rows if r["kind"] == f and r["step"] == s)),
                              float(max(r["k_bi"] for r in rows if r["kind"] == f and r["step"] == s))] for s in STEPS} for f in FAM}
    kbi_median = {f: float(np.median([r["k_bi"] for r in rows if r["kind"] == f])) for f in FAM}
    krow_median = {f: float(np.median([r["k_row"] for r in rows if r["kind"] == f])) for f in FAM}
    kcol_median = {f: float(np.median([r["k_col"] for r in rows if r["kind"] == f])) for f in FAM}
    H_median = {f: {"H_row": float(np.median([r["H_row"] for r in rows if r["kind"] == f])),
                    "H_col": float(np.median([r["H_col"] for r in rows if r["kind"] == f]))} for f in FAM}

    # ---- model fits
    fits = {"pooled_all7": fit_block(rows, "all"),
            "pooled_partA5": fit_block([r for r in rows if r["kind"] in FAM_A], "partA5"),
            "qk_pooled": fit_block(qk, "qk")}
    for f in FAM:
        fits[f] = fit_block([r for r in rows if r["kind"] == f], f)

    # ---- per-family residual under the pooled-all M2 (is v / o / down an exception?)
    beta_all = np.array(fits["pooled_all7"]["M2"]["coef"]); beta_all1 = np.array(fits["pooled_all7"]["M1"]["coef"])
    fam_resid = {}
    for f in FAM:
        sub = [r for r in rows if r["kind"] == f]
        kb = np.array([r["k_bi"] for r in sub]); kr = np.array([r["k_raw"] for r in sub])
        p2 = design(sub, "M2") @ beta_all; p1 = design(sub, "M1") @ beta_all1
        fam_resid[f] = {"mae_pooledM2": float(np.mean(np.abs((kb ** -2 + p2) ** -0.5 - kr))),
                        "mae_pooledM1": float(np.mean(np.abs((kb ** -2 + p1) ** -0.5 - kr))),
                        "mean_signed_resid_M2_k": float(np.mean((kb ** -2 + p2) ** -0.5 - kr)),
                        "n": len(sub)}

    # ---- variance decomposition summary
    vdec = {f: {"resid_share_median": float(np.median([r["resid_share"] for r in rows if r["kind"] == f])),
                "resid_share_range": [float(min(r["resid_share"] for r in rows if r["kind"] == f)),
                                      float(max(r["resid_share"] for r in rows if r["kind"] == f))],
                "share_lna_median": float(np.median([r["var_lna"] / r["var_lnW"] for r in rows if r["kind"] == f])),
                "share_lnb_median": float(np.median([r["var_lnb"] / r["var_lnW"] for r in rows if r["kind"] == f])),
                "share_lnZ_median": float(np.median([r["var_lnZ"] / r["var_lnW"] for r in rows if r["kind"] == f]))} for f in FAM}

    # ---- public Pythia: M2 on target k_raw^-2 - k_both^-2
    P = json.load(open(PUBLIC))["rows"]; assert len(P) == 348
    prow = [{"k_raw": r["k_raw"], "k_bi": r["k_both"], "H_row": r["H_row"], "H_col": r["H_col"], "kind": r["kind"], "size": r["size"],
             "arm": r["size"], "seed": 0} for r in P]
    pub = {}
    for m in ("M1", "M2", "M3"):
        beta, r0, mae = origin_fit(prow, m)
        pub[m] = {"coef": [float(x) for x in beta], "R0_sq": r0, "mae_insample": mae,
                  "pred_mae_leave_one_size_out": loo_mae(prow, m, lambda r: r["size"])}
    pub_fam = {}
    for kd in sorted(set(r["kind"] for r in prow)):
        sub = [r for r in prow if r["kind"] == kd]
        b2 = np.array(pub["M2"]["coef"]); kb = np.array([r["k_bi"] for r in sub]); kr = np.array([r["k_raw"] for r in sub])
        pub_fam[kd] = {"n": len(sub), "mae_pooledM2": float(np.mean(np.abs((kb ** -2 + design(sub, "M2") @ b2) ** -0.5 - kr))),
                       "k_both_median": float(np.median(kb))}

    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    R = {"schema": "p5v2-twoaxis-bridge-v1", "script": Path(__file__).name, "script_sha256": sha,
         "protocol": "weibull_fit_1024 middle-80% (kmix_bridge_v1_1); H = sd(ln RMS) over rows / columns; "
                     "two-axis normalization = alternating row/column RMS normalization to <1e-6 rel. flatness, global RMS preserved; "
                     "W = diag(a) Z diag(b) exact; target D = k_raw^-2 - k_bi^-2; models through the origin",
         "n_matrices": len(rows), "families": FAM, "steps": STEPS, "arms": ARMS, "seeds": SEEDS,
         "partA_crosscheck_max_abs_dev": max_dev, "baseline_partA_reproduced": {"alpha_qk": float(base_alpha[0]), "R0_sq": base_r0, "mae": base_mae},
         "r2_gate": R2_GATE, "n_fits_below_gate": len(below), "fits_below_gate": below,
         "k_bi_range_by_family_step": kbi_range, "k_bi_median": kbi_median, "k_row_median": krow_median, "k_col_median": kcol_median,
         "H_median": H_median, "fits": fits, "family_residual_under_pooled_all7": fam_resid,
         "variance_decomposition": vdec, "public_pythia": {"models": pub, "per_family": pub_fam},
         "per_matrix": rows}
    json.dump(R, open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_JSON)

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8})
    COL = {"q_proj": "#B2182B", "k_proj": "#E08214", "v_proj": "#1B7837", "o_proj": "#762A83",
           "gate_proj": "#2166AC", "up_proj": "#5AAE61", "down_proj": "#8C510A"}
    MRK = {"q_proj": "o", "k_proj": "s", "v_proj": "^", "o_proj": "P", "gate_proj": "D", "up_proj": "v", "down_proj": "X"}
    fig = plt.figure(figsize=(13.4, 4.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.32)
    # (a) D vs pooled-all M2 prediction
    ax = fig.add_subplot(gs[0, 0])
    y_all = target(rows); p_all = design(rows, "M2") @ beta_all
    for f in FAM:
        m = np.array([r["kind"] == f for r in rows])
        ax.scatter(p_all[m], y_all[m], s=11, marker=MRK[f], color=COL[f], alpha=0.65, lw=0.4, edgecolor="white", label=LAB[f])
    lim = float(max(y_all.max(), p_all.max())) * 1.05
    ax.plot([0, lim], [0, lim], color="0.4", lw=1, ls=(0, (5, 3)))
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"pooled two-axis prediction  $a_r (H_{\mathrm{row}})^2 + a_c (H_{\mathrm{col}})^2$")
    ax.set_ylabel(r"$k_{\mathrm{raw}}^{-2} - k_{\mathrm{bi}}^{-2}$")
    ax.set_title("(a) Two-axis mixture term vs pooled fit", loc="left", pad=7)
    ax.legend(frameon=False, ncol=2, loc="lower right", handletextpad=0.3, columnspacing=0.8)
    ax.text(0.02, 0.97, "n = %d, 4 arms x 3 seeds x 4 steps x 7 families x 6 layers\n$a_r$ = %.2f, $a_c$ = %.2f, $R_0^2$ = %.3f"
            % (len(rows), beta_all[0], beta_all[1], fits["pooled_all7"]["M2"]["R0_sq"]), transform=ax.transAxes, va="top", fontsize=8, color="0.3")
    # (b) a_r, a_c per family with 12-run range (M2, per-family fit)
    ax = fig.add_subplot(gs[0, 1])
    ys = np.arange(len(FAM))[::-1]
    for i, f in enumerate(FAM):
        m2 = fits[f]["M2"]; lo, hi = m2["run_coef_range"]
        ax.plot([lo[0], hi[0]], [ys[i] + 0.16] * 2, color=COL[f], lw=1.6, alpha=0.5)
        ax.plot(m2["coef"][0], ys[i] + 0.16, marker="o", color=COL[f], ms=6, mec="white")
        ax.plot([lo[1], hi[1]], [ys[i] - 0.16] * 2, color=COL[f], lw=1.6, alpha=0.5, ls=(0, (3, 2)))
        ax.plot(m2["coef"][1], ys[i] - 0.16, marker="s", mfc="white", color=COL[f], ms=6)
    ax.axvline(0, color="0.5", lw=0.8)
    XL = 2.5
    ax.set_xlim(-XL, XL)
    clipped = [LAB[f] for f in FAM if min(fits[f]["M2"]["run_coef_range"][0]) < -XL or max(fits[f]["M2"]["run_coef_range"][1]) > XL]
    if clipped:
        ax.text(0.02, 0.02, "run ranges clipped at $\\pm$%.1f: %s" % (XL, ", ".join(clipped)), transform=ax.transAxes, fontsize=7.5, color="0.35")
    ax.set_yticks(ys); ax.set_yticklabels([LAB[f] for f in FAM])
    ax.set_xlabel(r"coefficient (per-family M2, through the origin)")
    ax.set_title("(b) Row and column coefficients per family", loc="left", pad=7)
    hb = [Line2D([], [], marker="o", color="0.3", ms=6, lw=1.6, label=r"$a_r$ (rows); bar = range over 12 runs"),
          Line2D([], [], marker="s", mfc="white", color="0.3", ms=6, lw=1.6, ls=(0, (3, 2)), label=r"$a_c$ (columns)")]
    ax.legend(handles=hb, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.62))
    # (c) LORO median MAE for M1/M2/M3 per family
    ax = fig.add_subplot(gs[0, 2])
    w = 0.26; x = np.arange(len(FAM))
    for j, (m, c, lab) in enumerate((("M1", "0.6", r"M1: $H_{\mathrm{row}}$ only"), ("M2", "#2166AC", r"M2: + $H_{\mathrm{col}}$"), ("M3", "#B2182B", r"M3: + cross term"))):
        med = [fits[f][m]["pred_mae_LORO"]["median"] for f in FAM]; mx = [fits[f][m]["pred_mae_LORO"]["max"] for f in FAM]
        ax.bar(x + (j - 1) * w, med, w, color=c, alpha=0.85, label=lab)
        ax.plot(x + (j - 1) * w, mx, ls="none", marker="_", color="0.15", ms=7, mew=1.2)
    ax.set_xticks(x); ax.set_xticklabels([LAB[f] for f in FAM])
    ax.set_ylabel(r"leave-one-run-out MAE of predicted $k_{\mathrm{raw}}$")
    ax.set_title("(c) Held-out error by model and family", loc="left", pad=7)
    ax.legend(frameon=False, loc="upper left", title="bar: median of 12 folds; tick: worst fold", title_fontsize=8)
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT_FIG)

    # ------------------------------------------------------------------ console summary
    print("partA max dev", max_dev, "| baseline alpha_qk", base_alpha[0], "| fits below gate", len(below))
    print("k_bi median", {LAB[f]: round(v, 4) for f, v in kbi_median.items()})
    print("H median", {LAB[f]: (round(v["H_row"], 3), round(v["H_col"], 3)) for f, v in H_median.items()})
    for key in ("pooled_all7", "pooled_partA5", "qk_pooled") + tuple(FAM):
        b = fits[key]
        print(f"{key:14s}", {m: (["%.3f" % c for c in b[m]["coef"]], "R0 %.3f" % b[m]["R0_sq"], "in %.4f" % b[m]["mae_insample"],
                                 "LORO %.4f/%.4f" % (b[m]["pred_mae_LORO"]["median"], b[m]["pred_mae_LORO"]["max"]),
                                 "LOAO %.4f/%.4f" % (b[m]["pred_mae_LOAO"]["median"], b[m]["pred_mae_LOAO"]["max"]),
                                 "T %.4f" % b[m]["transpose_check"]["mae_insample_transposed_fields"]) for m in ("M1", "M2", "M3")})
    print("family resid under pooled-all M1/M2", {LAB[f]: ("%.4f" % v["mae_pooledM1"], "%.4f" % v["mae_pooledM2"]) for f, v in fam_resid.items()})
    print("var decomposition resid share median", {LAB[f]: "%.4f" % v["resid_share_median"] for f, v in vdec.items()})
    print("public", {m: (["%.3f" % c for c in pub[m]["coef"]], "R0 %.3f" % pub[m]["R0_sq"], "in %.4f" % pub[m]["mae_insample"]) for m in pub},
          {k: "%.4f" % v["mae_pooledM2"] for k, v in pub_fam.items()})
