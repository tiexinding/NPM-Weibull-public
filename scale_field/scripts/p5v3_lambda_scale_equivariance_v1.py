#!/usr/bin/env python3
"""P5: scale equivariance of the fitted Weibull scale (zero GPU).
For every matrix of the IVN base run (LLaMA-style 70M, Gaussian init, seed 1) at the 9 stored steps:
  s      = RMS(W)                       (global scale, second moment)
  k_P, lam_P = middle-80% Weibull-plot fit of |W| (protocol weibull_fit_1024, as analysis.json)
  C_lam  = lam_P(W/s)  (= lam_P / s by equivariance; computed directly as a check)
  lam_2m = s / sqrt(Gamma(1+2/k_P))     (second-moment reference form)
Reports per kind (median over 6 layers): trajectories of log s, log lam_P, log C_lam, k, and per-segment
Delta log lam_P = Delta log s + Delta log C_lam.  Output: 08_paper5_draft/data/p5v3_lambda_scale_equivariance_v1.json
"""
import json, sys, numpy as np, torch
from math import lgamma
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
CK = BASE / "05_ivn_runs/ivn_base/ckpt"; OUT = BASE / "08_paper5_draft/data/p5v3_lambda_scale_equivariance_v1.json"
STEPS = [0, 800, 3200, 5000, 10000, 15000, 20000, 25000, 30000]
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRIM = 0.80
def fit(w):
    aw = np.maximum(np.abs(w.ravel()), 1e-13)
    hist, edges = np.histogram(np.log10(aw), bins=1024, range=(-12.0, 2.0)); hist = hist.astype(np.float64)
    cu = np.cumsum(hist) / hist.sum(); cl = (np.cumsum(hist) - hist) / hist.sum(); F = (cu + cl) / 2
    bc = (edges[:-1] + edges[1:]) / 2 * np.log(10.0); m = (F >= 0.1) & (F <= 0.9) & (hist > 0)
    y = np.log(-np.log(1 - np.clip(F[m], 1e-12, 1 - 1e-12))); X = np.vstack([bc[m], np.ones(m.sum())]).T; WX = X * hist[m][:, None]
    b, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None); k = float(b[0]); return k, float(np.exp(-b[1] / k))
rows = []
for st in STEPS:
    f = CK / f"step{st}.pt"
    if not f.exists(): print("missing", f); continue
    sd = torch.load(f, map_location="cpu"); sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    for name, W in sd.items():
        kind = next((k for k in KINDS if name.endswith(k + ".weight")), None)
        if kind is None or W.ndim != 2: continue
        w = W.float().numpy(); s = float(np.sqrt((w ** 2).mean()))
        k, lam = fit(w); k2, c = fit(w / s)
        rows.append({"step": st, "mat": name, "kind": kind, "s": s, "k": k, "lam": lam, "C_lam": c, "k_norm": k2,
                     "lam_2m": s / np.sqrt(np.exp(lgamma(1 + 2 / k)))})
    print("step", st, "matrices", sum(r["step"] == st for r in rows), flush=True)
steps = sorted(set(r["step"] for r in rows)); med = {}
for kind in KINDS:
    med[kind] = {}
    for st in steps:
        rr = [r for r in rows if r["kind"] == kind and r["step"] == st]
        med[kind][str(st)] = {q: float(np.median([r[q] for r in rr])) for q in ("s", "k", "lam", "C_lam", "lam_2m")}
json.dump({"schema": "p5v3-lambda-scale-equivariance-v1", "run": "ivn_base", "steps": steps, "trim": TRIM, "per_matrix": rows, "kind_medians": med}, open(OUT, "w"), indent=1)
print("\nequivariance check: max |lam_P/s - C_lam| / C_lam =", max(abs(r["lam"] / r["s"] - r["C_lam"]) / r["C_lam"] for r in rows))
print("\nper kind (median over 6 layers): step  s(e-3)  k     lam_P(e-3)  C_lam   lam_P/lam_2m")
for kind in KINDS:
    print("==", kind)
    for st in steps:
        m = med[kind][str(st)]; print(f"   {st:>6} {1e3*m['s']:7.2f} {m['k']:.3f} {1e3*m['lam']:8.2f}  {m['C_lam']:.4f}  {m['lam']/m['lam_2m']:.4f}")
    print("   segments: Δlog lam_P = Δlog s + Δlog C_lam")
    for a, b in zip(steps[:-1], steps[1:]):
        ma, mb = med[kind][str(a)], med[kind][str(b)]
        print(f"   {a:>5}-{b:<5} {np.log(mb['lam']/ma['lam']):+.3f} = {np.log(mb['s']/ma['s']):+.3f} + {np.log(mb['C_lam']/ma['C_lam']):+.3f}")
