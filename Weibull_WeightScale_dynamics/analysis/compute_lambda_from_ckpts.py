#!/usr/bin/env python
"""Fit the Weibull scale lambda(t) from saved checkpoints (Paper #1 middle-80% probability-plot protocol).

For each checkpoint: pool the transmission-class weight magnitudes (attention output + FFN:
o_proj / gate / up / down), fit a middle-80% Weibull, and report (k, lambda, sigma=rms).
lambda = exp(-intercept/k) from the probability plot ln(-ln(1-F)) = k*ln x - k*ln lambda; this is
the Paper #1 Weibull lambda, not the RMS. Writes lambda_trajectory.json:
[{step, k, lambda, sigma, lambda_over_sigma}].

Usage: python compute_lambda_from_ckpts.py --ckpt_dir <ckpt_dir> --arch pythia --out lambda_trajectory.json
"""
import os, re, json, argparse, math
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir", default="ckpts")
ap.add_argument("--out", default="lambda_trajectory.json")
ap.add_argument("--arch", default="llama", choices=["pythia", "llama"])
ap.add_argument("--sub", type=int, default=4000000)  # 池化后子采样上限(加速 fit, 不影响 λ 趋势)
a = ap.parse_args()
TRANS = {"pythia": ["attention.dense", "dense_h_to_4h", "dense_4h_to_h"],
         "llama": ["o_proj", "gate_proj", "up_proj", "down_proj"]}[a.arch]
def is_trans(n): return any(s in n for s in TRANS) and n.endswith(".weight")

def weibull_kl(x):
    """middle-80% 概率图 lstsq → (k, λ). λ=exp(-b/k)."""
    x = np.sort(np.abs(x)); x = x[x > 0]; n = len(x)
    F = (np.arange(1, n + 1) - 0.5) / n
    m = (F >= 0.1) & (F <= 0.9)
    y = np.log(-np.log(1 - F[m])); lx = np.log(x[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    lam = math.exp(-b / k)
    return float(k), float(lam)

steps = sorted(int(re.search(r"step(\d+)", f).group(1)) for f in os.listdir(a.ckpt_dir) if f.startswith("step") and f.endswith(".pt"))
print(f"[lam] {len(steps)} ckpts: {steps[:3]}...{steps[-3:]}", flush=True)
rng = np.random.default_rng(0); rows = []
for s in steps:
    sd = torch.load(f"{a.ckpt_dir}/step{s}.pt", map_location="cpu")
    vals = [sd[nm].flatten().abs().float().numpy() for nm in sd if is_trans(nm)]
    w = np.concatenate(vals)
    sigma = float(np.sqrt(np.mean(w ** 2)))
    if len(w) > a.sub: w = w[rng.choice(len(w), a.sub, replace=False)]
    k, lam = weibull_kl(w)
    rows.append({"step": s, "k": k, "lambda": lam, "sigma": sigma, "lambda_over_sigma": lam / sigma})
    print(f"[lam] step{s:>6} k={k:.3f} lambda={lam:.5f} sigma={sigma:.5f} λ/σ={lam/sigma:.4f}", flush=True)
    del sd, vals, w
json.dump(rows, open(a.out, "w"))
los = [r["lambda_over_sigma"] for r in rows]
print(f"[lam] DONE → {a.out}; λ/σ 范围 {min(los):.4f}-{max(los):.4f} (CV {100*np.std(los)/np.mean(los):.1f}%) = k锁桥稳定性", flush=True)
