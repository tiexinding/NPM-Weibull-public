#!/usr/bin/env python
"""E-A 离线分析器 (spec v2 §5: 主判决值一律离线 1024-bin 标准管线重算).

输入: run 目录 (ea_train.py 产出); 输出: <run>/analysis.json —
每 ckpt 每矩阵: 1024-bin k/R² (uncentered 论文 estimand) / d_init / d_G / 位移 / cos。
拟合核 = npm_weibull.core.weibull._weibull_fit_core 的忠实转写 (加权 lstsq on log-log
Weibull plot, F_mid 裁剪; 转写自 NPM-Weibull-public@213fd40, 数值等价已在本地对照验证)。
"""
import os, json, glob, argparse, re
import numpy as np
import torch
from scipy.stats import norm

TRIM = 0.80
F_GRID = np.arange(0.15, 0.851, 0.05)
HN_REF = np.log10(norm.ppf((1 + F_GRID) / 2.0) / norm.ppf(0.75))

def weibull_fit_1024(w):
    aw = np.maximum(np.abs(w.ravel()), 1e-13)
    hist, edges = np.histogram(np.log10(aw), bins=1024, range=(-12.0, 2.0))
    hist = hist.astype(np.float64); total = hist.sum()
    if total < 100: return float("nan"), float("nan")
    cdf_u = np.cumsum(hist) / total; cdf_l = (np.cumsum(hist) - hist) / total
    F_mid = (cdf_u + cdf_l) / 2.0
    bc_log = (edges[:-1] + edges[1:]) / 2.0 * np.log(10.0)
    margin = (1 - TRIM) / 2.0
    m = (F_mid >= margin) & (F_mid <= 1 - margin) & (hist > 0)
    if m.sum() < 5: return float("nan"), float("nan")
    y = np.log(-np.log(1 - np.clip(F_mid[m], 1e-12, 1 - 1e-12)))
    X = np.vstack([bc_log[m], np.ones(m.sum())]).T
    wgt = hist[m]
    WX = X * wgt[:, None]
    beta, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)
    yhat = X @ beta
    ss_res = float((wgt * (y - yhat) ** 2).sum()); ss_tot = float((wgt * (y - (wgt * y).sum() / wgt.sum()) ** 2).sum())
    return float(beta[0]), 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

def qvec(w):
    aw = np.abs(w.ravel()); q = np.quantile(aw, F_GRID); med = np.quantile(aw, 0.5)
    return np.log10(np.maximum(q, 1e-13) / max(med, 1e-13))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    ckpts = sorted(glob.glob(f"{a.run}/ckpt/step*.pt"),
                   key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    rows, q0, W0 = [], {}, {}
    for cp in ckpts:
        step = int(re.search(r"step(\d+)", cp).group(1))
        sd = torch.load(cp, map_location="cpu", weights_only=True)
        for nm, tw in sd.items():
            w = tw.numpy()
            if step == 0: q0[nm] = qvec(w); W0[nm] = w
            k, r2 = weibull_fit_1024(w)
            qv = qvec(w)
            rows.append({"step": step, "matrix": nm,
                         "kind": next(kk for kk in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj") if kk in nm),
                         "k": k, "R2": r2,
                         "d_init": float(np.sqrt(np.mean((qv - q0[nm]) ** 2))),
                         "d_G": float(np.sqrt(np.mean((qv - HN_REF) ** 2))),
                         "rel_fro": float(np.linalg.norm(w - W0[nm]) / np.linalg.norm(W0[nm])),
                         "cos0": float((w * W0[nm]).sum() / (np.linalg.norm(w) * np.linalg.norm(W0[nm]) + 1e-30))})
        print(f"step{step} done", flush=True)
    json.dump({"schema": "ea-offline-analysis-v1", "run": a.run, "rows": rows},
              open(f"{a.run}/analysis.json", "w"), indent=1)
    print(f"DONE -> {a.run}/analysis.json ({len(rows)} rows)")

if __name__ == "__main__":
    main()
