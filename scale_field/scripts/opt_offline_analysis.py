#!/usr/bin/env python
"""OPT 离线分析器 (fork of ea_offline_analysis.py, OPT spec v2 §4.3/§6.6).

拟合核**逐字未动** (口径不变), 只增出三列尺度读数:
  - lam    : Weibull 尺度 λ = exp(−b/k)  (b = 同一 log-log 拟合的截距)
  - fro    : ‖W‖_F
  - rms    : RMS(W) = ‖W‖_F/√n

为什么 fork: ea_offline_analysis.py 是 E-A 九 run 的复现资产 (FREEZE_RECORD.md), 保持不动;
本文件写 analysis_v2.json (不覆盖 analysis.json), 并以"旧六列逐行一致"为门 (verify_offline_parity.py)。
"""
import os, json, glob, argparse, re
import numpy as np
import torch
from scipy.stats import norm

TRIM = 0.80
F_GRID = np.arange(0.15, 0.851, 0.05)
HN_REF = np.log10(norm.ppf((1 + F_GRID) / 2.0) / norm.ppf(0.75))

def weibull_fit_1024(w):
    """与 ea_offline_analysis.weibull_fit_1024 逐字同构, 仅多返回截距 b。"""
    aw = np.maximum(np.abs(w.ravel()), 1e-13)
    hist, edges = np.histogram(np.log10(aw), bins=1024, range=(-12.0, 2.0))
    hist = hist.astype(np.float64); total = hist.sum()
    if total < 100: return float("nan"), float("nan"), float("nan")
    cdf_u = np.cumsum(hist) / total; cdf_l = (np.cumsum(hist) - hist) / total
    F_mid = (cdf_u + cdf_l) / 2.0
    bc_log = (edges[:-1] + edges[1:]) / 2.0 * np.log(10.0)
    margin = (1 - TRIM) / 2.0
    m = (F_mid >= margin) & (F_mid <= 1 - margin) & (hist > 0)
    if m.sum() < 5: return float("nan"), float("nan"), float("nan")
    y = np.log(-np.log(1 - np.clip(F_mid[m], 1e-12, 1 - 1e-12)))
    X = np.vstack([bc_log[m], np.ones(m.sum())]).T
    wgt = hist[m]
    WX = X * wgt[:, None]
    beta, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)
    yhat = X @ beta
    ss_res = float((wgt * (y - yhat) ** 2).sum()); ss_tot = float((wgt * (y - (wgt * y).sum() / wgt.sum()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(beta[0]), r2, float(beta[1])

def qvec(w):
    aw = np.abs(w.ravel()); q = np.quantile(aw, F_GRID); med = np.quantile(aw, 0.5)
    return np.log10(np.maximum(q, 1e-13) / max(med, 1e-13))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--out_name", default="analysis_v2.json")
    a = ap.parse_args()
    ckpts = sorted(glob.glob(f"{a.run}/ckpt/step*.pt"),
                   key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
    rows, q0, W0 = [], {}, {}
    for cp in ckpts:
        step = int(re.search(r"step(\d+)", cp).group(1))
        sd = torch.load(cp, map_location="cpu", weights_only=True)
        for nm, tw in sd.items():
            w = tw.numpy()
            if step == 0: q0[nm] = qvec(w); W0[nm] = w
            k, r2, b = weibull_fit_1024(w)
            qv = qvec(w)
            fro = float(np.linalg.norm(w))
            rows.append({"step": step, "matrix": nm,
                         "kind": next(kk for kk in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj") if kk in nm),
                         "k": k, "R2": r2,
                         "d_init": float(np.sqrt(np.mean((qv - q0[nm]) ** 2))),
                         "d_G": float(np.sqrt(np.mean((qv - HN_REF) ** 2))),
                         "rel_fro": float(np.linalg.norm(w - W0[nm]) / np.linalg.norm(W0[nm])),
                         "cos0": float((w * W0[nm]).sum() / (np.linalg.norm(w) * np.linalg.norm(W0[nm]) + 1e-30)),
                         # ---- OPT spec v2 §4.3 新增尺度读数 ----
                         "lam": float(np.exp(-b / k)) if k == k and k != 0 else float("nan"),
                         "fro": fro,
                         "rms": fro / np.sqrt(w.size)})
        print(f"step{step} done", flush=True)
    json.dump({"schema": "opt-offline-analysis-v2", "run": a.run, "rows": rows},
              open(f"{a.run}/{a.out_name}", "w"), indent=1)
    print(f"DONE -> {a.run}/{a.out_name} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
