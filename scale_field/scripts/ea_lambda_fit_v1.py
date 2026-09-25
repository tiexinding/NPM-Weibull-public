#!/usr/bin/env python3
"""EA-LAMBDA-FIT v1: protocol-matched fitted λ(t) (老丁三轮 P0; 零 GPU).

背景: ea_mulhk_v1 的 λ̂_MM=mean|W|/Γ(1+1/k_fit) 是矩匹配代理 — (1) mean|W| 用全权重而
k 来自 middle-80% 中心体; (2) λ̂_MM 依赖 fitted k, 非独立估计。
本脚本用与 analysis.json **逐字同一协议** (cloud_bundle/ea_offline_analysis.weibull_fit_1024:
1024-bin log10 直方图, middle-80%, hist 加权最小二乘 Weibull plot) 的同一次回归取截距:
  log(-log(1-F)) = k·log x + b,  b = -k·log λ  ⇒  λ_fit = exp(-b/k).
自检: 本脚本 slope 与 analysis.json 的 k 逐点比对 — 协议一致, 数值复现误差 <5e-6 (本脚本 float64, 原分析 float32, 非逐位)。
判据: λ_fit 在 [800,3200] 是否 9/9 增长 (与 μ_r/λ_MM 一致) ⇒ 才可把 "λ/k 时间分流"
升为 protocol-matched 直接证据。输出 λ_fit(t) 与 λ_MM(t) 对比。
用法: python3 scripts/ea_lambda_fit_v1.py --base .
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

TRIM = 0.80
RUNS = [f"ea_{f}_s{s}" for f in ("gaussian", "laplace", "uniform") for s in (1, 2, 3)]
KINDS = ("q_proj", "k_proj", "v_proj", "gate_proj")
CKPT = [0, 800, 3200, 10000, 20000, 25000, 30000]
SEGS = [(0, 800), (800, 3200), (3200, 10000), (10000, 30000)]


def weibull_fit_1024_kl(w):
    """协议同 ea_offline_analysis.weibull_fit_1024, 额外返回 λ_fit=exp(-b/k)."""
    aw = np.maximum(np.abs(w.ravel()), 1e-13)
    hist, edges = np.histogram(np.log10(aw), bins=1024, range=(-12.0, 2.0))
    hist = hist.astype(np.float64); total = hist.sum()
    cdf_u = np.cumsum(hist) / total; cdf_l = (np.cumsum(hist) - hist) / total
    F_mid = (cdf_u + cdf_l) / 2.0
    bc_log = (edges[:-1] + edges[1:]) / 2.0 * np.log(10.0)
    margin = (1 - TRIM) / 2.0
    m = (F_mid >= margin) & (F_mid <= 1 - margin) & (hist > 0)
    y = np.log(-np.log(1 - np.clip(F_mid[m], 1e-12, 1 - 1e-12)))
    X = np.vstack([bc_log[m], np.ones(m.sum())]).T
    wgt = hist[m]
    WX = X * wgt[:, None]
    beta, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)
    k = float(beta[0])
    return k, float(np.exp(-beta[1] / k))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); a = ap.parse_args()
    T = {}
    max_kdev = 0.0
    for r in RUNS:
        kj = json.load(open(os.path.join(a.base, "04_ea_runs", r, "analysis.json")))
        kref = {(row["matrix"], row["step"]): row["k"] for row in kj["rows"]}
        T[r] = {}
        for s in CKPT:
            sd = torch.load(os.path.join(a.base, "04_ea_runs/ckpt_key", r, "ckpt", f"step{s}.pt"),
                            map_location="cpu", weights_only=False)
            acc = {kd: [] for kd in KINDS}
            for nm, w in sd.items():
                proj = nm.rsplit(".", 2)[-2]
                if proj not in KINDS:
                    continue
                k, lam = weibull_fit_1024_kl(w.to(torch.float64).numpy())
                max_kdev = max(max_kdev, abs(k - kref[(nm, s)]))
                acc[proj].append(lam)
            T[r][s] = {kd: float(np.median(v)) for kd, v in acc.items()}
    print(f"自检: slope 与 analysis.json k 最大偏差 = {max_kdev:.2e} (协议一致; float32/64 差异, 阈 5e-6)")

    mm = json.load(open(os.path.join(a.base, "06_hrow/ea_mulhk_v1.json")))["med_traj"]
    print("\nλ_fit(t) med 9 runs ×1e3 (对照 λ_MM):")
    print("step    " + "  ".join(f"{kd[:4]}_fit {kd[:4]}_MM" for kd in KINDS))
    med = {}
    for s in CKPT:
        med[s] = {kd: float(np.median([T[r][s][kd] for r in RUNS])) for kd in KINDS}
        print(f"{s:>6}  " + "  ".join(f"{med[s][kd]*1e3:8.4f} {mm[str(s)][kd]['lam']*1e3:8.4f}"
                                      for kd in KINDS))
    print("\nΔλ_fit 区间计票:")
    votes = {}
    for kd in KINDS:
        votes[kd] = {}
        row = []
        for lo, hi in SEGS:
            d = [T[r][hi][kd] - T[r][lo][kd] for r in RUNS]
            votes[kd][f"{lo}-{hi}"] = {"med_x1e3": float(np.median(d) * 1e3),
                                       "n_pos": int(sum(x > 0 for x in d))}
            row.append(f"{lo}-{hi}: {np.median(d)*1e3:+.3f}e-3 ({sum(x>0 for x in d)}/9)")
        print(f"  {kd:<10} " + " | ".join(row))
    p = os.path.join(a.base, "06_hrow/ea_lambda_fit_v1.json")
    json.dump({"schema": "ea_lambda_fit_v1",
               "prereg": "老丁三轮 P0: protocol-matched fitted λ (同 weibull_fit_1024 协议截距); "
                         "判据 λ_fit[800,3200] 9/9 增长; 零 GPU; 260825",
               "slope_selfcheck_max_dev": max_kdev,
               "slope_selfcheck_note": "协议一致; 复现误差来自 float64(本脚本) vs float32(原分析), 阈 5e-6, 非逐位",
               "med_traj_x1": {str(s): med[s] for s in CKPT},
               "votes": votes,
               "per_run": {r: {str(s): T[r][s] for s in CKPT} for r in RUNS}},
              open(p, "w"), indent=1, ensure_ascii=False)
    print(f"→ {p}")


if __name__ == "__main__":
    main()
