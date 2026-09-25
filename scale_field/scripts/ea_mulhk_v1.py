#!/usr/bin/env python3
"""EA-MULHK v1: μ_r / λ̂ / H^W / k 四量同轨 (E-A 9 runs, 零 GPU).

目的 (老丁 8-25 审核建议): 检验 800–3200 抵消窗口内是否出现
  λ (how much) 继续增长  ‖  H^W、k 相对 Δ_spread (where) 近停滞/延迟
即 "训练作用仍在增加, 但功能轴分配暂时抵消" 的最直接动态证据。
量 (per matrix → kind 跨 6 层中位 → 9 runs 计票):
  μ_r(t)  = mean_i log r_i (r_i = 行 RMS), 权重侧直接量;
  H^W(t)  = sd_i(log r_i) (老丁口径; 与 IQR 版一致性另见 ea_kphase);
  k(t)    = 逐矩阵 Weibull shape (复用 04_ea_runs/*/analysis.json 现成拟合);
  λ̂(t)   = mean|W| / Γ(1+1/k(t)) — 矩匹配导出量 (E|W|=λΓ(1+1/k)), 非重新拟合, 照报口径.
段: [0,800] / [800,3200] / [3200,10000] / [10000,30000]; 报每步归一速率便于跨段比较.
用法: python3 scripts/ea_mulhk_v1.py --base .
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from math import gamma

RUNS = [f"ea_{f}_s{s}" for f in ("gaussian", "laplace", "uniform") for s in (1, 2, 3)]
KINDS = ("q_proj", "k_proj", "v_proj", "gate_proj")
CKPT = [0, 800, 3200, 10000, 20000, 25000, 30000]
SEGS = [(0, 800), (800, 3200), (3200, 10000), (10000, 30000)]


def per_step(base, run):
    kj = json.load(open(os.path.join(base, "04_ea_runs", run, "analysis.json")))
    kfit = {(r["matrix"], r["step"]): r["k"] for r in kj["rows"]}
    out = {}
    for s in CKPT:
        sd = torch.load(os.path.join(base, "04_ea_runs/ckpt_key", run, "ckpt", f"step{s}.pt"),
                        map_location="cpu", weights_only=False)
        acc = {kd: {"mu": [], "hw": [], "k": [], "lam": []} for kd in KINDS}
        for nm, w in sd.items():
            proj = nm.rsplit(".", 2)[-2]
            if proj not in KINDS:
                continue
            w = w.to(torch.float64)
            logr = 0.5 * np.log((w ** 2).mean(dim=1).numpy())
            kf = kfit[(nm, s)]
            acc[proj]["mu"].append(float(logr.mean()))
            acc[proj]["hw"].append(float(logr.std()))
            acc[proj]["k"].append(kf)
            acc[proj]["lam"].append(float(w.abs().mean()) / gamma(1.0 + 1.0 / kf))
        out[s] = {kd: {q: float(np.median(v)) for q, v in d.items()} for kd, d in acc.items()}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); a = ap.parse_args()
    T = {r: per_step(a.base, r) for r in RUNS}
    med = {s: {kd: {q: float(np.median([T[r][s][kd][q] for r in RUNS]))
                    for q in ("mu", "hw", "k", "lam")} for kd in KINDS} for s in CKPT}
    print("med 9 runs (kind 跨 6 层中位):")
    for kd in KINDS:
        print(f"\n[{kd}]  step:   μ_r      λ̂×1e3    H^W      k")
        for s in CKPT:
            m = med[s][kd]
            print(f"  {s:>6}  {m['mu']:+.4f}  {m['lam']*1e3:7.4f}  {m['hw']:.4f}  {m['k']:.4f}")
    print("\n区间增量计票 (Δ per seg; rate = Δ/1k 步, 便于跨段比):")
    votes = {}
    for kd in KINDS:
        votes[kd] = {}
        for q in ("mu", "lam", "hw", "k"):
            row = []
            for lo, hi in SEGS:
                d = [T[r][hi][kd][q] - T[r][lo][kd][q] for r in RUNS]
                votes[kd][f"{q}:{lo}-{hi}"] = {"med": float(np.median(d)),
                                               "n_pos": int(sum(x > 0 for x in d)),
                                               "rate_per_1k": float(np.median(d) / (hi - lo) * 1000)}
                row.append(f"{lo}-{hi}: {np.median(d):+.4f}({sum(x>0 for x in d)}/9, r{np.median(d)/(hi-lo)*1000:+.4f})")
            print(f"  {kd:<10} {q:<3} " + " | ".join(row))
    p = os.path.join(a.base, "06_hrow/ea_mulhk_v1.json")
    json.dump({"schema": "ea_mulhk_v1",
               "prereg": "老丁 8-25 审核建议项; exploratory, 零 GPU; λ̂=矩匹配导出 (非重拟合)",
               "med_traj": {str(s): med[s] for s in CKPT}, "votes": votes},
              open(p, "w"), indent=1, ensure_ascii=False)
    print(f"→ {p}")


if __name__ == "__main__":
    main()
