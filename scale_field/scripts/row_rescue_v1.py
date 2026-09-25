#!/usr/bin/env python3
"""行尺度归一化救援实验 (260821 老丁提议; 零 GPU, 判别"行尺度混合→k 偏离"直接统计机制).

方法: 对终端窗 3 ckpt {20000,25000,30000} 每权重矩阵执行 W̃_i,: = W_i,: · (RMS_mat/RMS_row_i)
(全行归一到同一 RMS; k/d_G 对整体尺度不变, 归一常数只为数值稳定), 用 opt_offline_analysis
的拟合核 (weibull_fit_1024 / qvec vs HN_REF, 逐字同构) 重拟 k, d_G。
聚合口径同 a3_dose_judgment_v1: kind 组内各矩阵中位 (qk=12 矩阵合并) → 三步中位;
k_FFN = mean(gate,up,down); Δ_spread = k_FFN − k_qk。
剂量点: λ ∈ {0 (A1), 0.03, 0.1 (B0 = EA 3 seed, 逐 seed 报+中位), 0.3}; 两族分开。
预期 (登记于跑前): 若"行尺度混合导致 k 偏离"成立 → 归一后 k_qk → ~1.20, FFN 几乎不变,
各剂量归一后曲线趋于重合 (Δ_spread_rescued ≈ 平坦)。
用法:
  python3 row_rescue_v1.py --selftest
  python3 row_rescue_v1.py --base . --out 06_hrow/row_rescue_v1.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cloud_bundle"))
from opt_offline_analysis import weibull_fit_1024, qvec, HN_REF

TERM = [20000, 25000, 30000]
FAMS = ("gaussian", "laplace")
DOSES = (0.0, 0.03, 0.1, 0.3)
GROUPS = {"gate": ["gate_proj"], "up": ["up_proj"], "down": ["down_proj"],
          "o": ["o_proj"], "qk": ["q_proj", "k_proj"], "v": ["v_proj"]}


def row_normalize(w):
    rms_row = np.sqrt((w ** 2).mean(axis=1, keepdims=True))
    rms_mat = np.sqrt((w ** 2).mean())
    return w * (rms_mat / np.maximum(rms_row, 1e-30))


def fit_kd(w):
    """(k, d_G, λ_fit, RMS). v1.1: 补报拟合 λ 与 RMS (评审 260821: 防把"全局 RMS 不变"
    误写成"拟合 λ 必然不变" — 两参数拟合中 k 与 λ 会相互调整)."""
    k, r2, b = weibull_fit_1024(w)
    qv = qvec(w)
    lam = float(np.exp(-b / k)) if k == k and k != 0 else float("nan")
    return (k, float(np.sqrt(np.mean((qv - HN_REF) ** 2))), lam,
            float(np.sqrt((np.asarray(w) ** 2).mean())))


def rescue_run(ckpt_dir):
    """终端窗 3 ckpt: 逐矩阵 before/after 拟合 → kind 组中位 → 三步中位.
    断言: 行归一严格保持全矩阵 RMS (各行等长 → 解析成立, 数值容差 1e-12 相对)."""
    per_step = {kd: [] for kd in GROUPS}
    for s in TERM:
        sd = torch.load(os.path.join(ckpt_dir, f"step{s}.pt"),
                        map_location="cpu", weights_only=True)
        vals = {kd: [] for kd in GROUPS}
        for nm, tw in sd.items():
            proj = nm.rsplit(".", 2)[-2]
            kd = next((g for g, subs in GROUPS.items() if proj in subs), None)
            if kd is None:
                continue
            w = tw.numpy().astype(np.float64)
            kb, dgb, lamb, rmsb = fit_kd(w)
            ka, dga, lama, rmsa = fit_kd(row_normalize(w))
            assert abs(rmsa - rmsb) <= 1e-12 * rmsb, (nm, s, rmsb, rmsa)
            vals[kd].append((ka, dga, kb, dgb, lamb, lama, rmsb))
        for kd in GROUPS:
            per_step[kd].append(tuple(float(np.median([v[i] for v in vals[kd]]))
                                      for i in range(7)))
    out = {}
    for kd in GROUPS:
        med = tuple(float(np.median([t[i] for t in per_step[kd]])) for i in range(7))
        out[kd] = {"k": med[0], "d_G": med[1], "k_before": med[2], "d_G_before": med[3],
                   "lam_before": med[4], "lam_after": med[5], "rms": med[6]}
    out["k_FFN"] = float(np.mean([out[g]["k"] for g in ("gate", "up", "down")]))
    out["k_FFN_before"] = float(np.mean([out[g]["k_before"] for g in ("gate", "up", "down")]))
    out["spread"] = out["k_FFN"] - out["qk"]["k"]
    out["spread_before"] = out["k_FFN_before"] - out["qk"]["k_before"]
    return out


def selftest():
    rng = np.random.default_rng(0)
    # ① iid 高斯矩阵: 归一前后 k 均 ≈1.20 (拟合核 sanity)
    w = rng.normal(0, 0.02, (512, 512))
    k0 = fit_kd(w)[0]; k1 = fit_kd(row_normalize(w))[0]
    assert abs(k0 - 1.205) < 0.03 and abs(k1 - 1.205) < 0.03, (k0, k1)
    # ② 行尺度混合矩阵 (σ_i 双峰 1:4): 归一前 k 显著低, 归一后回 ~1.20 (可 FAIL 双向)
    sig = np.where(rng.random(512) < 0.5, 0.01, 0.04)[:, None]
    wm = rng.normal(0, 1, (512, 512)) * sig
    km0 = fit_kd(wm)[0]; km1 = fit_kd(row_normalize(wm))[0]
    assert km0 < 1.05, km0            # 混合必须压低 k, 否则本实验无判别力
    assert abs(km1 - 1.205) < 0.03, km1
    print(f"SELFTEST PASS: iid {k0:.3f}->{k1:.3f}; mixture {km0:.3f}->{km1:.3f} (归一救回)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--out", default="06_hrow/row_rescue_v1.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    base = a.base
    points = {}
    for fam in FAMS:
        points[fam] = {}
        for dose, dirs in ((0.0, [f"05_opt_runs/opt_a1_{fam}_s1/ckpt"]),
                           (0.03, [f"05_opt_runs/opt_a3w003_{fam}_s1/ckpt"]),
                           (0.1, [f"04_ea_runs/ckpt_key/ea_{fam}_s{s}/ckpt" for s in (1, 2, 3)]),
                           (0.3, [f"05_opt_runs/opt_a3w030_{fam}_s1/ckpt"])):
            per = [dict(rescue_run(os.path.join(base, d)), dir=d) for d in dirs]
            med = {"k_qk": float(np.median([p["qk"]["k"] for p in per])),
                   "k_FFN": float(np.median([p["k_FFN"] for p in per])),
                   "spread": float(np.median([p["spread"] for p in per])),
                   "d_G_qk": float(np.median([p["qk"]["d_G"] for p in per])),
                   "d_G_FFN_med3": float(np.median([np.median([p[g]["d_G"] for g in ("gate", "up", "down")]) for p in per]))}
            points[fam][dose] = {"per_run": per, "rescued": med}
            print(f"  {fam} λ={dose}: rescued k_qk={med['k_qk']:.4f} k_FFN={med['k_FFN']:.4f} "
                  f"spread={med['spread']:.4f} d_G_qk={med['d_G_qk']:.5f}")
    json.dump({"schema": "row_rescue_v1.1", "norm": "row -> matrix RMS",
               "fit": "opt_offline_analysis.weibull_fit_1024 (逐字同构)",
               "aggregation": "a3_dose_judgment_v1 口径", "doses": list(DOSES),
               "points": points},
              open(os.path.join(base, a.out), "w"), indent=1, ensure_ascii=False)
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
