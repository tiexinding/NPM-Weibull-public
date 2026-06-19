#!/usr/bin/env python
"""预测 λ vs 实测 λ 闭环图.
预测 λ: 用记录的三项力前向积分 ΣW²(t) → σ(t) → λ_pred = σ·(λ/σ k锁桥比) [三项力推导路径].
实测 λ: ckpt 上 Paper#1 Weibull 拟合 (独立测量, lambda_trajectory).
画两条 + 残差 + logRMSE。诚实: 预测用实测力积分(主方程恒等式), 实测λ是独立Weibull拟合 →
误差=数值积分离散化 + k锁桥近似 (端到端), 非纯记账。
Usage: python make_predicted_lambda_fig.py --true <v1b_spline_true.jsonl> --lam <lambda_trajectory.json> --out F.png
"""
import json, argparse, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--true", default="derived_data/self_train/v1b_spline_true_llama.jsonl")
ap.add_argument("--lam", default="derived_data/self_train/lambda_trajectory_llama.json")
ap.add_argument("--out", default="F_lambda_pred_vs_obs.png")
ap.add_argument("--label", default="Llama-70m")
ap.add_argument("--lwd", type=float, default=0.01)
a = ap.parse_args()

tr = [json.loads(l) for l in open(a.true) if l.strip()]
lam = json.load(open(a.lam))
# k锁桥比 λ/σ (实测均值)
los = np.mean([r["lambda_over_sigma"] for r in lam])

# 从力恢复 N (dec_full = 2η·λwd·N·σ² → N = dec_full/(2η·λwd·rms²))
r0 = tr[5]
N = r0["dec_full"] / (2 * r0["eta"] * a.lwd * r0["rms"] ** 2)
# 前向积分 ΣW²: 每步 Δ(ΣW²) = −align + inj − dec (绝对Σ单位), rec 间隔 dt=step差
steps = [r["step"] for r in tr]
SW2 = [tr[0]["rms"] ** 2 * N]  # 初值 = 实测 σ²·N
for i in range(1, len(tr)):
    dt = steps[i] - steps[i - 1]
    pr = tr[i - 1]
    dperstep = (-pr["true_align_full"]) + pr["inj_full"] - pr["dec_full"]  # 每优化步 Δ(ΣW²)
    SW2.append(SW2[-1] + dperstep * dt)
sig_pred = np.sqrt(np.array(SW2) / N)
lam_pred = sig_pred * los          # 三项力推导 λ (经 k锁桥)
sig_obs_inline = np.array([r["rms"] for r in tr])  # 内联实测 σ (参考)

# 实测 Weibull-λ (ckpt grid) → 插值到 tr 的 step 网格 比较
lk_step = np.array([r["step"] for r in lam]); lk_lam = np.array([r["lambda"] for r in lam])
lam_obs_on_tr = np.interp(steps, lk_step, lk_lam)
mask = np.array(steps) >= lk_step.min()  # 只在实测λ覆盖区比
logrmse = np.sqrt(np.mean((np.log(lam_pred[mask]) - np.log(lam_obs_on_tr[mask])) ** 2))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
# 左: 预测 vs 实测 λ
ax[0].plot(steps, lam_pred, "-", c="C0", lw=2, label="predicted λ (three-force integration → k-lock bridge)")
ax[0].plot(lk_step, lk_lam, "o", c="C3", ms=5, label="observed λ (Weibull fit on checkpoints)")
ax[0].set_xlabel("training step"); ax[0].set_ylabel("Weibull λ"); ax[0].set_title("Predicted vs observed λ(t)")
ax[0].legend(fontsize=8); ax[0].text(0.04, 0.9, f"logRMSE = {logrmse:.3f}", transform=ax[0].transAxes, fontsize=10, color="k")
# 右: 残差
resid = 100 * (lam_pred[mask] - lam_obs_on_tr[mask]) / lam_obs_on_tr[mask]
ax[1].axhline(0, c="gray", lw=0.8); ax[1].plot(np.array(steps)[mask], resid, "-", c="C2")
ax[1].set_xlabel("training step"); ax[1].set_ylabel("relative error (%)"); ax[1].set_title("Residual: (predicted − observed)/observed")
ax[1].text(0.04, 0.9, f"max |err| = {np.max(np.abs(resid)):.1f}%", transform=ax[1].transAxes, fontsize=9)
fig.suptitle("Three-force-predicted λ vs Weibull-fit λ (self-trained "+a.label+")", fontsize=12)
fig.text(0.01, 0.005, "predicted: forward-Euler integration of measured 3 forces + k-lock bridge (λ/σ=%.3f); observed: independent Paper#1 Weibull fit on checkpoints. Residual = discretization + bridge." % los, fontsize=6.5, color="gray")
fig.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(a.out, dpi=140); plt.close(fig)
print(f"[pred] saved {a.out} | logRMSE={logrmse:.4f} max|err|={np.max(np.abs(resid)):.1f}% λ/σ={los:.3f} N={N:.3e}", flush=True)
