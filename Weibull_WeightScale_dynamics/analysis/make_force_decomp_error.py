#!/usr/bin/env python
"""预测 λ 的三项力贡献分解 + 误差来源拆解.
(a) 累积力贡献: σ²(t) = σ²(0) + ∫align + ∫inj + ∫decay → 看 align 顶升/decay 刹车/inj 小.
(b) 误差拆解: 总误差(pred λ vs obs Weibull-λ) = 积分分量(pred σ² vs obs σ², 纯前向欧拉) + 桥分量(σ·ratio vs Weibull-λ, 纯 k锁桥).
用法(云端 figwork): python make_force_decomp_error.py
"""
import json, argparse, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--true", default="derived_data/self_train/v1b_spline_true_llama.jsonl")
ap.add_argument("--lam", default="derived_data/self_train/lambda_trajectory_llama.json")
ap.add_argument("--out", default="F_force_decomp_error.png")
ap.add_argument("--label", default="Llama-70m")
ap.add_argument("--lwd", type=float, default=0.01)
a = ap.parse_args()
tr = [json.loads(l) for l in open(a.true) if l.strip()]
lam = json.load(open(a.lam))
ratio = np.mean([r["lambda_over_sigma"] for r in lam])
steps = [r["step"] for r in tr]
r0 = tr[5]; N = r0["dec_full"] / (2 * r0["eta"] * a.lwd * r0["rms"] ** 2)

# 累积力贡献 (σ²=ΣW²/N 单位)
sig2_0 = tr[0]["rms"] ** 2
cumA = [0.0]; cumJ = [0.0]; cumD = [0.0]
for i in range(1, len(tr)):
    dt = steps[i] - steps[i - 1]; pr = tr[i - 1]
    cumA.append(cumA[-1] + (-pr["true_align_full"]) * dt / N)
    cumJ.append(cumJ[-1] + (pr["inj_full"]) * dt / N)
    cumD.append(cumD[-1] + (-pr["dec_full"]) * dt / N)
cumA = np.array(cumA); cumJ = np.array(cumJ); cumD = np.array(cumD)
sig2_pred = sig2_0 + cumA + cumJ + cumD
sig_pred = np.sqrt(np.clip(sig2_pred, 1e-12, None))

# 误差拆解 (在 ckpt grid: lambda_trajectory 有 lambda(Weibull) + sigma(rms))
lk = np.array([r["step"] for r in lam]); lL = np.array([r["lambda"] for r in lam]); lS = np.array([r["sigma"] for r in lam])
sig_pred_on_lk = np.interp(lk, steps, sig_pred)
lam_pred_on_lk = sig_pred_on_lk * ratio
err_total = 100 * (lam_pred_on_lk - lL) / lL                 # 总: pred λ vs Weibull λ
err_integ = 100 * (sig_pred_on_lk - lS) / lS                 # 积分: pred σ vs 实测 σ (纯欧拉)
err_bridge = 100 * (lS * ratio - lL) / lL                    # 桥: σ·ratio vs Weibull λ (纯桥)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
# (a) 累积力贡献
ax[0].plot(steps, cumA, "-", c="C0", lw=2, label="∫ alignment (drives ↑)")
ax[0].plot(steps, cumD, "-", c="C2", lw=2, label="∫ decay (brakes ↓)")
ax[0].plot(steps, cumJ, "-", c="C1", lw=1.5, label="∫ injection (small)")
ax[0].plot(steps, sig2_pred - sig2_0, "--", c="k", lw=2, label="net Δσ² (= predicted)")
ax[0].axhline(0, c="gray", lw=0.6)
ax[0].set_xlabel("training step"); ax[0].set_ylabel("cumulative contribution to σ²")
ax[0].set_title("(a) Force decomposition of predicted growth")
ax[0].legend(fontsize=8)
# (b) 误差拆解
ax[1].axhline(0, c="gray", lw=0.6)
ax[1].plot(lk, err_total, "-", c="C3", lw=2, label="total error (pred λ vs Weibull λ)")
ax[1].plot(lk, err_integ, "-", c="C0", lw=1.5, label="integration/reconstruction component (Δt-independent)")
ax[1].plot(lk, err_bridge, "-", c="C4", lw=1.5, label="k-lock bridge (σ·ratio vs λ)")
ax[1].set_xlabel("training step"); ax[1].set_ylabel("relative error (%)")
ax[1].set_title("(b) Error source decomposition")
ax[1].legend(fontsize=8)
# 标注主导
sat = lk >= 10000
ax[1].text(0.04, 0.06, f"@saturation: total {np.mean(err_total[sat]):.1f}% = integ {np.mean(err_integ[sat]):.1f}% + bridge {np.mean(err_bridge[sat]):.1f}%",
           transform=ax[1].transAxes, fontsize=8)
fig.suptitle("Predicted λ: three-force contributions + error sources (self-trained "+a.label+")", fontsize=12)
fig.text(0.01, 0.005, "predicted via forward-Euler integration of measured 3 forces + k-lock bridge; observed = Weibull fit on ckpts", fontsize=6.5, color="gray")
fig.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(a.out, dpi=140); plt.close(fig)
print(f"[decomp] saved {a.out}", flush=True)
print(f"[decomp] @saturation total={np.mean(err_total[sat]):.2f}% integ={np.mean(err_integ[sat]):.2f}% bridge={np.mean(err_bridge[sat]):.2f}%", flush=True)
print(f"[decomp] 累积@末: align=+{cumA[-1]:.5f} decay={cumD[-1]:.5f} inj=+{cumJ[-1]:.5f} (σ²单位; align/|decay|={cumA[-1]/abs(cumD[-1]):.2f})", flush=True)
