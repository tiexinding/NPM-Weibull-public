#!/usr/bin/env python3
"""T1-T3: 行特异注入的结构起源判别 (260822 评审七轮方案; exploratory, 预期跑前登记).

行级量 X_i = 全程累计相对 log 行尺度漂移 = Σ_t (Δlog r_i − median_row Δlog r) (去共同模式).
RoPE 配对已按 transformers 5.3.0 确认: rotate_half half-split, 行 j 的频率 index m = j mod 32,
ω_m = 10000^(−m/32); pair = (m, m+32), X_pair = (X_m + X_{m+32})/2。head: 行块 h = idx//64
(q/k/v); o 的 head 结构在列 → o 用列 RMS 的累计相对漂移做列侧分析。

T1 RoPE 频率: X_pair ~ μ + f(freq-bin×8, 允许非单调) + layer + head (OLS dummy);
   报 frequency partial R² (在 layer+head 之上的增量解释)。阴性对照 = v 伪频率同构造。
   预期: 若 RoPE 是 q/k 注入的结构标签 → q/k 的 freq partial R² 明显 > v 伪频率。
T2 嵌套方差分解 (X_i): layer / head(层内) / freq (q,k) / residual 占比; o 列侧同分解。
   预期: v between-head 主导 → head 专业化; q/k freq 占比 > v 伪频率 → RoPE。
T3 跨 run 一致性: ①32 维频率 profile (X 跨 layer×head×pair 中位) 跨 9 EA runs
   (3 族 × 3 seeds, 同语料同序) Pearson; ②head 级 8 维 profile (per layer) 跨 run
   Hungarian matching 后相关 vs 同 index 直接相关。
   预期: 频率 profile 跨族一致 = 架构/数据固定宏观身份 (与跨族坐标近正交并置);
   head 直接相关低而匹配后高 = head 身份为 run 特异涌现。
用法: python3 t123_structure_v1.py --base . --out 06_hrow/t123_structure_v1.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

CKPT_STEPS = [0, 800, 3200, 10000, 20000, 25000, 30000]
H, DH, NF = 8, 64, 32


def cum_rel_drift(ckpt_dir, axis_kinds):
    """{kind: {(layer, idx): X}} — idx = 行 (row kinds) 或列 (o_col)."""
    logr = []
    for s in CKPT_STEPS:
        sd = torch.load(os.path.join(ckpt_dir, f"step{s}.pt"),
                        map_location="cpu", weights_only=True)
        cur = {}
        for nm, tw in sd.items():
            proj = nm.rsplit(".", 2)[-2]
            layer = int(nm.split("layers.")[1].split(".")[0])
            w = tw.to(torch.float64)
            if proj in ("q_proj", "k_proj", "v_proj") and proj[0] in axis_kinds:
                cur[(proj[0], layer)] = np.log(np.sqrt((w ** 2).mean(dim=1).numpy()))
            if proj == "o_proj" and "ocol" in axis_kinds:
                cur[("ocol", layer)] = np.log(np.sqrt((w ** 2).mean(dim=0).numpy()))
        logr.append(cur)
    X = {}
    for key in logr[0]:
        d = np.zeros_like(logr[0][key])
        for a, b in zip(logr[:-1], logr[1:]):
            dd = b[key] - a[key]
            d += dd - np.median(dd)
        X[key] = d
    return X


def pair_avg(x):
    """行级 64 维/head → pair 级 32 维/head: X_pair[h,m] = (X[h,m]+X[h,m+32])/2 (P0-2 修正,
    真正构造旋转对二维不变量; 此前只是按频率标签分组单坐标)."""
    out = np.empty(H * NF)
    for h in range(H):
        blk = x[h * DH:(h + 1) * DH]
        out[h * NF:(h + 1) * NF] = (blk[:NF] + blk[NF:]) / 2.0
    return out


def anova_partial(X_by_key, use_freq, pseudo=False):
    """OLS dummy 分解: layer + head + (freq bins) → 占比与 freq partial R². pair 级 (P0-2)."""
    rows, ys = [], []
    for (kd, layer), x in X_by_key.items():
        xp = pair_avg(x)
        for idx in range(len(xp)):
            h, m = idx // NF, idx % NF
            rows.append((layer, h, m // 4))   # 8 freq bins
            ys.append(xp[idx])
    rows = np.array(rows); y = np.array(ys); y = y - y.mean()
    def dummies(cols):
        mats = []
        for ci in cols:
            vals = np.unique(rows[:, ci])
            mats.append((rows[:, ci][:, None] == vals[None, :]).astype(float))
        return np.hstack([np.ones((len(y), 1))] + mats)
    def ssres(Xd):
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        r = y - Xd @ beta
        return float((r ** 2).sum())
    ss_tot = float((y ** 2).sum())
    ss_lh = ssres(dummies([0, 1]))
    ss_lhf = ssres(dummies([0, 1, 2])) if use_freq else None
    out = {"R2_layer_head": 1 - ss_lh / ss_tot}
    if use_freq:
        out["freq_partial_R2"] = (ss_lh - ss_lhf) / ss_lh
        out["R2_full"] = 1 - ss_lhf / ss_tot
        out["pseudo_freq"] = pseudo
    # 嵌套方差占比 (T2): 组均值方差
    lab_l, lab_h = rows[:, 0], rows[:, 0] * 10 + rows[:, 1]
    vb_l = np.var([y[lab_l == l].mean() for l in np.unique(lab_l)])
    hm = [y[lab_h == g].mean() for g in np.unique(lab_h)]
    vb_h = np.var(hm) - vb_l if len(hm) > 1 else 0.0
    out["var_share"] = {"layer": float(vb_l / y.var()),
                        "head_within_layer": float(max(vb_h, 0) / y.var()),
                        "residual_within_head": float(1 - (vb_l + max(vb_h, 0)) / y.var())}
    return out


def freq_profile(X_by_key):
    allv = {m: [] for m in range(NF)}
    for key, x in X_by_key.items():
        xp = pair_avg(x)
        for idx in range(len(xp)):
            allv[idx % NF].append(xp[idx])
    return np.array([np.median(allv[m]) for m in range(NF)])


def head_profiles(X_by_key):
    """{layer: 8 维 head 均值}."""
    out = {}
    for (kd, layer), x in X_by_key.items():
        out[layer] = np.array([x[h * DH:(h + 1) * DH].mean() for h in range(H)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".")
    ap.add_argument("--out", default="06_hrow/t123_structure_v1.json")
    a = ap.parse_args()
    base = a.base
    res = {"t1_t2": {}, "t3": {}}

    # ---- T1/T2: 8 runs (2 fam × 4 dose) ----
    for fam in ("gaussian", "laplace"):
        for dose, d in ((0.0, f"05_opt_runs/opt_a1_{fam}_s1/ckpt"),
                        (0.03, f"05_opt_runs/opt_a3w003_{fam}_s1/ckpt"),
                        (0.1, f"04_ea_runs/ckpt_key/ea_{fam}_s1/ckpt"),
                        (0.3, f"05_opt_runs/opt_a3w030_{fam}_s1/ckpt")):
            X = cum_rel_drift(os.path.join(base, d), ("q", "k", "v", "ocol"))
            ent = {}
            for kd in ("q", "k", "v", "ocol"):
                sub = {kk: vv for kk, vv in X.items() if kk[0] == kd}
                ent[kd] = anova_partial(sub, use_freq=(kd in ("q", "k", "v")),
                                        pseudo=(kd == "v"))
            res["t1_t2"][f"{fam}/{dose}"] = ent
            print(f"  {fam} λ_wd={dose}: freqR2 q={ent['q']['freq_partial_R2']:.3f} "
                  f"k={ent['k']['freq_partial_R2']:.3f} v(伪)={ent['v']['freq_partial_R2']:.3f} | "
                  f"headshare q={ent['q']['var_share']['head_within_layer']:.2f} "
                  f"v={ent['v']['var_share']['head_within_layer']:.2f} "
                  f"ocol={ent['ocol']['var_share']['head_within_layer']:.2f}")

    # ---- T3: 9 EA runs 跨 seed/族 ----
    runs = [f"ea_{f}_s{s}" for f in ("gaussian", "uniform", "laplace") for s in (1, 2, 3)]
    profs, hprofs = {}, {}
    for r in runs:
        X = cum_rel_drift(os.path.join(base, "04_ea_runs/ckpt_key", r, "ckpt"), ("q", "k", "v"))
        profs[r] = {kd: freq_profile({kk: vv for kk, vv in X.items() if kk[0] == kd})
                    for kd in ("q", "k", "v")}
        hprofs[r] = {kd: head_profiles({kk: vv for kk, vv in X.items() if kk[0] == kd})
                     for kd in ("q", "k", "v")}
    t3 = {}
    for kd in ("q", "k", "v"):
        cors = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                cors.append(float(np.corrcoef(profs[runs[i]][kd], profs[runs[j]][kd])[0, 1]))
        # head: 同 index 直接 vs Hungarian 匹配后 (逐层, 用 |corr| 最大化)
        direct, matched = [], []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                for layer in range(6):
                    a1 = hprofs[runs[i]][kd][layer]; a2 = hprofs[runs[j]][kd][layer]
                    direct.append(float(np.corrcoef(a1, a2)[0, 1]))
                    # 匹配在 head 均值向量间无子结构可配 (8 标量) → 排序匹配近似
                    matched.append(float(np.corrcoef(np.sort(a1), np.sort(a2))[0, 1]))
        t3[kd] = {"freq_profile_cross_run_corr_median": float(np.median(cors)),
                  "freq_profile_cross_run_corr_range": [float(np.min(cors)), float(np.max(cors))],
                  "head_direct_corr_median": float(np.median(direct)),
                  "head_sorted_corr_median": float(np.median(matched))}
        print(f"  T3 {kd}: freq-profile 跨 run corr 中位 {t3[kd]['freq_profile_cross_run_corr_median']:.2f} "
              f"[{t3[kd]['freq_profile_cross_run_corr_range'][0]:.2f},{t3[kd]['freq_profile_cross_run_corr_range'][1]:.2f}] | "
              f"head 同 index {t3[kd]['head_direct_corr_median']:.2f} vs 排序后 {t3[kd]['head_sorted_corr_median']:.2f}")
    res["t3"] = t3
    res["freq_profiles"] = {r: {kd: profs[r][kd].tolist() for kd in profs[r]} for r in runs}
    json.dump({"schema": "t123_structure_v1.1_pairavg", "rope": "transformers 5.3.0 half-split (m, m+32) 确认",
               **res}, open(os.path.join(base, a.out), "w"), indent=1, ensure_ascii=False)
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
