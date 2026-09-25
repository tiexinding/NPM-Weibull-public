#!/usr/bin/env python3
"""KMIX v1.2 (260828): 老丁 v1.1 复核三 P0 修正 + KMIX 三项定量分解.

 P0-1 稳定随机化: hash() (进程随机化, 不可复现) → sha256 前缀整数; 全部随机场可逐位复现;
 P0-2 设计拆分与改名:
   marginal-shape experiment (随机重指派, 研究 h 边际高阶矩): lognormal / two-point assignment /
     sparse / arcsine-marginal (原 "smooth" 被置换后的实名) — 原 "sawtooth" 与 two-point 同分布, 删;
   profile-order experiment (保留顺序, 研究与 Z_i 行序的结构配对): smooth-ordered / sawtooth-ordered
     vs 同边际随机指派的差 = 顺序配对效应;
 P0-3 三项分解 (每个真实矩阵, D1–D4×3s×6L×q/k @step5000, 10 perms):
   Δ_actual = k_raw⁻² − k_norm⁻²
   Δ_perm   = E_π[k_π⁻² − k_norm⁻²]   (真实 r_i 随机重指派)
   Δ_pair   = Δ_actual − Δ_perm        (尺度—行内形状配对)
   Δ_high   = Δ_perm − (6/π²)(H^W)²    (h 高阶结构+协议修正)
   ⇒ Δ_actual = (6/π²)H² + Δ_high + Δ_pair  (恒等分解)
输出 06_hrow/kmix_bridge_v1_2.json。零 GPU。
用法: python3 scripts/kmix_bridge_v1_2.py"""
import glob, hashlib, json, os, re
import numpy as np
import torch

WT = "04_tclock_runs/wt_ckpts"
HB, HR = 1024, (-12.0, 2.0)
A0 = 6 / np.pi ** 2

def sint(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

def fit_hist_r2(hist):
    hist = hist.astype(np.float64); total = hist.sum()
    edges = np.linspace(HR[0], HR[1], HB + 1)
    cdf_u = np.cumsum(hist) / total; cdf_l = (np.cumsum(hist) - hist) / total
    F = (cdf_u + cdf_l) / 2.0
    bc = (edges[:-1] + edges[1:]) / 2.0 * np.log(10.0)
    m = (F >= 0.1) & (F <= 0.9) & (hist > 0)
    y = np.log(-np.log(1 - np.clip(F[m], 1e-12, 1 - 1e-12)))
    X = np.vstack([bc[m], np.ones(m.sum())]).T
    w = hist[m]; WX = X * w[:, None]
    beta, *_ = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)
    res = y - X @ beta; ybar = (w * y).sum() / w.sum()
    return float(beta[0]), float(1 - (w * res ** 2).sum() / (w * (y - ybar) ** 2).sum())

def khist(x):
    h = np.histogram(np.log10(np.maximum(np.abs(x.ravel()), 1e-13)), bins=HB, range=HR)[0]
    return fit_hist_r2(h)

def rownorm(w):
    r = np.sqrt((w ** 2).mean(1, keepdims=True))
    z = w / np.maximum(r, 1e-30)
    return z * np.sqrt((w ** 2).mean()) / np.sqrt((z ** 2).mean())

def inject(z, h, gref):
    wp = np.exp(h)[:, None] * z
    return wp * np.sqrt(gref / (wp ** 2).mean())

NSEED = 10

# ============ Part D: 三项分解 (全臂真实矩阵) ============
D_rows = []
for run in sorted(d for d in glob.glob(f"{WT}/tclock_D*_s*") if os.path.isdir(d)):
    arm, sd = re.search(r"tclock_(D\d)_s(\d)", run).groups()
    st = torch.load(f"{run}/step5000.pt", map_location="cpu", weights_only=True)
    for nm, w in st.items():
        mm = re.search(r"layers\.(\d+)\.self_attn\.(q_proj|k_proj)\.weight$", nm)
        if not mm:
            continue
        w = w.double().numpy(); L, proj = int(mm.group(1)), mm.group(2)
        z = rownorm(w); r = np.sqrt((w ** 2).mean(1)); gref = (w ** 2).mean()
        kr, _ = khist(w); kn, _ = khist(z)
        hw = float(np.log(r).std())
        kps = []
        for s in range(NSEED):
            rng = np.random.default_rng(sint("permD", arm, sd, L, proj, s))
            h = np.log(rng.permutation(r))
            kps.append(khist(inject(z, h, gref))[0])
        d_act = kr ** -2 - kn ** -2
        d_perm = float(np.mean([kp ** -2 - kn ** -2 for kp in kps]))
        hln = np.log(r) - np.log(r).mean()
        D_rows.append({"arm": arm, "seed": int(sd), "L": L, "proj": proj,
                       "k_raw": kr, "k_norm": kn, "hw": hw,
                       "skew_h": float((hln ** 3).mean() / hln.std() ** 3),
                       "d_actual": d_act, "d_perm": d_perm,
                       "d_perm_sd": float(np.std([kp ** -2 - kn ** -2 for kp in kps])),
                       "d_pair": d_act - d_perm, "d_high": d_perm - A0 * hw ** 2})
    print(f"D done {arm}_s{sd}", flush=True)

def agg(rows, f):
    return float(np.median([f(r) for r in rows]))
partD = {"per_matrix": D_rows, "summary": {}}
for arm in ("D1", "D2", "D3", "D4"):
    sub = [r for r in D_rows if r["arm"] == arm]
    n = len(sub)
    partD["summary"][arm] = {
        "n": n,
        "d_actual_med": agg(sub, lambda r: r["d_actual"]),
        "base_term_med": agg(sub, lambda r: A0 * r["hw"] ** 2),
        "d_high_med": agg(sub, lambda r: r["d_high"]),
        "d_pair_med": agg(sub, lambda r: r["d_pair"]),
        "pair_pos_frac": float(np.mean([r["d_pair"] > 0 for r in sub])),
        "pair_over_own_sd_med": agg(sub, lambda r: abs(r["d_pair"]) / max(r["d_perm_sd"], 1e-12)),
        "share_med": {"base": agg(sub, lambda r: A0 * r["hw"] ** 2 / r["d_actual"]),
                      "high": agg(sub, lambda r: r["d_high"] / r["d_actual"]),
                      "pair": agg(sub, lambda r: r["d_pair"] / r["d_actual"])}}
allD = [r for r in D_rows]
partD["summary"]["ALL"] = {"n": len(allD),
                           "pair_pos_frac": float(np.mean([r["d_pair"] > 0 for r in allD])),
                           "share_med": partD["summary"]["D1"]["share_med"]}

# ============ Part C': marginal-shape + profile-order (稳定种子) ============
st = torch.load(f"{WT}/tclock_D1_s1/step5000.pt", map_location="cpu", weights_only=True)
mats = {f"L{i}|{p}": st[f"model.layers.{i}.self_attn.{p}.weight"].double().numpy()
        for i in range(6) for p in ("q_proj", "k_proj")}
MARG = ("lognormal", "twopoint_assign", "sparse", "arcsine_marginal")
def marg_field(design, n, H, rng):
    if design == "lognormal": h = rng.standard_normal(n)
    elif design == "twopoint_assign": h = rng.permutation(np.repeat([-1.0, 1.0], n // 2))
    elif design == "sparse":
        h = np.zeros(n); h[rng.choice(n, n // 10, replace=False)] = -1.0
    elif design == "arcsine_marginal": h = rng.permutation(np.sin(2 * np.pi * np.arange(n) / n * 3))
    h = h - h.mean()
    return h / h.std() * H
cellsC = {}
for key, w in mats.items():
    z = rownorm(w); gref = (w ** 2).mean()
    kn, _ = khist(z)
    cellsC[key] = {"k_norm": kn, "marginal": {}, "profile_order": {}}
    for des in MARG:
        for H in (0.10, 0.20, 0.35):
            ks = [khist(inject(z, marg_field(des, z.shape[0], H,
                    np.random.default_rng(sint("marg", key, des, H, s))), gref))[0]
                  for s in range(NSEED)]
            cellsC[key]["marginal"][f"{des}|H{H}"] = {"k_med": float(np.median(ks)),
                                                      "k_mc_sd": float(np.std(ks))}
    # profile-order: 保序 smooth/sawtooth vs 同边际随机指派
    n = z.shape[0]
    for des, hord in (("smooth_ordered", np.sin(2 * np.pi * np.arange(n) / n * 3)),
                      ("sawtooth_ordered", np.where(np.arange(n) % 2 == 0, 1.0, -1.0))):
        for H in (0.10, 0.20, 0.35):
            h0 = (hord - hord.mean()) / hord.std() * H
            k_ord = khist(inject(z, h0, gref))[0]
            ks_sh = [khist(inject(z, np.random.default_rng(sint("ord", key, des, H, s)).permutation(h0),
                                  gref))[0] for s in range(NSEED)]
            cellsC[key]["profile_order"][f"{des}|H{H}"] = {
                "k_ordered": float(k_ord), "k_shuffled_med": float(np.median(ks_sh)),
                "k_shuffled_sd": float(np.std(ks_sh)),
                "order_effect": float(k_ord - np.median(ks_sh))}
    print(f"C' done {key}", flush=True)

# 汇总
ordeff = {f"{des}|H{H}": {"med": float(np.median([c["profile_order"][f"{des}|H{H}"]["order_effect"]
                                                  for c in cellsC.values()])),
                          "med_abs_over_sd": float(np.median(
                              [abs(c["profile_order"][f"{des}|H{H}"]["order_effect"])
                               / max(c["profile_order"][f"{des}|H{H}"]["k_shuffled_sd"], 1e-12)
                               for c in cellsC.values()]))}
          for des in ("smooth_ordered", "sawtooth_ordered") for H in (0.10, 0.20, 0.35)}

res = {"partD_decomposition": {"summary": partD["summary"], "per_matrix": partD["per_matrix"]},
       "partC2": {"cells": cellsC, "order_effect_summary": ordeff},
       "provenance": {"script": os.path.basename(__file__),
                      "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
                      "rng": "sha256-prefix stable seeds (P0-1 fix); NSEED=10",
                      "identity": "d_actual = A0*H^2 + d_high + d_pair (A0=6/pi^2)",
                      "audit_basis": "老丁 v1.1 复核三 P0 (260828)"}}
out = "06_hrow/kmix_bridge_v1_2.json"
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"partD": partD["summary"], "order_effects": ordeff}, indent=1))
print(f"-> {out}")
