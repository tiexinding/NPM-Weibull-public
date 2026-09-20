#!/usr/bin/env python3
"""KMIX v1.1: 老丁结果后审计九条修正版 (260827; v1 判据/命名修正, 数据与理论不变).

修正清单 (相对 kmix_bridge_v1.py):
 P0-1 过原点 R0² = 1 − Σ(y−αx)²/Σy² 为主判据 (corr² 保留为附加读数);
 P0-2 leave-one-run-out + leave-one-arm-out 的真预测 MAE (原样本内 MAE 改称 goodness-of-fit);
 P0-3 fit adequacy: raw/norm/injected 全部保存加权概率图回归 R²_fit;
 P0-4 "perm_control" 改名 scale-assignment test (检验尺度—行内形状配对, 非行排列);
      真整行置换 = 多重集恒等 ⇒ 仅作代码自检 (assert k 逐位不变);
 P0-5 噪声门改 cross-layer reference spread + 随机场 10 seeds 的 within-cell MC spread;
 C 部全六层 × q/k 两投影; J-C3 按 H 分层判定 (M0=6/π² 与 M1=α_TCLOCK 双模型);
 M2: Δ_k^mix 残差 ~ skew(h)/kurt(h) 回归 (descriptive);
 Part B 改称 raw trajectory consistency diagnostic (非正式 α 估计)。
输出 06_hrow/kmix_bridge_v1_1.json (逐矩阵审计表全存)。
用法: python3 scripts/kmix_bridge_v1_1.py"""
import glob, hashlib, json, os, re, sys
import numpy as np
import torch

WT = "04_tclock_runs/wt_ckpts"
HB, HR = 1024, (-12.0, 2.0)
KINDS_A = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")

def fit_hist_r2(hist):
    """逐字复刻 tclock_secondary_v1.fit_hist + 加权回归 R²_fit."""
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
    r2f = 1 - (w * res ** 2).sum() / (w * (y - ybar) ** 2).sum()
    return float(beta[0]), float(r2f)

def khist(x):
    h = np.histogram(np.log10(np.maximum(np.abs(x.ravel()), 1e-13)), bins=HB, range=HR)[0]
    return fit_hist_r2(h)

def rownorm(w):
    r = np.sqrt((w ** 2).mean(1, keepdims=True))
    z = w / np.maximum(r, 1e-30)
    return z * np.sqrt((w ** 2).mean()) / np.sqrt((z ** 2).mean())

# ---------- Part A ----------
A_rows = []
for run in sorted(d for d in glob.glob(f"{WT}/tclock_D*_s*") if os.path.isdir(d)):
    arm, sd = re.search(r"tclock_(D\d)_s(\d)", run).groups()
    for t in (800, 1600, 3200, 5000):
        st = torch.load(f"{run}/step{t}.pt", map_location="cpu", weights_only=True)
        for nm, w in st.items():
            mm = re.search(r"\.(\w+_proj)\.weight$", nm)
            if not mm or mm.group(1) not in KINDS_A:
                continue
            w = w.double().numpy()
            kr, r2r = khist(w); kn, r2n = khist(rownorm(w))
            A_rows.append({"arm": arm, "seed": int(sd), "step": t, "kind": mm.group(1),
                           "mat": nm, "k_raw": kr, "k_norm": kn,
                           "r2fit_raw": r2r, "r2fit_norm": r2n,
                           "hw": float(np.log(np.sqrt((w ** 2).mean(1))).std())})
    print(f"A done {arm}_s{sd}", flush=True)

def origin_fit(rows):
    x = np.array([r["hw"] ** 2 for r in rows])
    y = np.array([r["k_raw"] ** -2 - r["k_norm"] ** -2 for r in rows])
    a = float((x @ y) / (x @ x))
    r0 = float(1 - ((y - a * x) ** 2).sum() / (y ** 2).sum())          # P0-1 主判据
    corr2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    mae_in = float(np.mean(np.abs((np.array([r["k_norm"] for r in rows]) ** -2 + a * x) ** -0.5
                                  - np.array([r["k_raw"] for r in rows]))))
    return a, r0, corr2, mae_in

def loo_mae(rows, key):
    groups = sorted(set(key(r) for r in rows)); errs = []
    for g in groups:
        tr = [r for r in rows if key(r) != g]; te = [r for r in rows if key(r) == g]
        a = origin_fit(tr)[0]
        errs.append(float(np.mean([abs((r["k_norm"] ** -2 + a * r["hw"] ** 2) ** -0.5 - r["k_raw"])
                                   for r in te])))
    return {"per_group": dict(zip(map(str, groups), errs)), "median": float(np.median(errs)),
            "max": float(np.max(errs))}

partA = {}
for fam, kinds in (("q_proj", ("q_proj",)), ("k_proj", ("k_proj",)), ("qk_pooled", ("q_proj", "k_proj")),
                   ("v_proj", ("v_proj",)), ("gate_proj", ("gate_proj",)), ("up_proj", ("up_proj",))):
    rows = [r for r in A_rows if r["kind"] in kinds]
    a, r0, c2, mi = origin_fit(rows)
    run_alpha = [origin_fit([r for r in rows if (r["arm"], r["seed"]) == rr])[0]
                 for rr in sorted(set((r["arm"], r["seed"]) for r in rows))]
    partA[fam] = {"alpha": a, "R0_sq": r0, "corr_sq_aux": c2, "n": len(rows),
                  "runs_same_sign": int(sum(v > 0 for v in run_alpha)),
                  "run_alpha_range": [float(min(run_alpha)), float(max(run_alpha))],
                  "gof_mae_insample": mi,
                  "pred_mae_LORO": loo_mae(rows, lambda r: (r["arm"], r["seed"])),
                  "pred_mae_LOAO": loo_mae(rows, lambda r: r["arm"]),
                  "fit_adequacy": {"r2fit_raw_min": float(min(r["r2fit_raw"] for r in rows)),
                                   "r2fit_raw_med": float(np.median([r["r2fit_raw"] for r in rows])),
                                   "r2fit_norm_min": float(min(r["r2fit_norm"] for r in rows)),
                                   "r2fit_norm_med": float(np.median([r["r2fit_norm"] for r in rows]))}}
qa = partA["qk_pooled"]
partA["J_A_v11"] = ("supported" if qa["R0_sq"] >= 0.9 and 0.3 <= qa["alpha"] <= 1.2
                    and qa["runs_same_sign"] == 12 else
                    "not-supported" if qa["R0_sq"] < 0.7 else "indeterminate")

# ---------- Part C ----------
rng_master = np.random.default_rng(20260827)
st = torch.load(f"{WT}/tclock_D1_s1/step5000.pt", map_location="cpu", weights_only=True)
mats = {f"L{i}|{p}": st[f"model.layers.{i}.self_attn.{p}.weight"].double().numpy()
        for i in range(6) for p in ("q_proj", "k_proj")}
NSEED = 10
def field(design, n, H, r_real, rng):
    if design == "lognormal": h = rng.standard_normal(n)
    elif design == "twopoint": h = rng.permutation(np.repeat([-1.0, 1.0], n // 2))
    elif design == "sparse":
        h = np.zeros(n); h[rng.choice(n, n // 10, replace=False)] = -1.0
    elif design == "smooth": h = rng.permutation(np.sin(2 * np.pi * np.arange(n) / n * 3))
    elif design == "sawtooth": h = rng.permutation(np.where(np.arange(n) % 2 == 0, 1.0, -1.0))
    elif design == "scale_assign": return np.log(rng.permutation(r_real))
    h = h - h.mean()
    return h / h.std() * H
DESIGNS = ("lognormal", "twopoint", "sparse", "smooth", "sawtooth")
cells = {}; selfcheck_ok = True
for key, w in mats.items():
    z = rownorm(w); r_real = np.sqrt((w ** 2).mean(1))
    kn, r2n = khist(z); kr, r2r = khist(w)
    cells[key] = {"k_norm": kn, "k_raw": kr, "r2fit_norm": r2n, "r2fit_raw": r2r,
                  "inj": {}, "scale_assign": []}
    # 代码自检: 真整行置换 = 多重集恒等 ⇒ k 逐位不变
    kperm, _ = khist(w[rng_master.permutation(w.shape[0])])
    selfcheck_ok &= (kperm == kr)
    for sd in range(NSEED):
        rng = np.random.default_rng(1000 * sd + hash(key) % 997)
        h = field("scale_assign", z.shape[0], None, r_real, rng)
        wp = np.exp(h)[:, None] * z; wp *= np.sqrt((w ** 2).mean() / (wp ** 2).mean())
        cells[key]["scale_assign"].append(khist(wp)[0])
    for des in DESIGNS:
        for H in (0.10, 0.20, 0.35):
            ks, r2s, sk, ku = [], [], [], []
            for sd in range(NSEED):
                rng = np.random.default_rng(7000 + 1000 * sd + hash((key, des)) % 997)
                h = field(des, z.shape[0], H, r_real, rng)
                wp = np.exp(h)[:, None] * z; wp *= np.sqrt((w ** 2).mean() / (wp ** 2).mean())
                kv, r2v = khist(wp); ks.append(kv); r2s.append(r2v)
                sk.append(float(((h - h.mean()) ** 3).mean() / h.std() ** 3))
                ku.append(float(((h - h.mean()) ** 4).mean() / h.std() ** 4 - 3))
            cells[key]["inj"][f"{des}|H{H}"] = {
                "k_med": float(np.median(ks)), "k_mc_sd": float(np.std(ks)),
                "r2fit_med": float(np.median(r2s)), "r2fit_min": float(np.min(r2s)),
                "skew_med": float(np.median(sk)), "kurt_med": float(np.median(ku))}
    print(f"C done {key}", flush=True)

# 判据 v1.1
mc_spread = float(np.median([c["inj"][f"{d}|H0.2"]["k_mc_sd"] for c in cells.values() for d in DESIGNS]))
xlayer = {f"H{H}": float(np.median([np.ptp([cells[f"L{i}|{p}"]["inj"][f"{d}|H{H}"]["k_med"] for i in range(6)])
                                    for p in ("q_proj", "k_proj") for d in DESIGNS]))
          for H in (0.10, 0.20, 0.35)}
design_spread = {f"H{H}": float(np.median([np.ptp([c["inj"][f"{d}|H{H}"]["k_med"] for d in DESIGNS])
                                           for c in cells.values()])) for H in (0.10, 0.20, 0.35)}
sa_dk = [abs(np.median(c["scale_assign"]) - c["k_raw"]) for c in cells.values()]
aM1 = partA["qk_pooled"]["alpha"]
jc3 = {}
for H in (0.10, 0.20, 0.35):
    errs = {"M0": [], "M1": []}
    for c in cells.values():
        for d in DESIGNS:
            kk = c["inj"][f"{d}|H{H}"]["k_med"]
            for tag, a in (("M0", 6 / np.pi ** 2), ("M1", aM1)):
                errs[tag].append(abs((c["k_norm"] ** -2 + a * H ** 2) ** -0.5 - kk))
    jc3[f"H{H}"] = {t: {"med": float(np.median(v)), "max": float(np.max(v))} for t, v in errs.items()}
# M2 (descriptive): 残差 ~ skew/kurt
X, Y = [], []
for c in cells.values():
    for d in DESIGNS:
        for H in (0.10, 0.20, 0.35):
            e = c["inj"][f"{d}|H{H}"]
            X.append([H ** 2, e["skew_med"] * H ** 2, e["kurt_med"] * H ** 2])
            Y.append(e["k_med"] ** -2 - c["k_norm"] ** -2)
X = np.array(X); Y = np.array(Y)
beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
r2_m2 = float(1 - ((Y - X @ beta) ** 2).sum() / ((Y - (X[:, :1] @ np.linalg.lstsq(X[:, :1], Y, rcond=None)[0])) ** 2).sum())
partC = {"cells": cells, "verdicts": {
    "selfcheck_rowperm_bitwise": bool(selfcheck_ok),
    "mc_spread_within_cell": mc_spread,
    "cross_layer_reference_spread": xlayer,
    "design_spread_by_H": design_spread,
    "J_C1_scale_assignment": {"dk_med": float(np.median(sa_dk)), "dk_max": float(np.max(sa_dk)),
                              "vs_mc_spread": "weak-pairing" if np.median(sa_dk) <= 3 * mc_spread else "pairing-matters"},
    "J_C3_stratified": jc3,
    "M2_skew_kurt": {"alpha": float(beta[0]), "beta_skew": float(beta[1]),
                     "gamma_kurt": float(beta[2]), "resid_R2_gain_vs_H2only": r2_m2}}}

res = {"partA": partA, "partA_audit_rows": A_rows,
       "partB_note": "v1 Part B 改称 raw trajectory consistency diagnostic (混入 init 形状/暂态/耦合, 非正式 α 估计); 数值见 kmix_bridge_v1.json",
       "partC": partC,
       "provenance": {"script": os.path.basename(__file__),
                      "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
                      "audit_basis": "老丁结果后审计九条 (260827); v1 = kmix_bridge_v1.json"}}
out = "06_hrow/kmix_bridge_v1_1.json"
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"partA_qk": {k: v for k, v in partA["qk_pooled"].items() if k != "pred_mae_LORO"},
                  "J_A_v11": partA["J_A_v11"], "verdicts": partC["verdicts"]}, indent=1, default=str))
print(f"-> {out}")
