#!/usr/bin/env python3
"""KMIX Part-A on IVN 终端矩阵 (P5 P0 补强, 260901). 零 GPU.

目的 (老丁 9-01 批准):
  ① 把 F3(c) 从"不能当验证的外推"升级为**同矩阵桥检验** —— IVN 四臂此前从未做过
     行归一重拟合, 故根本算不出 Δ_k^mix; 本脚本补上, 于是 base↔ropeperm 的臂间比较
     可以显式分解出 k_norm 的贡献 (此前"k_norm 不变"是未检验的那一步);
  ② 检验 α_P 是否跨 regime 稳定 (TCLOCK 拟合窗 ≤5000 步 vs IVN 终端 20k–30k)。

口径与 kmix_bridge_v1_1.py 逐字一致 (不得改):
  - fit_hist_r2: 1024 bins, log10 范围 (-12,2), middle-80% (0.1≤F≤0.9) 加权概率图回归;
  - rownorm: 每行除以自身 RMS 后重标到全局 RMS;
  - KINDS_A = q/k/v/gate/up (输出行), o_proj/down_proj 不入 (行不是输出维);
  - origin_fit: 过原点 y=αx, 主判据 R0² = 1 − Σ(y−αx)²/Σy².
判决无论方向如何都照报: α_P 若漂移则写入 Limitations, 不得只报有利一侧。
输出 08_paper5_draft/data/p5_kmix_on_ivn_v1.json。
用法: python3 scripts/p5_kmix_on_ivn_v1.py"""
import glob, hashlib, json, os, re
import numpy as np
import torch

IVN = "05_ivn_runs"
ARMS = ("base", "flatv", "nomom", "ropeperm")
STEPS = (800, 3200, 5000, 10000, 15000, 20000, 25000, 30000)
EARLY = (800, 3200, 5000)          # TCLOCK 拟合窗同量级
LATE = (20000, 25000, 30000)       # IVN 终端 regime
HB, HR = 1024, (-12.0, 2.0)
KINDS_A = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
ALPHA_TCLOCK = {"q_proj": 0.7986408775073409, "k_proj": 0.7851889890738926,
                "qk_pooled": 0.7913930565519072, "v_proj": 0.5929727314869616,
                "gate_proj": 0.6370214866786047, "up_proj": 0.5881924039923598}
N_LAYERS = 6


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
    r2f = 1 - (w * res ** 2).sum() / (w * (y - ybar) ** 2).sum()
    return float(beta[0]), float(r2f)


def khist(x):
    h = np.histogram(np.log10(np.maximum(np.abs(x.ravel()), 1e-13)), bins=HB, range=HR)[0]
    return fit_hist_r2(h)


def rownorm(w):
    r = np.sqrt((w ** 2).mean(1, keepdims=True))
    z = w / np.maximum(r, 1e-30)
    return z * np.sqrt((w ** 2).mean()) / np.sqrt((z ** 2).mean())


# ---------- 输入断言 (守则: 绘图/分析脚本必须 assert 输入) ----------
for arm in ARMS:
    for t in STEPS:
        p = f"{IVN}/ivn_{arm}/ckpt/step{t}.pt"
        assert os.path.exists(p), f"missing ckpt {p}"

# ---------- 逐矩阵读数 ----------
rows = []
for arm in ARMS:
    for t in STEPS:
        st = torch.load(f"{IVN}/ivn_{arm}/ckpt/step{t}.pt", map_location="cpu", weights_only=True)
        n_here = 0
        for nm, w in st.items():
            mm = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.weight$", nm)
            if not mm or mm.group(2) not in KINDS_A:
                continue
            w = w.double().numpy()
            kr, r2r = khist(w); kn, r2n = khist(rownorm(w))
            rows.append({"arm": arm, "step": t, "L": int(mm.group(1)), "kind": mm.group(2),
                         "mat": nm, "k_raw": kr, "k_norm": kn, "r2fit_raw": r2r, "r2fit_norm": r2n,
                         "hw": float(np.log(np.sqrt((w ** 2).mean(1))).std())})
            n_here += 1
        assert n_here == N_LAYERS * len(KINDS_A), f"{arm}@{t}: got {n_here} matrices"
        print(f"read {arm} step{t} ({n_here} mats)", flush=True)
assert len(rows) == len(ARMS) * len(STEPS) * N_LAYERS * len(KINDS_A), len(rows)


def origin_fit(rs):
    x = np.array([r["hw"] ** 2 for r in rs])
    y = np.array([r["k_raw"] ** -2 - r["k_norm"] ** -2 for r in rs])
    a = float((x @ y) / (x @ x))
    r0 = float(1 - ((y - a * x) ** 2).sum() / (y ** 2).sum())
    corr2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    mae = float(np.mean(np.abs((np.array([r["k_norm"] for r in rs]) ** -2 + a * x) ** -0.5
                               - np.array([r["k_raw"] for r in rs]))))
    return {"alpha": a, "R0_sq": r0, "corr_sq_aux": corr2, "gof_mae_insample": mae, "n": len(rs)}


FAMS = {"q_proj": ("q_proj",), "k_proj": ("k_proj",), "qk_pooled": ("q_proj", "k_proj"),
        "v_proj": ("v_proj",), "gate_proj": ("gate_proj",), "up_proj": ("up_proj",)}

# ---------- ② α_P 跨 regime / 跨臂 ----------
partA = {}
for fam, kinds in FAMS.items():
    sub = [r for r in rows if r["kind"] in kinds]
    ent = {"all_steps": origin_fit(sub),
           "early_le5000": origin_fit([r for r in sub if r["step"] in EARLY]),
           "late_20k_30k": origin_fit([r for r in sub if r["step"] in LATE]),
           "by_arm_late": {a: origin_fit([r for r in sub if r["arm"] == a and r["step"] in LATE])
                           for a in ARMS},
           "by_step": {str(t): origin_fit([r for r in sub if r["step"] == t]) for t in STEPS},
           "alpha_TCLOCK_le5000": ALPHA_TCLOCK[fam]}
    ent["ratio_late_over_TCLOCK"] = ent["late_20k_30k"]["alpha"] / ALPHA_TCLOCK[fam]
    ent["ratio_early_over_TCLOCK"] = ent["early_le5000"]["alpha"] / ALPHA_TCLOCK[fam]
    ent["arm_alpha_spread_late"] = float(np.ptp([ent["by_arm_late"][a]["alpha"] for a in ARMS]))
    partA[fam] = ent

qk = partA["qk_pooled"]
partA["J_regime"] = {
    "criterion": "α_P 跨 regime 稳定 = late/TCLOCK 比值 ∈ [0.8,1.25] 且 late R0² ≥ 0.9",
    "ratio_late_over_TCLOCK": qk["ratio_late_over_TCLOCK"],
    "R0_sq_late": qk["late_20k_30k"]["R0_sq"],
    "verdict": ("stable" if 0.8 <= qk["ratio_late_over_TCLOCK"] <= 1.25
                and qk["late_20k_30k"]["R0_sq"] >= 0.9 else "drifted")}

# ---------- ① 同矩阵桥: base ↔ ropeperm 臂间分解 ----------
def idx(rs):
    return {(r["arm"], r["step"], r["L"], r["kind"]): r for r in rs}
R = idx(rows)
bridge = {"per_matrix": [], "summary": {}}
for t in LATE:
    for L in range(N_LAYERS):
        for kd in KINDS_A:
            b = R[("base", t, L, kd)]; p = R[("ropeperm", t, L, kd)]
            dH2 = p["hw"] ** 2 - b["hw"] ** 2
            d_raw = p["k_raw"] ** -2 - b["k_raw"] ** -2          # 实测臂间 Δ(k_raw^-2)
            d_norm = p["k_norm"] ** -2 - b["k_norm"] ** -2       # 🔴 此前不可测的那一项
            a = ALPHA_TCLOCK["qk_pooled"] if kd in ("q_proj", "k_proj") else ALPHA_TCLOCK[kd]
            bridge["per_matrix"].append({
                "step": t, "L": L, "kind": kd,
                "hw_base": b["hw"], "hw_perm": p["hw"], "dH2": dH2,
                "k_raw_base": b["k_raw"], "k_raw_perm": p["k_raw"],
                "k_norm_base": b["k_norm"], "k_norm_perm": p["k_norm"],
                "d_raw_inv2": d_raw, "d_norm_inv2": d_norm, "d_mix": d_raw - d_norm,
                "pred_mix_alpha_dH2": a * dH2,
                "resid": (d_raw - d_norm) - a * dH2,
                "naive_pred_err": d_raw - a * dH2})   # 旧外推 (假设 k_norm 不变) 的误差
for kd in KINDS_A:
    s = [r for r in bridge["per_matrix"] if r["kind"] == kd]
    dm = np.array([r["d_mix"] for r in s]); pr = np.array([r["pred_mix_alpha_dH2"] for r in s])
    dn = np.array([r["d_norm_inv2"] for r in s]); dr = np.array([r["d_raw_inv2"] for r in s])
    bridge["summary"][kd] = {
        "n": len(s),
        "sign_agree_frac": float(np.mean(np.sign(dm) == np.sign(pr))),
        "med_d_mix": float(np.median(dm)), "med_pred": float(np.median(pr)),
        "med_ratio_dmix_over_pred": float(np.median(dm / np.where(pr == 0, np.nan, pr))),
        "med_abs_d_norm": float(np.median(np.abs(dn))),
        "med_abs_d_raw": float(np.median(np.abs(dr))),
        "d_norm_share_of_d_raw_med": float(np.median(np.abs(dn) / np.maximum(np.abs(dr), 1e-15))),
        "med_abs_resid_true_bridge": float(np.median(np.abs([r["resid"] for r in s]))),
        "med_abs_err_naive_extrap": float(np.median(np.abs([r["naive_pred_err"] for r in s]))),
        "corr_dmix_pred": (float(np.corrcoef(dm, pr)[0, 1]) if len(s) > 2 and dm.std() > 0 else None)}
qkb = [r for r in bridge["per_matrix"] if r["kind"] in ("q_proj", "k_proj")]
dm = np.array([r["d_mix"] for r in qkb]); pr = np.array([r["pred_mix_alpha_dH2"] for r in qkb])
bridge["summary"]["qk_pooled"] = {
    "n": len(qkb), "sign_agree_frac": float(np.mean(np.sign(dm) == np.sign(pr))),
    "corr_dmix_pred": float(np.corrcoef(dm, pr)[0, 1]),
    "med_ratio_dmix_over_pred": float(np.median(dm / np.where(pr == 0, np.nan, pr))),
    "med_abs_resid_true_bridge": float(np.median(np.abs([r["resid"] for r in qkb]))),
    "med_abs_err_naive_extrap": float(np.median(np.abs([r["naive_pred_err"] for r in qkb])))}
bridge["caveats"] = [
    "单 seed 配对 (IVN 为 single-seed screen), 无误差棒; 不得升级为 confirmatory",
    "臂间比较不是 KMIX 桥本身的检验; 桥是同一矩阵内 raw vs 行归一重拟合的恒等分解",
    "α 取自 TCLOCK ≤5000 步拟合; 若 J_regime 判 drifted, 本表预测项须改用 IVN late α"]

fit_ok = {"r2fit_raw_min": float(min(r["r2fit_raw"] for r in rows)),
          "r2fit_raw_med": float(np.median([r["r2fit_raw"] for r in rows])),
          "r2fit_norm_min": float(min(r["r2fit_norm"] for r in rows)),
          "r2fit_norm_med": float(np.median([r["r2fit_norm"] for r in rows]))}

res = {"partA_regime": partA, "bridge_base_vs_ropeperm": bridge, "fit_adequacy": fit_ok,
       "per_matrix_readouts": rows,
       "provenance": {"script": os.path.basename(__file__),
                      "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
                      "protocol": "逐字复用 kmix_bridge_v1_1.py 的 fit_hist_r2 / rownorm / origin_fit",
                      "source": "05_ivn_runs/ivn_{base,flatv,nomom,ropeperm}/ckpt/step*.pt",
                      "steps": list(STEPS), "kinds": list(KINDS_A),
                      "alpha_reference": ALPHA_TCLOCK}}
os.makedirs("08_paper5_draft/data", exist_ok=True)
out = "08_paper5_draft/data/p5_kmix_on_ivn_v1.json"
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"J_regime": partA["J_regime"],
                  "qk_alpha": {"TCLOCK_le5000": ALPHA_TCLOCK["qk_pooled"],
                               "IVN_early": qk["early_le5000"],
                               "IVN_late": qk["late_20k_30k"]},
                  "bridge_qk": bridge["summary"]["qk_pooled"],
                  "fit_adequacy": fit_ok}, indent=1))
print(f"-> {out}")
