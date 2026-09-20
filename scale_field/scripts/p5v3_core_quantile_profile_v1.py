#!/usr/bin/env python3
"""Paper#5 v3 §3：归一化主体的**分位轮廓** ψ_Z(p)，不只是 k（零 GPU，260915）。

动机（骨架 §3）：只报 k_bi 等于把「主体」又压回一个标量，与本文「不要标量转标量」的主张自相矛盾。
要说「主体近似共同」，必须给出**整条形状**。

定义：Z = 双侧归一后的核（协议 verbatim 自 p5v2_ea_kbi_trajectory_v1.py 的 bothnorm_tracked，
      带 W = diag(a)·Z·diag(b) 恒等式断言），再整体缩放到单位 RMS。
      ψ_Z(p) = |Z| 的 p 分位。
参照：同协议下的高斯参照＝单位 RMS 的半正态，ψ_ref(p) = Φ⁻¹((1+p)/2)。
读数：ψ_Z(p)/ψ_ref(p) − 1，报 middle-80%（p∈[.1,.9]，即拟合所用区间）内的最大偏离，
      并把尾部（p 到 .99）一并画出以显示偏离出现在哪里。

⚠️ 这是**可证伪的**：k_bi 相同不蕴含 ψ_Z 相同。
输出 08_paper5_draft/data/p5v3_core_quantile_profile_v1.json
      （绘图见 scripts/p5v3_fig1b_core_quantile_plot_v1.py，读同一 json）
"""
import hashlib, json, re
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "04_ea_runs/ckpt_key"
OUT = ROOT / "08_paper5_draft/data/p5v3_core_quantile_profile_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
INITS = ["gaussian", "laplace", "uniform"]
SEEDS = [1, 2, 3]
STEPS = [0, 800, 3200, 30000]
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NLAYER = 6
P = np.concatenate([np.arange(0.05, 0.90001, 0.05), [0.925, 0.95, 0.975, 0.99]])
BAND = (0.10, 0.90)                      # 与拟合协议同一 middle-80%


def bothnorm_tracked(w, it=200, tol=1e-6):
    """verbatim from p5v2_ea_kbi_trajectory_v1.py"""
    scale = np.sqrt((w ** 2).mean())
    a = np.ones(w.shape[0]); b = np.ones(w.shape[1]); z = w.copy()
    for _ in range(it):
        r = np.sqrt((z ** 2).mean(1)); z = z / r[:, None]; a = a * r
        c = np.sqrt((z ** 2).mean(0)); z = z / c[None, :]; b = b * c
        rr = np.sqrt((z ** 2).mean(1)); cc = np.sqrt((z ** 2).mean(0))
        if rr.std() / rr.mean() < tol and cc.std() / cc.mean() < tol:
            break
    g = scale / np.sqrt((z ** 2).mean()); z = z * g; a = a / np.sqrt(g); b = b / np.sqrt(g)
    rr = np.sqrt((z ** 2).mean(1)); cc = np.sqrt((z ** 2).mean(0))
    assert rr.std() / rr.mean() < 1e-3 and cc.std() / cc.mean() < 1e-3, "not converged"
    assert np.abs(a[:, None] * z * b[None, :] - w).max() < 1e-9 * scale, "W = diag(a) Z diag(b) violated"
    return z


def do_ckpt(args):
    init, seed, step = args
    p = CK / f"ea_{init}_s{seed}/ckpt/step{step}.pt"
    st = torch.load(p, map_location="cpu", weights_only=True)
    st = st.get("model", st) if isinstance(st, dict) and "model" in st else st
    rows = []
    for nm, w in st.items():
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", nm)
        if not m or m.group(2) not in KINDS:
            continue
        W = w.double().numpy()
        Z = bothnorm_tracked(W)
        z = np.abs(Z).ravel(); z = z / np.sqrt((z ** 2).mean())      # 单位 RMS
        rows.append({"init": init, "seed": seed, "step": step,
                     "comp": m.group(2), "layer": int(m.group(1)),
                     "psi": np.quantile(z, P).tolist()})
    assert len(rows) == NLAYER * len(KINDS), (p, len(rows))
    return rows


if __name__ == "__main__":
    jobs = [(i, s, t) for i in INITS for s in SEEDS for t in STEPS]
    with Pool(min(8, len(jobs))) as pool:
        out = pool.map(do_ckpt, jobs)
    rows = [r for chunk in out for r in chunk]
    assert len(rows) == len(jobs) * NLAYER * len(KINDS), len(rows)

    PSI_REF = norm.ppf((1 + P) / 2.0)                   # 半正态分位
    PSI_REF = PSI_REF / np.sqrt(np.mean(norm.ppf((1 + np.linspace(1e-6, 1 - 1e-6, 200000)) / 2) ** 2))
    band = (P >= BAND[0]) & (P <= BAND[1])

    def agg(sel):
        A = np.array([r["psi"] for r in sel], float) / PSI_REF[None, :] - 1.0
        return {"n": len(sel), "median": np.median(A, 0).tolist(),
                "q10": np.quantile(A, .10, 0).tolist(), "q90": np.quantile(A, .90, 0).tolist(),
                "max_abs_dev_band": float(np.abs(np.median(A, 0)[band]).max()),
                "max_abs_dev_band_worst_matrix": float(np.abs(A[:, band]).max())}

    res = {"schema": "p5v3-core-quantile-profile-v1", "script_sha256_16": SHA,
           "p_levels": P.tolist(), "band": list(BAND), "psi_ref_unit_rms": PSI_REF.tolist(),
           "protocol": "Z = two-sided normalized core (verbatim bothnorm_tracked), rescaled to unit "
                       "RMS; psi_Z(p) = quantile of |Z|; reference = unit-RMS half-normal",
           "steps": STEPS, "inits": INITS, "seeds": SEEDS, "kinds": KINDS,
           "by_init_step": {f"{i}@{t}": agg([r for r in rows if r["init"] == i and r["step"] == t])
                            for i in INITS for t in STEPS},
           "by_kind_final": {k: agg([r for r in rows if r["comp"] == k and r["step"] == STEPS[-1]
                                     and r["init"] == "gaussian"]) for k in KINDS},
           "all_final": agg([r for r in rows if r["step"] == STEPS[-1]]),
           "rows": rows}
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT.relative_to(ROOT), f"({len(rows)} matrices)")
    print(f"\n=== ψ_Z(p)/ψ_ref(p) − 1 的 middle-80% 内最大偏离（中位轮廓）===")
    print(f"{'init':<10s}" + "".join(f"{t:>10d}" for t in STEPS))
    for i in INITS:
        print(f"{i:<10s}" + "".join(f"{res['by_init_step'][f'{i}@{t}']['max_abs_dev_band']:>10.4f}"
                                    for t in STEPS))
    print(f"\n终态（30,000）逐 kind，高斯族：")
    for k in KINDS:
        b = res["by_kind_final"][k]
        print(f"  {k.replace('_proj',''):<6s} 中位轮廓最大偏离 {b['max_abs_dev_band']:.4f}   "
              f"最差单矩阵 {b['max_abs_dev_band_worst_matrix']:.4f}  (n={b['n']})")
    a = res["all_final"]
    print(f"\n全部三族终态合并: 中位轮廓最大偏离 {a['max_abs_dev_band']:.4f}, "
          f"最差单矩阵 {a['max_abs_dev_band_worst_matrix']:.4f} (n={a['n']})")
