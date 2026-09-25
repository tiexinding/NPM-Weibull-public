#!/usr/bin/env python3
"""V̂ 的配对滞后：是「窗口长」还是「平方被最大值主导」？（零 GPU，260914）

C.7.1 测到 v/o 配对在 V̂ 这一环滞后约 2,900 步 ≈ 名义窗 1/(1−β₂)=1000 的 3 倍。
两个候选：(i) 单纯是 EMA 的窗口长；(ii) V̂ 是**平方**的 EMA，被窗口内最大梯度主导，
    在梯度幅值不均时有效记忆远长于名义窗。
判别：用 logger 的 G 行/列边缘直接仿真两族 EMA —— **同一个 β₂**下，
    sq  : EMA of (G 的均方边缘)      ← 真实 V̂ 的形式
    lin : EMA of (G 的 RMS 边缘)     ← 同窗口但不平方
若 sq 复现 3,528 而 lin 明显更早 ⇒ (ii)；若两者都滞后相近 ⇒ (i)。
再扫 β₂ ∈ {.99,.999,.9999} 看达阈时刻是否随名义窗线性。

⚠️ 近似：logger 每 200 步取 20 步窗，故按**窗级** EMA 仿真，衰减 = β₂^200；
   且 G_row_lnrms 是窗内 ln RMS 的均值（几何平均），非算术平均。两者都在下方标注。
输出 08_paper5_draft/data/p5t1_vhat_lag_simulation_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "11_p5mech_cloud/results/R0_base_snap/logger"
OUT = ROOT / "08_paper5_draft/data/p5t1_vhat_lag_simulation_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
NLAYER, EVERY = 6, 200
BETAS = [0.99, 0.999, 0.9999]
B2_TRUE = 0.999
PATHS = [("VO", ("v_proj", "row"), ("o_proj", "col")),
         ("UD", ("up_proj", "row"), ("down_proj", "col"))]
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
pre = lambda L, k: f"model.layers.{L}." + ("self_attn." if k in ATTN else "mlp.") + f"{k}.weight"

files = sorted(LOG.glob("win*.npz"), key=lambda p: int(re.search(r"win(\d+)", p.name).group(1)))
assert len(files) >= 100, len(files)
need = {(k, s) for _, (k1, s1), (k2, s2) in PATHS for k, s in ((k1, s1), (k2, s2))}
G, VT, ts = {}, {}, []
for f in files:
    z = np.load(f, allow_pickle=True)
    t0 = int(np.asarray(z["__meta__|t_start"]).item()); ts.append(t0)
    assert int(np.asarray(z["__meta__|n_steps"]).item()) > 0
    for kind, side in need:
        for L in range(NLAYER):
            G[(t0, L, kind, side)] = np.asarray(z[f"{pre(L,kind)}|G_{side}_lnrms"], float)
            VT[(t0, L, kind, side)] = np.asarray(z[f"{pre(L,kind)}|sqrtvhat_{side}_lnrms"], float)
ts = sorted(set(ts)); assert len(ts) == len(files)
gaps = np.diff(ts); assert np.all(gaps == EVERY), np.unique(gaps)
print(f"{len(ts)} 窗口, 步距 {EVERY}, {ts[0]}..{ts[-1]}")

sp = lambda a, b: float(spearmanr(a, b).statistic)


def cross(xs, ys, thr):
    for i in range(1, len(xs)):
        if ys[i - 1] < thr <= ys[i]:
            f = (thr - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
    return None


def simulate(b2, square):
    """窗级 EMA。square=True 用均方边缘 exp(2*lnrms)，False 用 RMS 边缘 exp(lnrms)。"""
    dec = b2 ** EVERY
    state, out = {}, {}
    for t in ts:
        for kind, side in need:
            for L in range(NLAYER):
                v = np.exp((2.0 if square else 1.0) * G[(t, L, kind, side)])
                key = (L, kind, side)
                state[key] = v if key not in state else dec * state[key] + (1 - dec) * v
                out[(t, L, kind, side)] = np.log(state[key])
    return out


res = {"true_Vhat": {}, "sims": {}}
for pn, (k1, s1), (k2, s2) in PATHS:
    ys = [float(np.median([sp(VT[(t, L, k1, s1)], VT[(t, L, k2, s2)]) for L in range(NLAYER)])) for t in ts]
    res["true_Vhat"][pn] = {"curve": ys, "t70": cross(ts, ys, 0.7)}
    ysg = [float(np.median([sp(G[(t, L, k1, s1)], G[(t, L, k2, s2)]) for L in range(NLAYER)])) for t in ts]
    res["true_Vhat"][pn + "_G"] = {"curve": ysg, "t70": cross(ts, ysg, 0.7)}

for b2 in BETAS:
    for square in (True, False):
        S = simulate(b2, square)
        tag = f"b2={b2}|{'sq' if square else 'lin'}"
        res["sims"][tag] = {"beta2": b2, "square": square,
                            "nominal_window": 1.0 / (1 - b2)}
        for pn, (k1, s1), (k2, s2) in PATHS:
            ys = [float(np.median([sp(S[(t, L, k1, s1)], S[(t, L, k2, s2)]) for L in range(NLAYER)])) for t in ts]
            res["sims"][tag][pn] = {"curve": ys, "t70": cross(ts, ys, 0.7)}
        print(f"  {tag:<18s} " + "  ".join(
            f"{pn} t@0.7 = {res['sims'][tag][pn]['t70']:.0f}" if res["sims"][tag][pn]["t70"]
            else f"{pn} t@0.7 = —" for pn, _, _ in PATHS), flush=True)

OUT.write_text(json.dumps({"schema": "p5-tier1-vhat-lag-simulation-v1", "script_sha256_16": SHA,
                           "steps": ts, "every": EVERY, "betas": BETAS, "beta2_true": B2_TRUE,
                           "caveat": "window-level EMA (decay beta2^200) on the logger's G marginals, "
                                     "which are the mean over window steps of ln RMS (geometric, not "
                                     "arithmetic); an approximation to the true per-step EMA",
                           **res}, indent=1))
print("\nwrote", OUT.relative_to(ROOT))
print(f"\n=== VO 配对 t@0.7 ===")
print(f"  真实 V̂        {res['true_Vhat']['VO']['t70']:.0f}")
print(f"  真实 G         {res['true_Vhat']['VO_G']['t70']:.0f}")
for b2 in BETAS:
    a = res["sims"][f"b2={b2}|sq"]["VO"]["t70"]; b = res["sims"][f"b2={b2}|lin"]["VO"]["t70"]
    print(f"  β₂={b2:<7} 名义窗 {1/(1-b2):>6.0f}   sq {(f'{a:.0f}' if a else '—'):>7s}   "
          f"lin {(f'{b:.0f}' if b else '—'):>7s}   sq/lin = {(a/b if a and b else float('nan')):.2f}")
print(f"\n=== UD 配对 t@0.7（对照：初始即有配对，应当都很早）===")
print(f"  真实 V̂        {res['true_Vhat']['UD']['t70']:.0f}   真实 G  {res['true_Vhat']['UD_G']['t70']:.0f}")
for b2 in BETAS:
    a = res["sims"][f"b2={b2}|sq"]["UD"]["t70"]; b = res["sims"][f"b2={b2}|lin"]["UD"]["t70"]
    print(f"  β₂={b2:<7} sq {(f'{a:.0f}' if a else '—'):>7s}   lin {(f'{b:.0f}' if b else '—'):>7s}")
