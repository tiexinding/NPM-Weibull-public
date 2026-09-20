#!/usr/bin/env python3
"""把整条链放在同一分析里，定位 2e（gate 锁定反号）与 2f（v/o 配对时间差）在哪一环进来（零 GPU，260914）。

链：  δ (输出侧误差) / x (输入侧激活)  →  G  →  M̂, V̂  →  W
logger（R0，150 窗口，到终态）已存各环的身份轴边缘，故可对每一环同时做：
  锁定 lock-in(t) = corr(X_t, X_终态)
  配对 pairing(t) = corr(路径左端行侧, 路径右端列侧)
2e：若 gate 的 V̂ 迟滞在 δ 那一环就已存在 ⇒ 是信号的性质，不是优化器的性质。
2f：若 v/o 的信号层配对也比 up/down 晚 ⇒ 时间差不是 V̂ 引入的。
输出 08_paper5_draft/data/p5t1_chain_localize_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "11_p5mech_cloud/results/R0_base_snap/logger"
OUT = ROOT / "08_paper5_draft/data/p5t1_chain_localize_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
NLAYER = 6
KINDS7 = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "o_proj", "down_proj"]
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
IDENT = {"q_proj": "row", "k_proj": "row", "v_proj": "row", "gate_proj": "row",
         "up_proj": "row", "o_proj": "col", "down_proj": "col"}
# 链上的各环：名称 -> logger 字段模板（{} 填 row/col）
CHAIN = [("signal", {"row": "delta_row_lnrms", "col": "x_col_lnrms"}),
         ("G",      {"row": "G_row_lnrms",     "col": "G_col_lnrms"}),
         ("Mhat",   {"row": "mhat_row_lnrms",  "col": "mhat_col_lnrms"}),
         ("Vhat",   {"row": "sqrtvhat_row_lnrms", "col": "sqrtvhat_col_lnrms"}),
         ("u",      {"row": "u_row_lnrms",     "col": "u_col_lnrms"}),
         ("W",      {"row": "h_row_start",     "col": "h_col_start"})]
PATHS = [("VO", ("v_proj", "row"), ("o_proj", "col")),
         ("UD", ("up_proj", "row"), ("down_proj", "col")),
         ("GD", ("gate_proj", "row"), ("down_proj", "col"))]
THR = [0.5, 0.7, 0.9]


def pre(L, kind):
    return f"model.layers.{L}." + ("self_attn." if kind in ATTN else "mlp.") + f"{kind}.weight"


files = sorted(LOG.glob("win*.npz"), key=lambda p: int(re.search(r"win(\d+)", p.name).group(1)))
assert len(files) >= 100, len(files)
S, ts = {}, []
for f in files:
    z = np.load(f, allow_pickle=True)
    t0 = int(np.asarray(z["__meta__|t_start"]).item()); ts.append(t0)
    for link, flds in CHAIN:
        for kind in KINDS7:
            for L in range(NLAYER):
                for side, fld in flds.items():
                    key = f"{pre(L, kind)}|{fld}"
                    assert key in z, key
                    S[(t0, L, kind, side, link)] = np.asarray(z[key], dtype=float)
ts = sorted(set(ts)); T = ts[-1]
assert len(ts) == len(files)
print(f"{len(ts)} 窗口, {ts[0]} .. {T}")
sp = lambda a, b: float(spearmanr(a, b).statistic)


def cross(series, thr):
    xs = [int(k) for k in series]; ys = [series[str(x)] for x in xs]
    for i in range(1, len(xs)):
        if ys[i - 1] < thr <= ys[i]:
            f = (thr - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
    return None


lockin, pairing = {}, {}
for link, _ in CHAIN:
    lockin[link] = {}
    for kind in KINDS7:
        side = IDENT[kind]
        lockin[link][kind] = {str(t): float(np.median(
            [sp(S[(t, L, kind, side, link)], S[(T, L, kind, side, link)]) for L in range(NLAYER)]))
            for t in ts}
    pairing[link] = {}
    for pn, (k1, s1), (k2, s2) in PATHS:
        pairing[link][pn] = {str(t): float(np.median(
            [sp(S[(t, L, k1, s1, link)], S[(t, L, k2, s2, link)]) for L in range(NLAYER)]))
            for t in ts}

cr = {"lockin": {l: {k: {str(t): cross(lockin[l][k], t) for t in THR} for k in KINDS7} for l, _ in CHAIN},
      "pairing": {l: {p[0]: {str(t): cross(pairing[l][p[0]], t) for t in THR} for p in PATHS} for l, _ in CHAIN}}

OUT.write_text(json.dumps({"schema": "p5-tier1-chain-localize-v1", "script_sha256_16": SHA,
                           "run": "11_p5mech_cloud/results/R0_base_snap", "steps": ts,
                           "reference_step": T, "thresholds": THR,
                           "chain": [c[0] for c in CHAIN], "fields": {c[0]: c[1] for c in CHAIN},
                           "lockin": lockin, "pairing": pairing, "crossings": cr}, indent=1))
print("\nwrote", OUT.relative_to(ROOT))

fmt = lambda v: (f"{v:.0f}" if v else "—")
print("\n=== 2e  锁定达阈步位 @0.7，沿链（身份轴）===")
print(f"{'kind':<11s}" + "".join(f"{l:>10s}" for l, _ in CHAIN))
for kind in KINDS7:
    print(f"{kind:<11s}" + "".join(f"{fmt(cr['lockin'][l][kind]['0.7']):>10s}" for l, _ in CHAIN)
          + ("   <- gate" if kind == "gate_proj" else ""))
print("\n  各环的七组件中位:")
for l, _ in CHAIN:
    v = [cr["lockin"][l][k]["0.7"] for k in KINDS7 if cr["lockin"][l][k]["0.7"]]
    g = cr["lockin"][l]["gate_proj"]["0.7"]
    print(f"    {l:<8s} 中位 {np.median(v):>7.0f}   gate {fmt(g):>7s}   "
          f"gate/中位 = {(g/np.median(v) if g else float('nan')):.2f}")

print("\n=== 2f  配对达阈步位 @0.7，沿链 ===")
print(f"{'path':<6s}" + "".join(f"{l:>10s}" for l, _ in CHAIN))
for pn, _, _ in PATHS:
    print(f"{pn:<6s}" + "".join(f"{fmt(cr['pairing'][l][pn]['0.7']):>10s}" for l, _ in CHAIN))
print("\n  配对曲线（VO vs UD，signal 环）:")
for pn in ("VO", "UD"):
    idx = [0, 10, 20, 30, 50, 75, 100, 149]
    print(f"    {pn}: " + " ".join(f"{ts[i]}:{pairing['signal'][pn][str(ts[i])]:+.2f}" for i in idx))
