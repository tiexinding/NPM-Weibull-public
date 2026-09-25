#!/usr/bin/env python3
"""同一条 run、150 点、到终态：V̂ 与 W 的锁定与配对涌现（零 GPU，260914）。

C.5 用第二批 dump 的 5 点网格得到「无一致领先」，但那是稀疏且参照点只到 10,000 步。
本脚本用第三批 R0_base_snap 的 150 点 logger 重做，参照点 = 最后一个窗口（终态），
以判定第三批「V̂ 锁定领先场 1,600–2,200 步」与 C.5 的混合符号如何调和。

读数（logger 已存的身份轴边缘，Spearman 对单调变换不变，故不需去中位）：
  V̂ 侧  sqrtvhat_{row,col}_lnrms   = ln sqrt(mean over the other axis of V̂)
  W 侧   h_{row,col}_start          = ln RMS of W along the axis, 窗口起点
⚠️ 与 C.5 的 a_i = mean_j log V̂_ij 不是同一聚合（log-of-mean vs mean-of-log），
   两者都是「二阶矩的行/列边缘」，但数值不可逐位互引。
输出 08_paper5_draft/data/p5t1_lockin_dense_R0_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "11_p5mech_cloud/results/R0_base_snap/logger"
OUT = ROOT / "08_paper5_draft/data/p5t1_lockin_dense_R0_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
NLAYER = 6
KINDS7 = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "o_proj", "down_proj"]
IDENT = {"q_proj": "row", "k_proj": "row", "v_proj": "row", "gate_proj": "row",
         "up_proj": "row", "o_proj": "col", "down_proj": "col"}
PATHS = [("VO", ("v_proj", "row"), ("o_proj", "col")),
         ("UD", ("up_proj", "row"), ("down_proj", "col")),
         ("GD", ("gate_proj", "row"), ("down_proj", "col")),
         ("QK", ("q_proj", "row"), ("k_proj", "row"))]
FIELD = {"V": "sqrtvhat_{}_lnrms", "W": "h_{}_start"}

files = sorted(LOG.glob("win*.npz"), key=lambda p: int(re.search(r"win(\d+)", p.name).group(1)))
assert len(files) >= 100, f"logger 点太少: {len(files)}"
print(f"{len(files)} 个窗口: {files[0].name} .. {files[-1].name}")

S = {}   # S[(t, layer, kind, side, which)] = vector
ts = []
for f in files:
    z = np.load(f, allow_pickle=True)
    t0 = int(np.asarray(z["__meta__|t_start"]).item())
    ts.append(t0)
    for kind in KINDS7:
        for L in range(NLAYER):
            pre = ("model.layers.%d.self_attn.%s.weight" % (L, kind)) if kind in (
                "q_proj", "k_proj", "v_proj", "o_proj") else ("model.layers.%d.mlp.%s.weight" % (L, kind))
            for which, tmpl in FIELD.items():
                for side in ("row", "col"):
                    key = f"{pre}|{tmpl.format(side)}"
                    assert key in z, key
                    S[(t0, L, kind, side, which)] = np.asarray(z[key], dtype=float)
ts = sorted(set(ts)); T = ts[-1]
assert len(ts) == len(files), (len(ts), len(files))
print(f"步位 {ts[0]} .. {T}")

sp = lambda a, b: float(spearmanr(a, b).statistic)
lockin = {"V": {}, "W": {}}
for which in ("V", "W"):
    for kind in KINDS7:
        side = IDENT[kind]
        lockin[which][kind] = {str(t): float(np.median(
            [sp(S[(t, L, kind, side, which)], S[(T, L, kind, side, which)]) for L in range(NLAYER)]))
            for t in ts}
pairing = {"V": {}, "W": {}}
for which in ("V", "W"):
    for pn, (k1, s1), (k2, s2) in PATHS:
        pairing[which][pn] = {str(t): float(np.median(
            [sp(S[(t, L, k1, s1, which)], S[(t, L, k2, s2, which)]) for L in range(NLAYER)]))
            for t in ts}


def cross(series, thr):
    xs = [int(k) for k in series]; ys = [series[str(x)] for x in xs]
    for i in range(1, len(xs)):
        if ys[i - 1] < thr <= ys[i]:
            f = (thr - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
    return None


THR = [0.5, 0.7, 0.9]
cr = {"lockin": {w: {k: {str(t): cross(lockin[w][k], t) for t in THR} for k in KINDS7} for w in ("V", "W")},
      "pairing": {w: {p[0]: {str(t): cross(pairing[w][p[0]], t) for t in THR} for p in PATHS} for w in ("V", "W")}}

OUT.write_text(json.dumps({"schema": "p5-tier1-lockin-dense-R0-v1", "script_sha256_16": SHA,
                           "run": "11_p5mech_cloud/results/R0_base_snap", "n_windows": len(ts),
                           "steps": ts, "reference_step": T, "thresholds": THR,
                           "readout": {k: v for k, v in FIELD.items()},
                           "caveat": "sqrt(V-hat) row/col ln RMS is log-of-mean, not the mean-of-log "
                                     "factor used in the dump-based analysis; both are second-moment "
                                     "marginals but the numbers are not interchangeable",
                           "lockin": lockin, "pairing": pairing, "crossings": cr}, indent=1))
print("\nwrote", OUT.relative_to(ROOT))

print(f"\n=== 锁定达阈步位（参照 = 终态 {T}）；Δ = W − V̂，正 = V̂ 领先 ===")
print(f"{'kind':<11s}" + "".join(f"{'V@'+str(t):>9s}{'W@'+str(t):>9s}{'Δ':>8s}" for t in THR))
lead = {t: [] for t in THR}
for kind in KINDS7:
    row = f"{kind:<11s}"
    for t in THR:
        a, b = cr["lockin"]["V"][kind][str(t)], cr["lockin"]["W"][kind][str(t)]
        d = (b - a) if (a and b) else None
        if d is not None:
            lead[t].append(d)
        row += f"{(f'{a:.0f}' if a else '—'):>9s}{(f'{b:.0f}' if b else '—'):>9s}{(f'{d:+.0f}' if d is not None else '—'):>8s}"
    print(row)
print("\n七组件的 Δ 中位（正 = V̂ 领先）：" + "  ".join(
    f"@{t}: {np.median(lead[t]):+.0f} 步 (n={len(lead[t])}, 正号 {sum(1 for x in lead[t] if x>0)}/{len(lead[t])})"
    for t in THR if lead[t]))
print(f"\n=== 配对涌现达阈步位 ===")
print(f"{'path':<6s}" + "".join(f"{'V@'+str(t):>9s}{'W@'+str(t):>9s}{'Δ':>8s}" for t in THR))
for pn, a_, b_ in PATHS:
    row = f"{pn:<6s}"
    for t in THR:
        a, b = cr["pairing"]["V"][pn][str(t)], cr["pairing"]["W"][pn][str(t)]
        d = (b - a) if (a and b) else None
        row += f"{(f'{a:.0f}' if a else '—'):>9s}{(f'{b:.0f}' if b else '—'):>9s}{(f'{d:+.0f}' if d is not None else '—'):>8s}"
    print(row)
