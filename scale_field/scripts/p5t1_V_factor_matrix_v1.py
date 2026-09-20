#!/usr/bin/env python3
"""V̂ 因子的全对相关矩阵：通道拓扑是否在二阶矩里留下分块结构（零 GPU，260914）。

每个组件有两个因子（log V̂ 的行均值 a_i、列均值 b_j），7 组件 ⇒ 14 个向量。
按**通道空间**分组（这就是预言本身）：
  residual (512)     q.col k.col v.col gate.col up.col  [输入侧 x 能量]  +  o.row down.row [输出侧 δ 能量]
  qk-logit (512)     q.row k.row
  head/value (512)   v.row o.col
  ffn-hidden (1376)  gate.row up.row down.col
⇒ 若 V̂ 的因子由通道拓扑组织，相关矩阵应呈**分块**结构；跨组 ≈ 0。
维度不同的对不可比，置 NaN。同时输出权重侧同一矩阵作对照。
输出 08_paper5_draft/data/p5t1_V_factor_matrix_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "10_p5batch_cloud/results/dump_early/tensors"
OUT = ROOT / "08_paper5_draft/data/p5t1_V_factor_matrix_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
NLAYER, N_NULL = 6, 200
RNG = np.random.default_rng(0)

# (标签, 组件, 侧, 通道空间, 能量类型)  side: row -> a_i (delta energy), col -> b_j (x energy)
SLOTS = [
    # 通道空间按**实际张量**定义，不按笼统的「残差流」：
    #   q/k/v 读 input-LN 的输出；gate/up 读 post-attn-LN 的输出；o/down 的行是写回残差的误差
    ("q.col",    "q_proj",    "col", "resid-pre",  "x"),
    ("k.col",    "k_proj",    "col", "resid-pre",  "x"),
    ("v.col",    "v_proj",    "col", "resid-pre",  "x"),
    ("gate.col", "gate_proj", "col", "resid-post", "x"),
    ("up.col",   "up_proj",   "col", "resid-post", "x"),
    ("o.row",    "o_proj",    "row", "resid-err",  "d"),
    ("down.row", "down_proj", "row", "resid-err",  "d"),
    ("q.row",    "q_proj",    "row", "qk-logit",   "d"),
    ("k.row",    "k_proj",    "row", "qk-logit",   "d"),
    ("v.row",    "v_proj",    "row", "head/value", "d"),
    ("o.col",    "o_proj",    "col", "head/value", "x"),
    ("gate.row", "gate_proj", "row", "ffn-hidden", "d"),
    ("up.row",   "up_proj",   "row", "ffn-hidden", "d"),
    ("down.col", "down_proj", "col", "ffn-hidden", "x"),
]
N = len(SLOTS)

steps = sorted(int(re.search(r"step(\d+)", f.name).group(1)) for f in DUMP.glob("step*.pt"))
assert steps, f"缺 {DUMP}"

Vf, Wf = {}, {}
for st in steps:
    d = torch.load(DUMP / f"step{st}.pt", map_location="cpu", weights_only=False)
    assert d["__meta__"]["step"] == st
    for key in d:
        if key == "__meta__" or not key.endswith("|W"):
            continue
        base = key[:-2]
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", base)
        assert m, base
        L, kind = int(m.group(1)), m.group(2)
        W = d[base + "|W"].double().numpy(); V = d[base + "|Vh"].double().numpy()
        assert W.shape == V.shape
        LV = np.log(V + np.finfo(np.float64).tiny); g = LV.mean()
        Vf[(st, L, kind, "row")] = LV.mean(1) - g
        Vf[(st, L, kind, "col")] = LV.mean(0) - g
        hr = np.log(np.sqrt((W ** 2).mean(1))); hc = np.log(np.sqrt((W ** 2).mean(0)))
        Wf[(st, L, kind, "row")] = hr - np.median(hr)
        Wf[(st, L, kind, "col")] = hc - np.median(hc)
    print("loaded", st, flush=True)
assert len(Vf) == len(steps) * NLAYER * 7 * 2, len(Vf)

# 形状自检：分组内维度必须一致
for grp in {s[3] for s in SLOTS}:
    dims = {Vf[(steps[0], 0, s[1], s[2])].shape[0] for s in SLOTS if s[3] == grp}
    assert len(dims) == 1, (grp, dims)
    print(f"  group {grp:<12s} dim {dims.pop()}")


def matrix(store, only_step=None):
    M = np.full((N, N), np.nan); NL = np.full((N, N), np.nan)
    for i in range(N):
        for j in range(N):
            vi, vj = SLOTS[i], SLOTS[j]
            vals, nulls = [], []
            for st in ([only_step] if only_step is not None else steps):
                for L in range(NLAYER):
                    u = store[(st, L, vi[1], vi[2])]; w = store[(st, L, vj[1], vj[2])]
                    if u.shape != w.shape:
                        continue
                    vals.append(float(spearmanr(u, w).statistic))
                    if i < j:
                        nulls.append(abs(float(spearmanr(RNG.permutation(u), w).statistic)))
            if vals:
                M[i, j] = float(np.median(vals))
                if nulls:
                    NL[i, j] = float(np.quantile(nulls, .95))
    return M, NL


MV, NLV = matrix(Vf)
MW, _ = matrix(Wf)
LAST = max(steps)
MV_L, NLV_L = matrix(Vf, LAST)
MW_L, _ = matrix(Wf, LAST)
# 逐时点矩阵（供补充材料的四时点小倍图；与热图同口径 mean-of-log）
MV_BY = {}
for st in steps:
    m, _ = matrix(Vf, st)
    MV_BY[str(st)] = m.tolist()
    print(f"  per-step matrix {st} done", flush=True)
assert np.allclose(np.diag(MV), 1.0, atol=1e-9), np.diag(MV)

# 组内 / 组间汇总（只取上三角、同维可比对）
def blocks(M):
    same, cross = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if np.isnan(M[i, j]):
                continue
            (same if SLOTS[i][3] == SLOTS[j][3] else cross).append(M[i, j])
    return (float(np.median(same)), len(same)), (float(np.median(cross)), len(cross))

# 按步的重点对（供补充图的时间依赖面板）
WATCH = [("q.col", "k.col"), ("q.col", "v.col"), ("k.col", "v.col"),
         ("gate.col", "up.col"), ("o.row", "down.row"), ("q.row", "k.row"),
         ("v.row", "o.col"), ("gate.row", "up.row")]
IDX = {s[0]: i for i, s in enumerate(SLOTS)}
by_step_pairs = {}
for a, b in WATCH:
    sa, sb = SLOTS[IDX[a]], SLOTS[IDX[b]]
    ser = {}
    for st in steps:
        v = [float(spearmanr(Vf[(st, L, sa[1], sa[2])], Vf[(st, L, sb[1], sb[2])]).statistic)
             for L in range(NLAYER)]
        ser[str(st)] = float(np.median(v))
    by_step_pairs[f"{a}~{b}"] = ser
    print(f"  {a}~{b:<10s} " + " ".join(f"{ser[str(st)]:+.3f}" for st in steps))

OUT.write_text(json.dumps({
    "schema": "p5-tier1-V-factor-matrix-v1", "script_sha256_16": SHA, "steps": steps,
    "slots": [{"label": s[0], "kind": s[1], "side": s[2], "space": s[3], "energy": s[4]} for s in SLOTS],
    "note": "median Spearman over 5 steps x 6 layers; NaN where the two factors have different lengths; "
            "null = 200 within-layer permutations (upper triangle)",
    "by_step_pairs": by_step_pairs, "last_step": LAST,
    "V_matrix_by_step": MV_BY, "V_matrix_last": MV_L.tolist(), "V_null_abs_p95_last": NLV_L.tolist(),
    "W_matrix_last": MW_L.tolist(),
    "V_matrix": MV.tolist(), "V_null_abs_p95": NLV.tolist(), "W_matrix": MW.tolist(),
    "summary": {"V": {"within_space": blocks(MV)[0], "cross_space": blocks(MV)[1]},
                "W": {"within_space": blocks(MW)[0], "cross_space": blocks(MW)[1]},
                "V_last": {"within_space": blocks(MV_L)[0], "cross_space": blocks(MV_L)[1]},
                "W_last": {"within_space": blocks(MW_L)[0], "cross_space": blocks(MW_L)[1]}}}, indent=1))
print("\nwrote", OUT.relative_to(ROOT))
lab = [s[0] for s in SLOTS]
print("\n=== V̂ 因子相关矩阵（中位 Spearman；· = 维度不可比）===")
print("            " + "".join(f"{l[:8]:>9s}" for l in lab))
for i in range(N):
    print(f"{lab[i]:<12s}" + "".join("        ·" if np.isnan(MV[i, j]) else f"{MV[i,j]:>9.2f}" for j in range(N))
          + f"   [{SLOTS[i][3]}/{SLOTS[i][4]}]")
(sw, nw), (sc, nc) = blocks(MV)
print(f"\nV̂: 组内中位 {sw:+.3f} (n={nw})   组间中位 {sc:+.3f} (n={nc})   null≈{np.nanmedian(NLV):.3f}")
(sw2, nw2), (sc2, nc2) = blocks(MW)
print(f"W : 组内中位 {sw2:+.3f} (n={nw2})   组间中位 {sc2:+.3f} (n={nc2})")
print(f"\n=== 仅末步 {LAST} ===")
print("            " + "".join(f"{l[:8]:>9s}" for l in lab))
for i in range(N):
    print(f"{lab[i]:<12s}" + "".join("        ·" if np.isnan(MV_L[i, j]) else f"{MV_L[i,j]:>9.2f}" for j in range(N))
          + f"   [{SLOTS[i][3]}/{SLOTS[i][4]}]")
(a1, n1), (a2, n2) = blocks(MV_L); (b1_, m1), (b2_, m2) = blocks(MW_L)
print(f"\nV̂@{LAST}: 组内 {a1:+.3f} (n={n1})  组间 {a2:+.3f} (n={n2})")
print("\n组内逐块（末步）:")
for grp in dict.fromkeys(sl[3] for sl in SLOTS):
    ix = [i for i, sl in enumerate(SLOTS) if sl[3] == grp]
    vv = [MV_L[i, j] for a_, i in enumerate(ix) for j in ix[a_ + 1:]]
    if vv:
        print(f"  {grp:<12s} n={len(vv):>2d}  中位 {np.median(vv):+.3f}  范围 [{min(vv):+.2f},{max(vv):+.2f}]")
print("\n跨组中 |r| 超过 null 的对（末步）:")
for i in range(N):
    for j in range(i + 1, N):
        if np.isnan(MV_L[i, j]) or SLOTS[i][3] == SLOTS[j][3]:
            continue
        if abs(MV_L[i, j]) > (NLV_L[i, j] if not np.isnan(NLV_L[i, j]) else 0.08):
            print(f"  {lab[i]:<10s}~{lab[j]:<10s} {MV_L[i,j]:+.3f}  (null {NLV_L[i,j]:.3f})  "
                  f"[{SLOTS[i][3]} vs {SLOTS[j][3]}]")
print(f"W @{LAST}: 组内 {b1_:+.3f} (n={m1})  组间 {b2_:+.3f} (n={m2})")
