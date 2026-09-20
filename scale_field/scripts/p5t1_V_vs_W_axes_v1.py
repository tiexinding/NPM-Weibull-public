#!/usr/bin/env python3
"""二阶矩的行/列因子 与 权重的行/列尺度场 的关系（零 GPU，260914）。

背景：V = exp_avg_sq 与 W **同形状同索引**（G = δxᵀ ⇒ V 的行 = 输出单元、列 = 输入单元），
所以「V 的行侧/列侧」和「权重侧的行/列」是同一个轴，不是两套坐标。
但「两者的结构是否对齐」此前没测。本脚本直接测。

每矩阵每时点：
  V 侧   logV_ij ≈ c + a_i + b_j 的最小二乘因子；R2_row / R2_col = 单因子解释占比
  W 侧   h_row = ln rowRMS(W) 去中位；h_col = ln colRMS(W) 去中位；H = sd(h)
  对齐   corr(a, h_row)、corr(b, h_col)（Pearson 与 Spearman）
  错位对照  corr(a, h_col) 与 corr(b, h_row) 在可比长度时不可算（维度不同），
            故改用**层内置换 null**：打乱 a 的顺序后重算，200 次。
输出 08_paper5_draft/data/p5t1_V_vs_W_axes_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "10_p5batch_cloud/results/dump_early/tensors"
OUT = ROOT / "08_paper5_draft/data/p5t1_V_vs_W_axes_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
KINDS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "o_proj", "down_proj"]
IDENT_AXIS = {"q_proj": "row", "k_proj": "row", "v_proj": "row", "gate_proj": "row",
              "up_proj": "row", "o_proj": "col", "down_proj": "col"}
NLAYER, N_NULL = 6, 200
RNG = np.random.default_rng(0)

steps = sorted(int(re.search(r"step(\d+)", f.name).group(1)) for f in DUMP.glob("step*.pt"))
assert steps, f"缺 {DUMP}"
rows = []
for st in steps:
    d = torch.load(DUMP / f"step{st}.pt", map_location="cpu", weights_only=False)
    meta = d["__meta__"]
    assert meta["step"] == st, (meta["step"], st)
    seen = set()
    for key in d:
        if key == "__meta__" or not key.endswith("|W"):
            continue
        base = key[:-2]
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", base)
        assert m, base
        L, kind = int(m.group(1)), m.group(2)
        assert kind in KINDS, kind
        W = d[base + "|W"].double().numpy()
        V = d[base + "|Vh"].double().numpy()
        assert W.shape == V.shape, (base, W.shape, V.shape)
        seen.add((L, kind))

        # --- V 侧：对数空间两因子分解 ---
        LV = np.log(V + np.finfo(np.float64).tiny)
        g0 = LV.mean()
        a = LV.mean(1) - g0            # 行因子 (len = out)
        b = LV.mean(0) - g0            # 列因子 (len = in)
        tot = ((LV - g0) ** 2).sum()
        r2_row = float((LV.shape[1] * (a ** 2).sum()) / tot)
        r2_col = float((LV.shape[0] * (b ** 2).sum()) / tot)
        fit = g0 + a[:, None] + b[None, :]
        r2_add = float(1 - ((LV - fit) ** 2).sum() / tot)

        # --- W 侧：行/列尺度场 ---
        hr = np.log(np.sqrt((W ** 2).mean(1))); hr -= np.median(hr)
        hc = np.log(np.sqrt((W ** 2).mean(0))); hc -= np.median(hc)

        def align(u, h):
            pe = float(np.corrcoef(u, h)[0, 1]); sp = float(spearmanr(u, h).statistic)
            nul = np.array([abs(np.corrcoef(RNG.permutation(u), h)[0, 1]) for _ in range(N_NULL)])
            return {"pearson": pe, "spearman": sp, "null_abs_p95": float(np.quantile(nul, .95))}

        rows.append({"step": st, "layer": L, "kind": kind, "shape": list(W.shape),
                     "ident_axis": IDENT_AXIS[kind],
                     "V_R2_add": r2_add, "V_R2_row": r2_row, "V_R2_col": r2_col,
                     "W_H_row": float(hr.std()), "W_H_col": float(hc.std()),
                     "align_row": align(a, hr), "align_col": align(b, hc)})
    assert len(seen) == NLAYER * len(KINDS), (st, len(seen))
    print("step", st, "done", flush=True)

assert len(rows) == len(steps) * NLAYER * len(KINDS), len(rows)


def agg(sel, path):
    def get(r):
        o = r
        for k in path.split("."):
            o = o[k]
        return o
    v = np.array([get(r) for r in sel], float)
    return {"n": len(v), "median": float(np.median(v)),
            "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75))}


summary = {}
for kind in KINDS:
    sel = [r for r in rows if r["kind"] == kind]
    summary[kind] = {"ident_axis": IDENT_AXIS[kind],
                     "V_R2_add": agg(sel, "V_R2_add"),
                     "V_R2_row": agg(sel, "V_R2_row"), "V_R2_col": agg(sel, "V_R2_col"),
                     "W_H_row": agg(sel, "W_H_row"), "W_H_col": agg(sel, "W_H_col"),
                     "align_row_spearman": agg(sel, "align_row.spearman"),
                     "align_col_spearman": agg(sel, "align_col.spearman"),
                     "align_row_null_p95": agg(sel, "align_row.null_abs_p95"),
                     "align_col_null_p95": agg(sel, "align_col.null_abs_p95")}
by_step = {}
for st in steps:
    sel = [r for r in rows if r["step"] == st]
    by_step[str(st)] = {"align_row_spearman": agg(sel, "align_row.spearman"),
                        "align_col_spearman": agg(sel, "align_col.spearman")}

OUT.write_text(json.dumps({"schema": "p5-tier1-V-vs-W-axes-v1", "script_sha256_16": SHA,
                           "steps": steps, "n_rows": len(rows),
                           "note": "V and W share the same index set (G = delta x^T); "
                                   "a_i / b_j are the row/column factors of log V, "
                                   "h_row/h_col the median-centred ln RMS fields of W",
                           "summary": summary, "by_step": by_step, "rows": rows}, indent=1))
print("\nwrote", OUT.relative_to(ROOT))
print(f"\n{'kind':<11s}{'ident':>6s} | {'V R2row':>8s}{'V R2col':>8s} | {'W H_row':>8s}{'W H_col':>8s} "
      f"| {'corr(a,hrow)':>13s}{'null':>7s} | {'corr(b,hcol)':>13s}{'null':>7s}")
for kind in KINDS:
    S = summary[kind]
    print(f"{kind:<11s}{S['ident_axis']:>6s} | {S['V_R2_row']['median']:>8.3f}{S['V_R2_col']['median']:>8.3f} "
          f"| {S['W_H_row']['median']:>8.3f}{S['W_H_col']['median']:>8.3f} "
          f"| {S['align_row_spearman']['median']:>+13.3f}{S['align_row_null_p95']['median']:>7.3f} "
          f"| {S['align_col_spearman']['median']:>+13.3f}{S['align_col_null_p95']['median']:>7.3f}")
print("\n按步（全 42 矩阵中位）：")
for st in steps:
    B = by_step[str(st)]
    print(f"  step {st:>6d}  corr(a,h_row) {B['align_row_spearman']['median']:+.3f}   "
          f"corr(b,h_col) {B['align_col_spearman']['median']:+.3f}")
