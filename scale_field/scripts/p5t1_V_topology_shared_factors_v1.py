#!/usr/bin/env python3
"""二阶矩的行/列因子是否服从架构拓扑（零 GPU，260914）。

老丁框架的强预言：若 b^V_j ← E[x_j²]、a^V_i ← E[δ_i²]，则**架构上共享同一通道空间的因子必须相同**。
  P1 同层 q/k/v 读同一残差流  ⇒ b^q ≈ b^k ≈ b^v   （最硬：同一 x）
  P2 o 的输入列 = v 的输出行（head/value 通道） ⇒ b^o ~ a^v
  P3 down 的输入列 = up/gate 的输出行（FFN 隐单元） ⇒ b^down ~ a^up, a^gate
  P4 gate/up 读同一残差流 ⇒ b^gate ≈ b^up
对照：① 层内置换 null；② 跨层错位（同一对但取相邻层），检验不是普遍性趋势。
同时报权重侧的同一对比较，检验「V 侧共享」是否也出现在 W 侧。
输出 08_paper5_draft/data/p5t1_V_topology_shared_factors_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "10_p5batch_cloud/results/dump_early/tensors"
OUT = ROOT / "08_paper5_draft/data/p5t1_V_topology_shared_factors_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
NLAYER, N_NULL = 6, 200
RNG = np.random.default_rng(0)
# (名称, 左组件与取哪侧, 右组件与取哪侧, 预言依据)
PAIRS = [
    ("P1 q.col ~ k.col",     ("q_proj", "col"),  ("k_proj", "col"),  "same residual input"),
    ("P1 q.col ~ v.col",     ("q_proj", "col"),  ("v_proj", "col"),  "same residual input"),
    ("P1 k.col ~ v.col",     ("k_proj", "col"),  ("v_proj", "col"),  "same residual input"),
    ("P4 gate.col ~ up.col", ("gate_proj", "col"), ("up_proj", "col"), "same residual input"),
    ("P2 o.col ~ v.row",     ("o_proj", "col"),  ("v_proj", "row"),  "head/value channel"),
    ("P3 down.col ~ up.row", ("down_proj", "col"), ("up_proj", "row"), "FFN hidden unit"),
    ("P3 down.col ~ gate.row", ("down_proj", "col"), ("gate_proj", "row"), "FFN hidden unit"),
]
CTRL = [
    ("ctrl q.col ~ o.col",   ("q_proj", "col"),  ("o_proj", "col"),  "residual vs head channel"),
    ("ctrl v.row ~ q.col",   ("v_proj", "row"),  ("q_proj", "col"),  "head channel vs residual"),
]

steps = sorted(int(re.search(r"step(\d+)", f.name).group(1)) for f in DUMP.glob("step*.pt"))
assert steps, f"缺 {DUMP}"


def factors(V, W):
    LV = np.log(V + np.finfo(np.float64).tiny); g = LV.mean()
    hr = np.log(np.sqrt((W ** 2).mean(1))); hc = np.log(np.sqrt((W ** 2).mean(0)))
    return {"V_row": LV.mean(1) - g, "V_col": LV.mean(0) - g,
            "W_row": hr - np.median(hr), "W_col": hc - np.median(hc)}


F = {}
for st in steps:
    d = torch.load(DUMP / f"step{st}.pt", map_location="cpu", weights_only=False)
    assert d["__meta__"]["step"] == st
    for key in d:
        if key == "__meta__" or not key.endswith("|W"):
            continue
        base = key[:-2]
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", base)
        assert m, base
        W = d[base + "|W"].double().numpy(); V = d[base + "|Vh"].double().numpy()
        assert W.shape == V.shape
        F[(st, int(m.group(1)), m.group(2))] = factors(V, W)
    print("loaded step", st, flush=True)
assert len(F) == len(steps) * NLAYER * 7, len(F)


def cmp_pair(st, L, left, right, side_key):
    (k1, s1), (k2, s2) = left, right
    u = F[(st, L, k1)][f"{side_key}_{s1}"]; w = F[(st, L, k2)][f"{side_key}_{s2}"]
    if u.shape != w.shape:
        return None
    sp = float(spearmanr(u, w).statistic)
    nul = np.array([abs(spearmanr(RNG.permutation(u), w).statistic) for _ in range(N_NULL)])
    Ln = (L + 1) % NLAYER
    wm = F[(st, Ln, k2)][f"{side_key}_{s2}"]
    mism = float(spearmanr(u, wm).statistic) if u.shape == wm.shape else None
    return {"spearman": sp, "null_abs_p95": float(np.quantile(nul, .95)), "mismatch_next_layer": mism}


res = {}
for name, left, right, why in PAIRS + CTRL:
    res[name] = {"why": why, "V": {}, "W": {}}
    for side_key, store in (("V", "V"), ("W", "W")):
        vals, nulls, mis = [], [], []
        for st in steps:
            for L in range(NLAYER):
                r = cmp_pair(st, L, left, right, side_key)
                assert r is not None, (name, side_key)
                vals.append(r["spearman"]); nulls.append(r["null_abs_p95"])
                if r["mismatch_next_layer"] is not None:
                    mis.append(r["mismatch_next_layer"])
        res[name][store] = {"median": float(np.median(vals)), "q25": float(np.quantile(vals, .25)),
                            "q75": float(np.quantile(vals, .75)), "min": float(np.min(vals)),
                            "null_abs_p95_med": float(np.median(nulls)),
                            "mismatch_next_layer_med": float(np.median(mis)) if mis else None,
                            "n": len(vals)}

OUT.write_text(json.dumps({"schema": "p5-tier1-V-topology-shared-factors-v1",
                           "script_sha256_16": SHA, "steps": steps,
                           "n_matrices_per_step": NLAYER * 7, "result": res}, indent=1))
print("\nwrote", OUT.relative_to(ROOT))
print(f"\n{'pair':<24s}{'why':<26s}| {'V: med':>8s}{'[q25,q75]':>16s}{'null':>7s}{'mism':>7s} "
      f"| {'W: med':>8s}{'null':>7s}{'mism':>7s}")
for name in res:
    V, W = res[name]["V"], res[name]["W"]
    print(f"{name:<24s}{res[name]['why']:<26s}| {V['median']:>+8.3f}"
          f"{'[%+.2f,%+.2f]' % (V['q25'], V['q75']):>16s}{V['null_abs_p95_med']:>7.3f}"
          f"{V['mismatch_next_layer_med']:>+7.2f} | {W['median']:>+8.3f}"
          f"{W['null_abs_p95_med']:>7.3f}{W['mismatch_next_layer_med']:>+7.2f}")
