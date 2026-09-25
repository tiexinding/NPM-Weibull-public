#!/usr/bin/env python3
"""Tier-1 ②：AdamW 分母的结构分解 —— 四臂配对路径轨迹（零 GPU，260914）。

读数与 S13/S14 **逐字同一协议**（h = 身份轴 ln RMS 去中位；配对 QK_pair / VO / UD；
gain g = h1+h2, balance b = h1-h2; 增益比 = var(g)/var(b); null = 层内 200 次重配对）。
四臂在**同一脚本、同一聚合口径**下重算，只用本地 ckpt：
  base           05_ivn_runs/ivn_base
  vA_colflat     V 列压平（保留列因子，丢行因子与单元残差）
  vB_permg       V 全局置换（保留边缘分布，破坏列身份对齐）
  rowflat_rncol  V 行压平 + 列范数匹配（列信息退为单列标量）
输出 08_paper5_draft/data/p5t1_vstructure_trajectory_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08_paper5_draft/data/p5t1_vstructure_trajectory_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
N_NULL = 200
RNG = np.random.default_rng(0)
GROUPS = ["QK_pair", "VO", "UD"]
ARMS = {
    "base":          ROOT / "05_ivn_runs/ivn_base/ckpt",
    "vA_colflat":    ROOT / "10_p5batch_cloud/results/vA_colflat/ckpt",
    "vB_permg":      ROOT / "10_p5batch_cloud/results/vB_permg/ckpt",
    "rowflat_rncol": ROOT / "10_p5batch_cloud/results/rowflat_rncol/ckpt",
}
# 架构常数从 IVN 元数据取，不硬编码
IVN = json.load(open(ROOT / "05_ivn_runs/ivn_j1_j3_hw_v1.json"))
HEADS, HEAD_DIM = int(IVN["heads"]), int(IVN["head_dim"])
NLAYER = 6


def steps_of(d):
    return sorted(int(m.group(1)) for p in d.glob("step*.pt")
                  if (m := re.fullmatch(r"step(\d+)\.pt", p.name)))


def prof(x, axis):
    lr = np.log(np.sqrt((x ** 2).mean(axis))); return lr - np.median(lr)


def load_layers(p):
    st = torch.load(p, map_location="cpu", weights_only=True)
    st = st.get("model", st) if isinstance(st, dict) and "model" in st else st
    lay = {}
    for nm, w in st.items():
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", nm)
        if m:
            lay.setdefault(int(m.group(1)), {})[m.group(2)] = w.double().numpy()
    return lay


def pair_profile(h_row):
    half = HEAD_DIM // 2
    m = h_row.reshape(HEADS, HEAD_DIM)
    return np.median((m[:, :half] + m[:, half:]) / 2, 0)


def profiles(m):
    q, k = prof(m["q_proj"], 1), prof(m["k_proj"], 1)
    v, o = prof(m["v_proj"], 1), prof(m["o_proj"], 0)
    u, d = prof(m["up_proj"], 1), prof(m["down_proj"], 0)
    assert q.shape == k.shape == v.shape == o.shape and u.shape == d.shape
    return {"QK_pair": (pair_profile(q), pair_profile(k)), "VO": (v, o), "UD": (u, d)}


def gb(h1, h2):
    g, b = h1 + h2, h1 - h2
    return {"var_g": float(g.var()), "var_b": float(b.var()),
            "ratio": float(g.var() / b.var()), "corr": float(np.corrcoef(h1, h2)[0, 1])}


def null_ratio(h1, h2):
    r = np.array([float((h1 + RNG.permutation(h2)).var() / (h1 - RNG.permutation(h2)).var())
                  for _ in range(N_NULL)])
    return float(np.quantile(r, 0.975))


avail = {a: steps_of(d) for a, d in ARMS.items()}
for a, d in ARMS.items():
    assert d.is_dir(), f"缺目录 {d}"
    assert avail[a], f"{a} 无 ckpt"
common = sorted(set(avail["base"]) & set(avail["vA_colflat"]) & set(avail["vB_permg"]))
assert len(common) >= 5, f"base/vA/vB 公共步位太少: {common}"
print("base/vA/vB 公共步位", common, "| rowflat 可用", avail["rowflat_rncol"])

rows, nulls = [], {g: [] for g in GROUPS}
for arm, d in ARMS.items():
    for s in (common if arm != "rowflat_rncol" else sorted(set(avail[arm]) & set(common))):
        lay = load_layers(d / f"step{s}.pt")
        assert sorted(lay) == list(range(NLAYER)), (arm, s, sorted(lay))
        P = {L: profiles(lay[L]) for L in range(NLAYER)}
        for L in range(NLAYER):
            e = {"arm": arm, "step": s, "layer": L}
            for g in GROUPS:
                h1, h2 = P[L][g]
                e[g] = gb(h1, h2)
                if arm == "base" and s == max(common):
                    nulls[g].append(null_ratio(h1, h2))
            rows.append(e)
        print(arm, s, "done", flush=True)

for g in GROUPS:
    assert len(nulls[g]) == NLAYER, (g, len(nulls[g]))
n_expect = len(ARMS) * len(common) * NLAYER - (len(common) - len(set(avail["rowflat_rncol"]) & set(common))) * NLAYER
assert len(rows) == n_expect, (len(rows), n_expect)


def summary():
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for g in GROUPS:
            out[arm][g] = {}
            for s in sorted({r["step"] for r in rows if r["arm"] == arm}):
                sel = [r[g] for r in rows if r["arm"] == arm and r["step"] == s]
                out[arm][g][str(s)] = {
                    "corr_med": float(np.median([x["corr"] for x in sel])),
                    "ratio_med": float(np.median([x["ratio"] for x in sel])),
                    "corr_per_layer": [x["corr"] for x in sel],
                    "ratio_per_layer": [x["ratio"] for x in sel]}
    return out


res = {"schema": "p5-tier1-vstructure-trajectory-v1", "script_sha256_16": SHA,
       "protocol": "h = identity-axis ln RMS minus its median on raw W; g=h1+h2, b=h1-h2; "
                   "ratio=var(g)/var(b); null = 200 within-layer re-pairings (base at last common step)",
       "arms": {a: str(p.relative_to(ROOT)) for a, p in ARMS.items()},
       "steps_available": avail, "steps_common": common,
       "heads": HEADS, "head_dim": HEAD_DIM, "n_layers": NLAYER,
       "null_ratio_q975_med": {g: float(np.median(nulls[g])) for g in GROUPS},
       "summary": summary(), "rows": rows}
OUT.write_text(json.dumps(res, indent=1))
print("\nwrote", OUT.relative_to(ROOT))
for g in GROUPS:
    print(f"\n-- {g}  (null ratio q97.5 med = {res['null_ratio_q975_med'][g]:.2f})")
    for arm in ARMS:
        S = res["summary"][arm][g]
        print(f"   {arm:<14s} " + "  ".join(f"{k}:{v['corr_med']:+.2f}/{v['ratio_med']:.1f}" for k, v in S.items()))
