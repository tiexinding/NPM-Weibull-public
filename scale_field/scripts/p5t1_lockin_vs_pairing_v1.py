#!/usr/bin/env python3
"""V̂ 与 W 的「锁定」与「配对涌现」孰先孰后（零 GPU，260914）。

动机：`v.row~o.col` 的 V̂ 配对在 800→3,200 步从 .14 跳到 .68；
第三批测到「V̂ 锁定领先权重场 1,600–2,200 步」。二者是不是同一件事？
本脚本把两个问题放在**同一来源、同一时间网格**上（dump_early 同时存 W 与 V̂）：

  锁定 lock-in(t) = spearman(X_t, X_T)，T = 最后一个 dump 步；X ∈ {V̂ 因子, W 场}
  配对 pairing(t) = spearman(左侧 profile_t, 右侧 profile_t)，V̂ 侧与 W 侧各一条

⚠️ 第三批那个「1,600–2,200 步」来自另一条 run（新卡 R0，150 点 logger），
   本脚本只能在自己的 5 点网格上回答**定性先后**，不复现该数值。
输出 08_paper5_draft/data/p5t1_lockin_vs_pairing_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "10_p5batch_cloud/results/dump_early/tensors"
OUT = ROOT / "08_paper5_draft/data/p5t1_lockin_vs_pairing_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
NLAYER = 6
KINDS7 = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "o_proj", "down_proj"]
IDENT = {"q_proj": "row", "k_proj": "row", "v_proj": "row", "gate_proj": "row",
         "up_proj": "row", "o_proj": "col", "down_proj": "col"}
# 路径：(名称, 左组件+侧, 右组件+侧)
PATHS = [("VO",  ("v_proj", "row"),  ("o_proj", "col")),
         ("UD",  ("up_proj", "row"), ("down_proj", "col")),
         ("GD",  ("gate_proj", "row"), ("down_proj", "col")),
         ("QK",  ("q_proj", "row"),  ("k_proj", "row"))]

steps = sorted(int(re.search(r"step(\d+)", f.name).group(1)) for f in DUMP.glob("step*.pt"))
assert len(steps) >= 3, steps
T = max(steps)

V, W = {}, {}
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
        Wm = d[base + "|W"].double().numpy(); Vm = d[base + "|Vh"].double().numpy()
        assert Wm.shape == Vm.shape
        LV = np.log(Vm + np.finfo(np.float64).tiny); g = LV.mean()
        V[(st, L, kind, "row")] = LV.mean(1) - g
        V[(st, L, kind, "col")] = LV.mean(0) - g
        hr = np.log(np.sqrt((Wm ** 2).mean(1))); hc = np.log(np.sqrt((Wm ** 2).mean(0)))
        W[(st, L, kind, "row")] = hr - np.median(hr)
        W[(st, L, kind, "col")] = hc - np.median(hc)
    print("loaded", st, flush=True)
assert len(V) == len(steps) * NLAYER * 7 * 2, len(V)

sp = lambda a, b: float(spearmanr(a, b).statistic)

# ---- 锁定：与末步的相关 ----
lockin = {"V": {}, "W": {}}
for store, name in ((V, "V"), (W, "W")):
    for kind in KINDS7:
        side = IDENT[kind]
        lockin[name][kind] = {str(st): float(np.median(
            [sp(store[(st, L, kind, side)], store[(T, L, kind, side)]) for L in range(NLAYER)]))
            for st in steps}

# ---- 配对涌现 ----
pairing = {"V": {}, "W": {}}
for store, name in ((V, "V"), (W, "W")):
    for pname, (k1, s1), (k2, s2) in PATHS:
        pairing[name][pname] = {str(st): float(np.median(
            [sp(store[(st, L, k1, s1)], store[(st, L, k2, s2)]) for L in range(NLAYER)]))
            for st in steps}

# ---- 汇总：到达阈值的插值步位（log 线性插值）----
def cross(series, thr):
    xs = [int(k) for k in series]; ys = [series[str(x)] for x in xs]
    for i in range(1, len(xs)):
        if ys[i - 1] < thr <= ys[i]:
            f = (thr - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(np.exp(np.log(xs[i - 1]) + f * (np.log(xs[i]) - np.log(xs[i - 1]))))
    return None if (not ys or ys[-1] < thr) else float(xs[0])

THR = [0.5, 0.7, 0.9]
cross_tbl = {"lockin": {}, "pairing": {}}
for name in ("V", "W"):
    cross_tbl["lockin"][name] = {k: {str(t): cross(lockin[name][k], t) for t in THR} for k in KINDS7}
    cross_tbl["pairing"][name] = {p[0]: {str(t): cross(pairing[name][p[0]], t) for t in THR} for p in PATHS}

OUT.write_text(json.dumps({"schema": "p5-tier1-lockin-vs-pairing-v1", "script_sha256_16": SHA,
                           "steps": steps, "reference_step": T, "thresholds": THR,
                           "caveat": "lock-in is measured against the last dump step (10,000), not a "
                                     "30k terminal state; the batch-3 '1,600-2,200 step lead' comes "
                                     "from a different run with a 150-point logger and is not reproduced here",
                           "lockin": lockin, "pairing": pairing, "crossings": cross_tbl}, indent=1))
print("\nwrote", OUT.relative_to(ROOT))

print(f"\n=== 锁定 lock-in(t) = corr(X_t, X_{T})，身份轴，六层中位 ===")
print(f"{'kind':<11s}{'axis':>5s} | " + "".join(f"{s:>8d}" for s in steps) + "   (上 V̂ / 下 W)")
for kind in KINDS7:
    print(f"{kind:<11s}{IDENT[kind]:>5s} | " + "".join(f"{lockin['V'][kind][str(s)]:>8.3f}" for s in steps) + "   V̂")
    print(f"{'':<11s}{'':>5s} | " + "".join(f"{lockin['W'][kind][str(s)]:>8.3f}" for s in steps) + "   W")
print(f"\n=== 配对涌现 pairing(t)，六层中位 ===")
print(f"{'path':<6s} | " + "".join(f"{s:>8d}" for s in steps))
for pname, a, b in PATHS:
    print(f"{pname:<6s} | " + "".join(f"{pairing['V'][pname][str(s)]:>8.3f}" for s in steps) + "   V̂")
    print(f"{'':<6s} | " + "".join(f"{pairing['W'][pname][str(s)]:>8.3f}" for s in steps) + "   W")
print(f"\n=== 达阈步位（log 插值；None = 到 {T} 步仍未达）===")
print(f"{'quantity':<18s}{'side':>5s}" + "".join(f"{'t@'+str(t):>10s}" for t in THR))
for pname, a, b in PATHS:
    for name in ("V", "W"):
        r = cross_tbl["pairing"][name][pname]
        print(f"{'pair '+pname:<18s}{name:>5s}" + "".join(
            f"{(f'{r[str(t)]:.0f}' if r[str(t)] else '—'):>10s}" for t in THR))
for kind in KINDS7:
    for name in ("V", "W"):
        r = cross_tbl["lockin"][name][kind]
        print(f"{'lock '+kind:<18s}{name:>5s}" + "".join(
            f"{(f'{r[str(t)]:.0f}' if r[str(t)] else '—'):>10s}" for t in THR))
