#!/usr/bin/env python3
"""2e 剩余项：gate 的权重场最早锁定，而其 δ 最晚锁定 —— 两者如何脱钩（零 GPU，260914）。

C.7.2：gate 的 δ 行 profile 到 11,619 步才锁定（七类最晚），但其权重场 4,828 步就锁定（七类最早）。
候选：权重场之所以早锁，是因为**驱动它分化的径向项早早塌掉**，此后 δ 再变也推不动场。

读数（logger，R0，150 窗口）：
  drive(t)   = sd over units of (−uapp_row_radial)   ← 场增长 = −radial（第三批符号约定）
               uapp_row_radial 本身已是「lr 加权的窗内有符号径向投影之和」
  |Δh|(t)    = sd over units of (h_row_start(t) − h_row_start(终态))   ← 场离终态还有多远（幅值，非排序）
  还报 drive 的中位（整体缩放）与 sd/中位（分化相对强度）
守则：先报原始量级，不用排序类读数代替。
输出 08_paper5_draft/data/p5t1_gate_decoupling_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "11_p5mech_cloud/results/R0_base_snap/logger"
OUT = ROOT / "08_paper5_draft/data/p5t1_gate_decoupling_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
NLAYER = 6
KINDS7 = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "o_proj", "down_proj"]
IDENT = {"q_proj": "row", "k_proj": "row", "v_proj": "row", "gate_proj": "row",
         "up_proj": "row", "o_proj": "col", "down_proj": "col"}
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
pre = lambda L, k: f"model.layers.{L}." + ("self_attn." if k in ATTN else "mlp.") + f"{k}.weight"

files = sorted(LOG.glob("win*.npz"), key=lambda p: int(re.search(r"win(\d+)", p.name).group(1)))
assert len(files) >= 100, len(files)
D, H, ts = {}, {}, []
for f in files:
    z = np.load(f, allow_pickle=True)
    t0 = int(np.asarray(z["__meta__|t_start"]).item()); ts.append(t0)
    for kind in KINDS7:
        side = IDENT[kind]
        for L in range(NLAYER):
            D[(t0, L, kind)] = -np.asarray(z[f"{pre(L,kind)}|uapp_{side}_radial"], float)  # 场增长方向
            H[(t0, L, kind)] = np.asarray(z[f"{pre(L,kind)}|h_{side}_start"], float)
ts = sorted(set(ts)); T = ts[-1]
assert len(ts) == len(files)

rows = {}
for kind in KINDS7:
    rows[kind] = {"drive_sd": {}, "drive_med_abs": {}, "resid_sd": {}}
    for t in ts:
        dv = [D[(t, L, kind)] for L in range(NLAYER)]
        rows[kind]["drive_sd"][str(t)] = float(np.median([np.std(v) for v in dv]))
        rows[kind]["drive_med_abs"][str(t)] = float(np.median([np.median(np.abs(v)) for v in dv]))
        rows[kind]["resid_sd"][str(t)] = float(np.median(
            [np.std(H[(t, L, kind)] - H[(T, L, kind)]) for L in range(NLAYER)]))


def half_life(series, frac=0.5):
    """drive_sd 从峰值衰减到 frac×峰 的步位。"""
    xs = [int(k) for k in series]; ys = np.array([series[str(x)] for x in xs])
    ip = int(np.argmax(ys)); peak = ys[ip]
    for i in range(ip + 1, len(xs)):
        if ys[i] <= frac * peak:
            f = (frac * peak - ys[i - 1]) / (ys[i] - ys[i - 1]) if ys[i] != ys[i - 1] else 0.0
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1])), float(xs[ip]), float(peak)
    return None, float(xs[ip]), float(peak)


summary = {}
for kind in KINDS7:
    h50, tpk, pk = half_life(rows[kind]["drive_sd"], 0.5)
    h10, _, _ = half_life(rows[kind]["drive_sd"], 0.1)
    summary[kind] = {"drive_peak_step": tpk, "drive_peak": pk,
                     "drive_half_step": h50, "drive_tenth_step": h10,
                     "resid_sd_at_peak": rows[kind]["resid_sd"][str(int(tpk))],
                     "resid_sd_final": rows[kind]["resid_sd"][str(T)]}

OUT.write_text(json.dumps({"schema": "p5-tier1-gate-decoupling-v1", "script_sha256_16": SHA,
                           "run": "11_p5mech_cloud/results/R0_base_snap", "steps": ts,
                           "reference_step": T, "note": "drive = sd over units of (-uapp_radial), the "
                           "lr-weighted signed radial projection of the applied update summed over the "
                           "20-step window; field growth = -radial (batch-3 sign convention)",
                           "summary": summary, "series": rows}, indent=1))
print("wrote", OUT.relative_to(ROOT))

print(f"\n=== 分化驱动 drive = sd_units(−uapp_radial)，六层中位 ===")
idx = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 149]
print(f"{'kind':<11s}" + "".join(f"{ts[i]:>9d}" for i in idx))
for kind in KINDS7:
    print(f"{kind:<11s}" + "".join(f"{rows[kind]['drive_sd'][str(ts[i])]:>9.2e}" for i in idx)
          + ("   <- gate" if kind == "gate_proj" else ""))
print(f"\n{'kind':<11s}{'峰步':>8s}{'峰值':>11s}{'半衰步':>9s}{'降到1/10':>10s}   (drive)")
for kind in KINDS7:
    S = summary[kind]
    h50 = f"{S['drive_half_step']:.0f}" if S['drive_half_step'] else "—"
    h10 = f"{S['drive_tenth_step']:.0f}" if S['drive_tenth_step'] else "—"
    print(f"{kind:<11s}{S['drive_peak_step']:>8.0f}{S['drive_peak']:>11.2e}{h50:>9s}{h10:>10s}"
          + ("   <- gate" if kind == "gate_proj" else ""))
print(f"\n=== 场离终态的距离 sd_units(h_t − h_终态)（幅值）===")
print(f"{'kind':<11s}" + "".join(f"{ts[i]:>9d}" for i in idx))
for kind in KINDS7:
    print(f"{kind:<11s}" + "".join(f"{rows[kind]['resid_sd'][str(ts[i])]:>9.3f}" for i in idx)
          + ("   <- gate" if kind == "gate_proj" else ""))
