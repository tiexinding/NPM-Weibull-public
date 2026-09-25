#!/usr/bin/env python3
"""一阶矩的零参数探针：⟨M̂,G⟩/‖G‖² 恒等式（零 GPU，260914）。

精确展开（无假设，bias correction 已在内）：
    ⟨M̂_t,G_t⟩/‖G_t‖² = (1−β₁)/(1−β₁^t) · [ 1 + Σ_{s≥1} β₁^s ⟨G_{t−s},G_t⟩/‖G_t‖² ]
⇒ 与 (1−β₁)/bc₁ 的**相对偏差就是 β₁ 加权的梯度相关项** S = Σ_{s≥1} β₁^s ρ_s。
每 lag 平均 = S / Σ_{s≥1}β₁^s = S·(1−β₁)/β₁。
各向同性噪声底 = 1/√d（d = 矩阵坐标数），用于判断是否可分辨于零。
同时记录 cos(M̂,G) 与 ‖M̂‖/‖G‖ 的实测与 t→∞ 闭式，以及有限 t 修正的大小。
输出 08_paper5_draft/data/p5t1_M_identity_probe_v1.json
"""
import hashlib, json, re
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "10_p5batch_cloud/results/dump_early/tensors"
OUT = ROOT / "08_paper5_draft/data/p5t1_M_identity_probe_v1.json"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

steps = sorted(int(re.search(r"step(\d+)", f.name).group(1)) for f in DUMP.glob("step*.pt"))
assert steps, f"缺 {DUMP}"
rows, betas = [], set()
for st in steps:
    d = torch.load(DUMP / f"step{st}.pt", map_location="cpu", weights_only=False)
    meta = d["__meta__"]; assert meta["step"] == st
    b1 = meta["beta1"]; betas.add(b1)
    bc1_meta = meta["bc1"]; bc1 = 1.0 - b1 ** st
    assert abs(bc1 - bc1_meta) < 1e-12, (bc1, bc1_meta)
    gnorm = []
    for key in d:
        if key == "__meta__" or not key.endswith("|G"):
            continue
        base = key[:-2]
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", base)
        assert m, base
        G = d[base + "|G"].double(); M = d[base + "|Mh"].double()
        assert G.shape == M.shape
        g2 = float((G * G).sum())
        prod = float((M * G).sum()) / g2                      # = cos · (‖M̂‖/‖G‖)
        target = (1 - b1) / bc1
        S = prod / target - 1.0                               # = Σ_{s≥1} β₁^s ρ_s
        rows.append({"step": st, "layer": int(m.group(1)), "kind": m.group(2),
                     "d": int(np.prod(G.shape)),
                     "prod": prod, "target": target, "S": S,
                     "per_lag": S * (1 - b1) / b1,
                     "noise_floor": 1.0 / np.sqrt(np.prod(G.shape)),
                     "cos": float((M * G).sum() / (M.norm() * G.norm())),
                     "norm_ratio": float(M.norm() / G.norm()),
                     "g_norm": float(G.norm())})
        gnorm.append(float(G.norm()))
    print(f"step {st:>6d}  n={len(gnorm)}  ‖G‖ med {np.median(gnorm):.4e}  bc1={bc1:.12f}", flush=True)
assert len(betas) == 1, betas
B1 = betas.pop()
assert len(rows) == len(steps) * 42, len(rows)

pred_cos = float(np.sqrt(1 - B1 ** 2))
pred_nr = float(np.sqrt((1 - B1) / (1 + B1)))


def q(sel, k):
    v = np.array([r[k] for r in sel], float)
    return {"median": float(np.median(v)), "q25": float(np.quantile(v, .25)),
            "q75": float(np.quantile(v, .75)), "n": len(v)}


by_step = {}
for st in steps:
    sel = [r for r in rows if r["step"] == st]
    by_step[str(st)] = {
        "prod": q(sel, "prod"), "S": q(sel, "S"), "per_lag": q(sel, "per_lag"),
        "cos": q(sel, "cos"), "norm_ratio": q(sel, "norm_ratio"),
        "g_norm": q(sel, "g_norm"),
        "target": sel[0]["target"], "finite_t_1_minus_bc1": float(B1 ** st),
        "noise_floor_med": float(np.median([r["noise_floor"] for r in sel])),
        "per_lag_in_sigma": float(np.median([r["per_lag"] / r["noise_floor"] for r in sel]))}

OUT.write_text(json.dumps({
    "schema": "p5-tier1-M-identity-probe-v1", "script_sha256_16": SHA,
    "beta1": B1, "steps": steps, "n_matrices_per_step": 42,
    "closed_form_t_to_inf": {"cos": pred_cos, "norm_ratio": pred_nr,
                             "rms_compression": 1.0 / pred_nr,
                             "note": "no fitted parameters; beta1 read from optimizer config"},
    "sum_beta1_powers_s_ge_1": B1 / (1 - B1),
    "by_step": by_step, "rows": rows}, indent=1))
print("\nwrote", OUT.relative_to(ROOT))
print(f"\nβ₁={B1}  闭式 cos={pred_cos:.4f}  ‖M̂‖/‖G‖={pred_nr:.4f}  压缩 {1/pred_nr:.2f}×  "
      f"Σ_{{s≥1}}β₁^s={B1/(1-B1):.1f}")
print(f"\n{'step':>7s}{'⟨M̂,G⟩/‖G‖²':>13s}{'解析':>9s}{'偏差%':>8s}{'S':>9s}{'每lag':>10s}{'σ':>7s}{'1−bc₁':>11s}")
for st in steps:
    B = by_step[str(st)]
    print(f"{st:>7d}{B['prod']['median']:>13.5f}{B['target']:>9.5f}"
          f"{100*(B['prod']['median']/B['target']-1):>8.2f}{B['S']['median']:>9.4f}"
          f"{B['per_lag']['median']:>10.5f}{B['per_lag_in_sigma']:>7.1f}{B['finite_t_1_minus_bc1']:>11.2e}")
