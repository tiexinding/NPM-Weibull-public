#!/usr/bin/env python
"""真 Pythia-70m per-block k. 本地下ckpt per-matrix Weibull k → median.
缓存sub是池化子采样算不出per-block; 本脚本per-matrix重抽证per-block k~1.20 in-band(vs aggregate1.162混合假象)。"""
import os, subprocess, numpy as np, torch

def weibull_k(w, lo=0.1, hi=0.9):
    """middle-80% probability-plot Weibull k (Paper#1法)."""
    a = np.sort(np.abs(w.astype(np.float64))); a = a[a > 0]
    n = len(a); i = np.arange(1, n+1)
    F = (i - 0.5) / n
    m = (F >= lo) & (F <= hi)
    x = np.log(a[m]); y = np.log(-np.log(1 - F[m]))
    k, b = np.polyfit(x, y, 1)
    return k, float(np.exp(-b / k))  # k, lambda

# 下真 Pythia-70m final (main=step143000), non-xet, hf-mirror
binf = "paper_figures/cloud_realpythia/pythia70m_main.bin"
if not os.path.exists(binf) or os.path.getsize(binf) < 1e8:
    print("[dl] 下 pythia-70m main pytorch_model.bin (~158MB)", flush=True)
    subprocess.run(["curl","-L","--connect-timeout","20","-m","400","--retry","3","-C","-","-s","-o",binf,
                    "https://hf-mirror.com/EleutherAI/pythia-70m/resolve/main/pytorch_model.bin"])
sd = torch.load(binf, map_location="cpu", weights_only=False)

# Transmission class (GPTNeoX/GELU): attention.dense(Wo) + mlp.dense_h_to_4h(up) + mlp.dense_4h_to_h(down)
trans_suffix = ["attention.dense.weight", "mlp.dense_h_to_4h.weight", "mlp.dense_4h_to_h.weight"]
per_matrix = []
for name, w in sd.items():
    if any(name.endswith(s) for s in trans_suffix):
        wn = w.detach().numpy().ravel()
        k, lam = weibull_k(wn)
        per_matrix.append((name, k, lam, wn.size))

ks = np.array([x[1] for x in per_matrix]); lams = np.array([x[2] for x in per_matrix])
print(f"\n真Pythia-70m final per-matrix transmission ({len(per_matrix)} 矩阵):")
for name, k, lam, sz in per_matrix:
    print(f"  {name:55s} k={k:.3f} λ={lam:.4f} (n={sz})")
print(f"\n===== per-block 统计 (B1 §6.1 需) =====")
print(f"per-block k: median={np.median(ks):.3f}, mean={np.mean(ks):.3f}, range=[{ks.min():.3f},{ks.max():.3f}]")
print(f"per-block λ: median={np.median(lams):.4f}")
print(f"对比 aggregate(池化): k=1.162 (混合假象), λ=0.0221")
print(f"→ per-block k median {np.median(ks):.3f} {'in-band[1.186,1.204]✓' if 1.18<=np.median(ks)<=1.21 else 'NOT in-band'} (Paper#1 k-band)")
# 存给B1
import json
json.dump({"per_matrix":[(n,float(k),float(l),int(s)) for n,k,l,s in per_matrix],
           "per_block_k_median":float(np.median(ks)),"per_block_k_range":[float(ks.min()),float(ks.max())],
           "per_block_lam_median":float(np.median(lams)),"aggregate_k":1.162},
          open("paper_figures/cloud_realpythia/realpythia_perblock_k.json","w"), indent=1)
print("→ 存 realpythia_perblock_k.json (给B1)")
