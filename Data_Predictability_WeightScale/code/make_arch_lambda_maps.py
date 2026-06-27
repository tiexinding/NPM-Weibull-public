"""lambda 2D map (layer x module) per rho — the "raw material" of C1 (its slope along rho).
lambda_module(layer) = rms / sqrt(Gamma(1+2/k80)). Tier-2 unify to p3_style (B2 6-22; was F149 6-21):
p3_style, NO per-cell numbers (legibility at paper width — color carries it), no bold, viridis
(lambda is a positive magnitude). Vector PDF + PNG. 守 #19/#76/#47.
"""
import json, numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from math import gamma
from p3_style import apply_style
apply_style()

V = 50304; Hr = 8.0; P = 0.59
TAGS = ["000","010","020","030","040","050","060"]; RHO = [0.0,0.1,0.2,0.3,0.4,0.5,0.6]
order = ["Attn-out","FFN-gate","FFN-up","FFN-down"]; LAYERS = list(range(6))
def mtype(n):
    if "h_to_4h" in n or "up_proj" in n: return "FFN-up"
    if "4h_to_h" in n or "down_proj" in n: return "FFN-down"
    if "gate_proj" in n: return "FFN-gate"
    if "dense.weight" in n or "o_proj" in n: return "Attn-out"
    return None
def budget(a, t):
    L = open(f"_tmp_arch/scratch_{a}_rho{t}_s0/v1b_perlayer_budget.jsonl").read().strip().split("\n")
    return json.loads(L[0])["layers"], json.loads(L[-1])["layers"]
def lam_grid(a, t):
    f, l = budget(a, t); M = np.full((len(order), len(LAYERS)), np.nan)
    for nm in l:
        mt = mtype(nm)
        if mt is None: continue
        M[order.index(mt), l[nm]["layer"]] = (l[nm]["rms"] / np.sqrt(gamma(1 + 2/l[nm]["k80"]))) * 1e2
    return M
ARCH = [("pythia", "Pythia (GELU FFN)"), ("llama", "Llama (SwiGLU FFN)")]
all_lam = np.concatenate([lam_grid(a, t)[~np.isnan(lam_grid(a, t))] for a, _ in ARCH for t in TAGS])
print(f"lambda(x1e2) range {all_lam.min():.2f}..{all_lam.max():.2f}")

nrow, ncol = len(ARCH), len(TAGS)
fig, axes = plt.subplots(nrow, ncol, figsize=(6.9, 2.7), gridspec_kw={"wspace": 0.10, "hspace": 0.22})
cmap = mpl.colormaps["viridis"].copy(); cmap.set_bad("0.90")
im = None
for ri, (a, lab) in enumerate(ARCH):
    for ci, t in enumerate(TAGS):
        ax = axes[ri, ci]; M = np.ma.masked_invalid(lam_grid(a, t))
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=1.75, vmax=2.45, origin="lower")
        ax.set_xticks(LAYERS); ax.set_yticks(range(len(order))); ax.grid(False)
        if ci == 0:
            ax.set_yticklabels(order, fontsize=6.5); ax.set_ylabel(lab.split(" ")[0], fontsize=9, labelpad=6)
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels([f"L{j}" for j in LAYERS], fontsize=5.5) if ri == nrow-1 else ax.set_xticklabels([])
        if ri == 0: ax.set_title(fr"$\rho{{=}}{RHO[ci]:.1f}$", fontsize=8)
        if ri == nrow-1: ax.set_xlabel("block", fontsize=7)
cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012)
cb.set_label(r"Weibull scale $\lambda$  ($\times10^{-2}$)", fontsize=8)
for ext in ("pdf", "png"):
    fig.savefig(f"03_paper/figures/p3_arch_lambda_maps.{ext}", bbox_inches="tight")
print("WROTE p3_arch_lambda_maps")
