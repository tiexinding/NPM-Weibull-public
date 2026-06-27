"""Per-layer Weibull scale lambda vs depth — clean(rho0) vs corrupted(rho0.6), Pythia vs Llama.
lambda_module = rms/sqrt(Gamma(1+2/k)). Tier-2 unify to p3_style (B2 6-22; was F149/bold 6-19):
p3_style, unified 4-module palette, no bold; arch name = non-bold identity. 守 #19/#76/#47.
"""
import json, numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from math import gamma
from matplotlib.lines import Line2D
from p3_style import apply_style, PALETTE
apply_style()

def mtype(n):
    if "h_to_4h" in n or "up_proj" in n: return "FFN-up"
    if "4h_to_h" in n or "down_proj" in n: return "FFN-down"
    if "gate_proj" in n: return "FFN-gate"
    if "dense.weight" in n or "o_proj" in n: return "Attn-out"
    return None
def last(a, t):
    L = open(f"_tmp_arch/scratch_{a}_rho{t}_s0/v1b_perlayer_budget.jsonl").read().strip().split("\n")
    return json.loads(L[-1])["layers"]
def lam_by_layer(a, t):
    d = last(a, t); out = {}
    for nm, v in d.items():
        mt = mtype(nm)
        if mt is None: continue
        out.setdefault(mt, {})[v["layer"]] = v["rms"] / np.sqrt(gamma(1 + 2/v["k80"]))
    return out

order = ["Attn-out", "FFN-gate", "FFN-up", "FFN-down"]
# unified module palette (Attn-out blue matches W_o in perlayer_robust; FFN green/orange; gate purple)
COL = {"Attn-out": PALETTE["ARCH_A"], "FFN-gate": "#7e57c2", "FFN-up": "#2a9d5a", "FFN-down": PALETTE["ARCH_B"]}
MK = {"Attn-out": "o", "FFN-gate": "s", "FFN-up": "^", "FFN-down": "v"}
ARCH = [("pythia", "Pythia (GELU FFN)"), ("llama", "Llama (SwiGLU FFN)")]

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), sharey=True)
for ax, (a, lab) in zip(axes, ARCH):
    clean = lam_by_layer(a, "000"); corr = lam_by_layer(a, "060")
    for mt in order:
        if mt not in clean: continue
        ly = sorted(clean[mt])
        ax.plot(ly, [clean[mt][l]*1e2 for l in ly], "-", color=COL[mt], marker=MK[mt], ms=5, lw=1.6)
        ax.plot(ly, [corr[mt][l]*1e2 for l in ly], "--", color=COL[mt], marker=MK[mt], ms=4, lw=1.1, alpha=0.55)
    ax.set_xlabel("block index"); ax.set_title(lab, fontsize=9); ax.set_xticks(range(6)); ax.grid(alpha=0.3)
axes[0].set_ylabel(r"per-layer Weibull scale $\lambda$  ($\times10^{-2}$)")
axes[0].set_ylim(bottom=1.0)   # headroom so the dipping Attn-out line clears the legend (老丁 6-22)
mod_handles = [Line2D([0],[0], color=COL[mt], marker=MK[mt], lw=1.6, label=mt) for mt in order]
axes[0].legend(handles=mod_handles, fontsize=7.5, loc="lower left", title="module", title_fontsize=8)
axes[1].legend(handles=[Line2D([0],[0], color=PALETTE["NEUTRAL"], ls="-", lw=1.6, label=r"clean ($\rho{=}0$)"),
                        Line2D([0],[0], color=PALETTE["NEUTRAL"], ls="--", lw=1.1, label=r"corrupted ($\rho{=}0.6$)")],
               fontsize=7.5, loc="lower left")
for ext in ("pdf", "png"):
    fig.savefig(f"03_paper/figures/p3_arch_perlayer_lambda.{ext}", bbox_inches="tight")
print("WROTE p3_arch_perlayer_lambda")
