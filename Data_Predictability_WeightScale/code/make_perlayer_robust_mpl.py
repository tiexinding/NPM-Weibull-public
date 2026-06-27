#!/usr/bin/env python
"""§6.4 per-layer robustness figure — matplotlib redraw of p3_perlayer_robust (B2 6-22).
Data pipeline verbatim from make_perlayer_robust.py (numbers unchanged); rendering rebuilt:
matplotlib + p3_style, English, no in-figure title, (a)/(b) panels. 守 #19/#76/#47/#25(local).
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p3_style import apply_style, PALETTE

TAGS = ["000","010","020","030","040","050","060"]; RHO = [int(t)/100 for t in TAGS]
def short(b):
    L = int(b.split('.')[2]); t = "Wo" if "attention" in b else ("up" if "h_to_4h" in b else "dn"); return L, t
def perblock_dr2(tag):
    r = [json.loads(l) for l in open(f"_tmp_corrupt/corrupt_rho{tag}_s0/v1b_perlayer_budget.jsonl")]
    bo = sorted(r[0]["layers"].keys(), key=short)
    return {b: (r[-1]["layers"][b]["rms"]**2 - r[0]["layers"][b]["rms"]**2)*1e4 for b in bo}, bo
def pooled_dl2(tag):
    j = json.load(open(f"_tmp_corrupt/corrupt_rho{tag}_s0/lambda_trajectory_continue.json"))
    return (j[-1]["lambda"]**2 - j[0]["lambda"]**2)*1e4

data = {t: perblock_dr2(t)[0] for t in TAGS}; _, bo = perblock_dr2("000")
pooled = np.array([pooled_dl2(t) for t in TAGS])
COL = {"Wo": PALETTE["ARCH_A"], "up": "#2a9d5a", "dn": PALETTE["ARCH_B"]}
CNAME = {"Wo": r"$W_o$ (attention output)", "up": r"FFN up ($h{\to}4h$)", "dn": r"FFN down ($4h{\to}h$)"}

apply_style()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 3.5))
ndown = 0; seen = set()
for b in bo:
    L, t = short(b); ys = np.array([data[tg][b] for tg in TAGS]); ndown += int(ys[-1] < ys[0])
    show = t not in seen; seen.add(t)
    axA.plot(RHO, ys, color=COL[t], lw=1.2, alpha=0.55, label=CNAME[t] if show else None)
    axB.plot(RHO, ys/ys[0], color=COL[t], lw=1.0, alpha=0.4)
axB.plot(RHO, pooled/pooled[0], color="#111111", lw=3.0, marker="o", ms=5,
         label=r"pooled $\lambda^2-\lambda_0^2$ (model-level)", zorder=5)
axA.set_xlabel(r"corruption $\rho$"); axA.set_ylabel(r"$\Delta\,\mathrm{rms}^2\ (\times10^{-4})$, growth proxy")
axB.set_xlabel(r"corruption $\rho$"); axB.set_ylabel(r"relative growth (normalized at $\rho{=}0$)")
axA.text(0.03, 0.97, "(a)", transform=axA.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
axB.text(0.03, 0.97, "(b)", transform=axB.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
axA.text(0.5, 0.05, f"{ndown}/18 blocks decrease ($\\rho\\,0{{\\to}}0.6$)", transform=axA.transAxes,
         ha="center", va="bottom", fontsize=8)
axB.text(0.97, 0.95, "pooled = parameter-count\nweighted aggregate (FFN-leaning)\n-> sits within the bundle",
         transform=axB.transAxes, ha="right", va="top", fontsize=7.5,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
axA.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none")
axB.legend(loc="lower left", fontsize=7.5)
# headroom so in-panel annotations clear the curves (老丁 6-27): (a) extend bottom, (b) extend top
y0, y1 = axA.get_ylim(); axA.set_ylim(y0 - 0.20*(y1 - y0), y1)
y0, y1 = axB.get_ylim(); axB.set_ylim(y0, y1 + 0.16*(y1 - y0))
for o in ["03_paper/figures/p3_perlayer_robust.pdf", "03_paper/figures/p3_perlayer_robust.png"]:
    fig.savefig(o); print("WROTE", o)
kr = [json.loads(l) for l in open("_tmp_corrupt/corrupt_rho000_s0/v1b_perlayer_budget.jsonl")][-1]["layers"]
k80 = [v["k80"] for v in kr.values()]
print(f"  {ndown}/18 blocks decrease; k80 {min(k80):.2f}~{max(k80):.2f}; pooled {pooled[0]:.3f}->{pooled[-1]:.3f}")
