#!/usr/bin/env python
"""§6.2 cross-seed figure — matplotlib/publication redraw of p3_crossseed (B2 6-22).
Data pipeline + bootstrap verbatim from make_crossseed.py (numbers unchanged); rendering rebuilt:
matplotlib + p3_style, English, no in-figure title, vector PDF. NOTATION FIX: slope k -> C_1
(README §0.5 rename). 守 #19/#76/#47/#25(local).
"""
import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p3_style import apply_style, PALETTE
V = 50304; Hr = 8.0; P = 0.59   # convex-law exponent (= 1/p), consistent with Fig 1 / the main law (老丁 6-27 fix: was linear)

def D_bi(t, n=2400000):
    t = t[:n].astype(np.int64); prev = t[:-1]; nxt = t[1:]
    def H(x): _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    return float(H(prev * V + nxt) - H(prev))
def Y2(f):
    j = json.load(open(f)); return (j[-1]["lambda"]**2 - j[0]["lambda"]**2) * 1e4

RHO = [(0.0, "000"), (0.2, "020"), (0.4, "040"), (0.6, "060")]
data = []
for r, tag in RHO:
    D = D_bi(np.load(f"tokens_corrupt/corrupt_rho{tag}_s0.npy")); ys = []
    for f in [f"_tmp_corrupt/corrupt_rho{tag}_s0/lambda_trajectory_continue.json",
              f"_tmp_grid/rho{tag}_e3e-4_s1/lambda_trajectory_continue.json",
              f"_tmp_grid/rho{tag}_e3e-4_s2/lambda_trajectory_continue.json"]:
        if os.path.exists(f): ys.append(Y2(f))
    data.append((r, D, ys))
Dv = np.array([d[1] for d in data]); ymean = np.array([np.mean(d[2]) for d in data])
ystd = np.array([np.std(d[2], ddof=1) if len(d[2]) > 1 else 0.0 for d in data]); nseed = [len(d[2]) for d in data]
X = (Hr - Dv)**P
k_m, C0_m = np.polyfit(X, ymean, 1)
R2_m = 1 - np.sum((ymean - (C0_m + k_m * X))**2) / np.sum((ymean - ymean.mean())**2)
Cs = []; ks = []
for b in range(2000):
    yb = np.array([d[2][(b * 7 + i * 13) % len(d[2])] for i, d in enumerate(data)])
    kb, Cb = np.polyfit(X, yb, 1); Cs.append(Cb); ks.append(kb)
Cs = np.array(Cs); ks = np.array(ks)
C0_lo, C0_hi = np.percentile(Cs, [2.5, 97.5]); k_lo, k_hi = np.percentile(ks, [2.5, 97.5])
seed_cv = np.mean(ystd / ymean) * 100

apply_style()
fig, ax = plt.subplots(figsize=(6.2, 4.2))
xr = np.linspace(Dv.min() - 0.05, Hr, 50)
ax.plot(xr, C0_m + k_m * (Hr - xr)**P, ls="--", lw=1.8, color=PALETTE["FIT"],
        label=fr"seed-mean fit: $C_0{{=}}{C0_m:.2f},\ C_1{{=}}{k_m:.2f},\ R^2{{=}}{R2_m:.3f}$ (exponent {P})", zorder=2)
for i, (r, D, ys) in enumerate(data):
    ax.scatter([D]*len(ys), ys, s=22, facecolors="none", edgecolors=PALETTE["NEUTRAL"],
               linewidths=0.9, zorder=3, label="single seed" if i == 0 else None)
ax.errorbar(Dv, ymean, yerr=ystd, fmt="o", ms=7, color=PALETTE["FIT"], mec=PALETTE["INK"], mew=0.5,
            ecolor=PALETTE["INK"], elinewidth=1.0, capsize=3, zorder=4, label=r"$\lambda^2$ seed mean $\pm$ std")
for D, ym, n in zip(Dv, ymean, nseed):
    ax.annotate(f"n={n}", (D, ym), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5)
ax.axvline(Hr, ls=":", lw=1.0, color=PALETTE["NEUTRAL"]); ax.text(Hr, ax.get_ylim()[1], r" $H_r{=}8.0$",
            ha="left", va="top", fontsize=8, color=PALETTE["NEUTRAL"])
ax.set_xlabel(r"data-side $D$ = local conditional entropy")
ax.set_ylabel(r"$\lambda^2-\lambda_0^2\ (\times10^{-4})$")
ax.text(0.03, 0.05,
        f"$C_0{{=}}{C0_m:.2f}$  (95% CI [{C0_lo:.2f}, {C0_hi:.2f}])\n"
        f"$C_1{{=}}{k_m:.2f}$  (95% CI [{k_lo:.2f}, {k_hi:.2f}])\n"
        f"seed CV $\\approx$ {seed_cv:.1f}%   (n = {nseed})",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
ax.legend(loc="upper right", fontsize=7.5)
for o in ["03_paper/figures/p3_crossseed.pdf", "03_paper/figures/p3_crossseed.png"]:
    fig.savefig(o); print("WROTE", o)
print(f"  C0={C0_m:.2f} [{C0_lo:.2f},{C0_hi:.2f}]  C1={k_m:.2f} [{k_lo:.2f},{k_hi:.2f}]  R2={R2_m:.3f}  CV={seed_cv:.1f}%")
