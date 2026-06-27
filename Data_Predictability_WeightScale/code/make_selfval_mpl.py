#!/usr/bin/env python
"""Appendix A self-validation figure — matplotlib redraw of p3_selfval (B2 6-22).
Data pipeline verbatim from make_selfval.py (numbers unchanged); rendering rebuilt:
matplotlib + p3_style, English, no in-figure title, (a)/(b) panels. 守 #19/#76/#47/#25(local).
"""
import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p3_style import apply_style, PALETTE
V = 50304; Hr, dH, PE = 8.0, 1.82, 1.69
Sinv = lambda D: np.power(np.clip((Hr - np.asarray(D)) / dH, 1e-9, None), 1.0 / PE)
def D_bi(t, n=2400000):
    t = t[:n].astype(np.int64); prev, nxt = t[:-1], t[1:]
    def H(x): _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    return float(H(prev * V + nxt) - H(prev))
def struct(f):
    a = np.load(f); n = (len(a) // 512) * 512
    o = np.load("tokens_corrupt/corrupt_rho000_s0.npy")[:n].reshape(-1, 512); c = a[:n].reshape(-1, 512)
    return float(((o[:, :-1] == c[:, :-1]) & (o[:, 1:] == c[:, 1:])).mean())
def dl2_of(jf): j = json.load(open(jf)); return (j[-1]["lambda"]**2 - j[0]["lambda"]**2) * 1e4, j[0]["lambda"]

TAGS = ["000","010","020","030","040","050","060"]; RHO = [int(t)/100 for t in TAGS]
CROSS = {"000","020","040","060"}
D1, S1, Y1, Yerr = [], [], [], []
for tg in TAGS:
    fs = [f"_tmp_corrupt/corrupt_rho{tg}_s0/lambda_trajectory_continue.json"]
    if tg in CROSS: fs += [f"_tmp_grid/rho{tg}_e3e-4_s{s}/lambda_trajectory_continue.json" for s in (1, 2)]
    ys = [dl2_of(f)[0] for f in fs if os.path.exists(f)]
    D1.append(D_bi(np.load(f"tokens_corrupt/corrupt_rho{tg}_s0.npy"))); S1.append(struct(f"tokens_corrupt/corrupt_rho{tg}_s0.npy"))
    Y1.append(np.mean(ys)); Yerr.append(np.std(ys) if len(ys) > 1 else 0.0)
D1, S1, Y1, Yerr = map(np.array, [D1, S1, Y1, Yerr])
fitA = lambda Sx, Yx: (lambda C1, C0: (C0, C1))(*np.polyfit(Sx, Yx, 1))
C0a, C1a = fitA(S1, Y1)
loo_full, loo_A = [], []
for i in range(len(TAGS)):
    m = np.arange(len(TAGS)) != i; C0, C1 = fitA(S1[m], Y1[m])
    loo_full.append(C0 + C1 * Sinv(D1[i])); loo_A.append(C0 + C1 * S1[i])
loo_full, loo_A = map(np.array, [loo_full, loo_A])
rmse_loo = float(np.sqrt(np.mean((Y1 - loo_full)**2)))
relerr_loo = float(np.mean(np.abs(Y1 - loo_full) / np.abs(Y1)))
NAT = {"wikitext": "tokens_corrupt/corrupt_rho000_s0.npy", "c4": "tokens/c4_30M.npy", "code": "tokens/code_30M.npy"}
def nat_y(name):
    fs = [f"L3_results_e3e4/continue_{name}_s{s}/lambda_trajectory_continue.json" for s in (0, 1, 2)]
    ys = [dl2_of(f)[0] for f in fs if os.path.exists(f)]
    if not ys:
        f = f"p3_out/continue_{name}/lambda_trajectory_continue.json"
        if os.path.exists(f): ys = [dl2_of(f)[0]]
    return (np.mean(ys) if ys else None)
L2 = []
for name, tk in NAT.items():
    Dn = D_bi(np.load(tk)); yo = nat_y(name)
    if yo is None: continue
    L2.append((name, yo, C0a + C1a * Sinv(Dn)))

apply_style()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 3.9), gridspec_kw={"width_ratios": [1.15, 1.0]})
# (a) predicted vs measured
lo, hi = 0, 2
axA.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=PALETTE["NEUTRAL"], label="$y=x$ (perfect)", zorder=1)
axA.errorbar(Y1, loo_full, xerr=Yerr, fmt="o", ms=7, color=PALETTE["FIT"], mec=PALETTE["INK"], mew=0.5,
             ecolor=PALETTE["NEUTRAL"], elinewidth=0.9, capsize=2, zorder=3, label="within-corpus LOO blind")
starcol = {"code": PALETTE["ACCENT"], "c4": "#2a9d5a", "wikitext": PALETTE["NEUTRAL"]}
for name, yo, yp in L2:
    axA.scatter([yo], [yp], s=150, marker="*", color=starcol[name], edgecolors=PALETTE["INK"],
                linewidths=0.5, zorder=5)
    axA.annotate(f" {name}", (yo, yp), ha="left", va="center", fontsize=8.5, color=starcol[name])
axA.set_xlim(0, 2); axA.set_ylim(0, 2); axA.set_aspect("equal", "box")   # equal-valued pred-vs-measured, 0->2 (老丁 6-23)
axA.set_xlabel(r"measured $\lambda^2-\lambda_0^2\ (\times10^{-4})$")
axA.set_ylabel(r"predicted $\lambda^2-\lambda_0^2\ (\times10^{-4})$")
axA.text(0.03, 0.975, "(a)", transform=axA.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
axA.text(0.03, 0.85, f"LOO RMSE = {rmse_loo:.3f}\n({relerr_loo*100:.1f}% rel.)", transform=axA.transAxes,
         ha="left", va="top", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
axA.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.85, edgecolor="none")
# (b) error decomposition bars
x = np.arange(len(RHO)); w = 0.4
axB.bar(x - w/2, 100*np.abs(Y1 - loo_A)/Y1, w, color="#2a9d5a", label="A-rung (true $S$)")
axB.bar(x + w/2, 100*np.abs(Y1 - loo_full)/Y1, w, color=PALETTE["FIT"], label="full chain ($D$ only)")
axB.set_ylim(0, 15)   # % scale; headroom so the (b) letter clears the tall rho=0 bar (老丁 6-23)
axB.set_xticks(x); axB.set_xticklabels([f"{r:.1f}" for r in RHO], fontsize=7.5)
axB.set_xlabel(r"corruption $\rho$"); axB.set_ylabel(r"relative error (\%)")
axB.text(0.03, 0.97, "(b)", transform=axB.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
axB.legend(loc="upper right", fontsize=7.5)
for o in ["03_paper/figures/p3_selfval.pdf", "03_paper/figures/p3_selfval.png"]:
    fig.savefig(o); print("WROTE", o)
print(f"  LOO RMSE={rmse_loo:.3f} ({relerr_loo*100:.1f}%)")
for name, yo, yp in L2: print(f"  L2 {name}: obs={yo:.2f} pred={yp:.2f} d={yo-yp:+.2f}")
