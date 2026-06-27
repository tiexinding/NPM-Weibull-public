#!/usr/bin/env python
"""Law vs training step — matplotlib redraw of p3_law_vs_step (B2 6-22).
Per recorded step t, fit lambda^2(t)-lambda0^2 = C0(t)+C1(t)*(Hr-D)^0.59. (a) coefficient evolution,
(b) R^2(t) holds throughout. Data pipeline verbatim; English, no title, (a)/(b), p3_style. 守 #19/#76.
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p3_style import apply_style, PALETTE
V = 50304; Hr = 8.0; P = 0.59
def D_bi(t, n=2400000):
    t = t[:n].astype(np.int64); prev, nxt = t[:-1], t[1:]
    def H(x): _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    return float(H(prev * V + nxt) - H(prev))
def R2(y, yh): y = np.asarray(y); return 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2)
TAGS = ['000','010','020','030','040','050','060']
D = np.array([D_bi(np.load(f'tokens_corrupt/corrupt_rho{t}_s0.npy')) for t in TAGS])
u = np.power(np.clip(Hr - D, 1e-9, None), P)
traj = {}
for t in TAGS:
    j = json.load(open(f'_tmp_corrupt/corrupt_rho{t}_s0/lambda_trajectory_continue.json'))
    traj[t] = {r['step']: r['lambda'] for r in j}; traj[t]['l0'] = j[0]['lambda']
steps = [s for s in sorted(set(s for t in TAGS for s in traj[t] if isinstance(s, int))) if all(s in traj[t] for t in TAGS)]
rows = []
for s in steps:
    y = np.array([(traj[t][s]**2 - traj[t]['l0']**2) * 1e4 for t in TAGS])
    C1, C0 = np.polyfit(u, y, 1); rows.append(dict(step=s, C0=C0, C1=C1, R2=R2(y, C0 + C1 * u)))
S = np.array([r['step'] for r in rows]); C0 = np.array([r['C0'] for r in rows]); C1 = np.array([r['C1'] for r in rows]); R = np.array([r['R2'] for r in rows])
sig = S >= 64; C1sat = C1[sig][-1]

apply_style()
fig, (aL, aR) = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"width_ratios": [1.2, 1.0]})
aL.plot(S, C1, "-o", color=PALETTE["FIT"], lw=1.8, ms=4, label=r"$C_1(t)$ (data-coupling slope)")
aL.plot(S, C0, "--s", color=PALETTE["ACCENT"], lw=1.5, ms=3.5, label=r"$C_0(t)$ (injection floor)")
aL.axhline(C1sat, ls=":", lw=1.0, color=PALETTE["NEUTRAL"])
aL.text(S[-1], C1sat, fr" $C_1$ sat $\approx{C1sat:.2f}$", ha="right", va="bottom", fontsize=7.5, color=PALETTE["NEUTRAL"])
aL.set_xlabel(r"training step $t$"); aL.set_ylabel(r"law coefficient ($\times10^{-4}$)")
aL.text(0.03, 0.97, "(a)", transform=aL.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
aL.legend(loc="lower right", fontsize=7.5)
aR.plot(S[sig], R[sig], "-o", color="#2a9d5a", lw=1.8, ms=4, label=r"signal region ($t\geq64$)")
aR.scatter(S[~sig], R[~sig], s=24, facecolors="none", edgecolors=PALETTE["NEUTRAL"], linewidths=0.9, label=r"early (no signal)")
aR.axhline(0.9, ls=":", lw=1.0, color=PALETTE["NEUTRAL"])
aR.set_ylim(0.4, 1.10); aR.set_xlabel(r"training step $t$")
aR.set_ylabel(r"$R^2(t)$ vs $(H_r-D)^{0.59}$")
aR.text(0.03, 0.97, "(b)", transform=aR.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
aR.legend(loc="lower right", fontsize=7.5)
for ext in ("pdf","png"): fig.savefig(f"03_paper/figures/p3_law_vs_step.{ext}", bbox_inches="tight")
print(f"WROTE p3_law_vs_step  (C1 {C1[sig][0]:.3f}->{C1sat:.3f}, R2 median(signal)={np.median(R[sig]):.3f})")
