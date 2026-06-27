#!/usr/bin/env python
"""HERO figure (§4 collapse) — matplotlib/publication redraw of p3_norm_convex (B2 6-22).
Data pipeline + collapse stats verbatim from make_convex.py fig1 (numbers unchanged); rendering
rebuilt: matplotlib + p3_style, English, no in-figure title, vector PDF. (eta_scaling + arch_compare,
the other two figs from make_convex.py, are redrawn separately in Tier 2.)
守 #19/#76/#47/#25(local).
"""
import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import t as _tdist
from p3_style import apply_style, PALETTE
V = 50304; Hr = 8.0; P = 0.59

def D_bi(t, n=2400000):
    t = t[:n].astype(np.int64); prev = t[:-1]; nxt = t[1:]
    def H(x): _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    return float(H(prev * V + nxt) - H(prev))
def Y2(f): j = json.load(open(f)); return (j[-1]["lambda"]**2 - j[0]["lambda"]**2) * 1e4
def R2f(y, yh): return 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2)
_Dc = {}
def Dtok(tag):
    if tag not in _Dc: _Dc[tag] = D_bi(np.load(f"tokens_corrupt/corrupt_rho{tag}_s0.npy"))
    return _Dc[tag]
def U(D): return np.power(np.clip(Hr - np.asarray(D), 1e-9, None), P)
def corr(t): return f"_tmp_corrupt/corrupt_rho{t}_s0/lambda_trajectory_continue.json"
def grid(t, e, s=0): return f"_tmp_grid/rho{t}_e{e}_s{s}/lambda_trajectory_continue.json"
TAGS4 = ["000","020","040","060"]; TAGS11 = ["000","010","020","030","040","050","060","070","080","090","100"]
ETA_SRC = {  # keep the established 4-eta colour identity (red/orange/blue/green), low->high eta
 "1e-3": (PALETTE["ACCENT"], [(t, [grid(t, "1e-3")]) for t in TAGS4]),
 "5e-4": (PALETTE["ARCH_B"], [(t, [grid(t, "5e-4", 0), grid(t, "5e-4", 1)]) for t in TAGS4]),
 "3e-4": (PALETTE["ARCH_A"], [(t, [corr(t)]) for t in TAGS11]),
 "1e-4": ("#2a9d5a",        [(t, [grid(t, "1e-4")]) for t in TAGS4]),
}
def load(src):
    D = []; Y = []
    for t, fs in src:
        ys = [Y2(f) for f in fs if os.path.exists(f)]
        if ys: D.append(Dtok(t)); Y.append(np.mean(ys))
    return np.array(D), np.array(Y)

coef = {}; pts = []
for nm, (col, src) in ETA_SRC.items():
    D, Y = load(src)
    if len(D) < 2: continue
    u = U(D); k, C0 = np.polyfit(u, Y, 1); yn = (Y - C0) / k
    coef[nm] = (C0, k, len(D)); pts.append((nm, col, D, yn))
Dv = np.concatenate([p[2] for p in pts]); yv = np.concatenate([p[3] for p in pts]); uv = U(Dv); n = len(yv)
b1, a1 = np.polyfit(uv, yv, 1)
resid = yv - (a1 + b1 * uv); s = float(np.sqrt(np.sum(resid**2) / (n - 2)))
ubar = float(uv.mean()); Suu = float(np.sum((uv - ubar)**2)); tval = float(_tdist.ppf(0.975, n - 2))
b_lo, b_hi = b1 - tval * s / np.sqrt(Suu), b1 + tval * s / np.sqrt(Suu)
R2fit = R2f(yv, a1 + b1 * uv); R2id = R2f(yv, uv); half = tval * s * np.sqrt(1 + 1 / n)
Dg = np.linspace(float(Dv.min()), float(Dv.max()), 80); ug = U(Dg); yfit = a1 + b1 * ug
se_m = s * np.sqrt(1 / n + (ug - ubar)**2 / Suu); se_p = s * np.sqrt(1 + 1 / n + (ug - ubar)**2 / Suu)
ci_lo, ci_hi = yfit - tval * se_m, yfit + tval * se_m; pi_lo, pi_hi = yfit - tval * se_p, yfit + tval * se_p

apply_style()
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.fill_between(Dg, pi_lo, pi_hi, color=PALETTE["NEUTRAL"], alpha=0.15, lw=0, zorder=1)
ax.fill_between(Dg, ci_lo, ci_hi, color=PALETTE["FIT"], alpha=0.20, lw=0, zorder=2)
ax.plot(Dg, yfit, color=PALETTE["FIT"], lw=2.0, zorder=3, label=fr"regression ($R^2{{=}}{R2fit:.3f}$)")
ax.plot(Dg, ug, color=PALETTE["INK"], lw=1.6, ls="--", zorder=3, label=r"identity $y=(H_r-D)^{0.59}$")
ETA_LBL = {"1e-3": r"10^{-3}", "5e-4": r"5{\times}10^{-4}", "3e-4": r"3{\times}10^{-4}", "1e-4": r"10^{-4}"}
for nm, col, D, yn in pts:
    C0, k, nn = coef[nm]
    ax.scatter(D, yn, s=42, color=col, edgecolors=PALETTE["INK"], linewidths=0.5, zorder=5,
               label=fr"$\eta={ETA_LBL[nm]}$  ($C_0{{=}}{C0:.2f},\ C_1{{=}}{k:.2f},\ n={nn}$)")
ax.set_ylim(top=2.0)   # headroom so the legend clears the band/data (老丁 6-22)
ax.axvline(Hr, ls=":", lw=1.0, color=PALETTE["NEUTRAL"])
ax.text(Hr, ax.get_ylim()[1], r" $H_r{=}8.0$", ha="left", va="top", fontsize=8, color=PALETTE["NEUTRAL"])
# legend handles: bands (patches) + lines + scatters
band_handles = [Patch(facecolor=PALETTE["NEUTRAL"], alpha=0.15, label=fr"95% prediction band $\pm{half:.2f}$"),
                Patch(facecolor=PALETTE["FIT"], alpha=0.20, label="95% confidence interval")]
h, l = ax.get_legend_handles_labels()
ax.legend(band_handles + h, [p.get_label() for p in band_handles] + l, loc="upper right", fontsize=7.2)
_consist = "consistent with theory $b{=}1$" if b_lo <= 1 <= b_hi else "inconsistent with theory $b{=}1$"
ax.text(0.03, 0.05,
        f"slope $b={b1:.2f}$  (95% CI [{b_lo:.2f}, {b_hi:.2f}]; {_consist})\n"
        f"residual $\\sigma={s:.2f}$  $\\cdot$  band half-width $\\pm{half:.2f}$  $\\cdot$  $n={n}$",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["GRID"], lw=0.8, alpha=0.9))
ax.set_xlabel(r"data-side $D$ = local conditional entropy")
ax.set_ylabel(r"normalized growth  $(\lambda^2-\lambda_0^2-C_0)/C_1$")
for o in ["03_paper/figures/p3_norm_convex.pdf", "03_paper/figures/p3_norm_convex.png"]:
    fig.savefig(o); print("WROTE", o)
print(f"  collapse: R2_identity={R2id:.3f}  R2_fit={R2fit:.3f}  slope b={b1:.2f} [{b_lo:.2f},{b_hi:.2f}]  sigma={s:.2f}  half={half:.2f}  n={n}")
for nm, (c, k, nn) in coef.items(): print(f"    eta={nm}: C0={c:.2f} C1={k:.2f} n={nn}")

# ════════ eta_scaling (§6.1): per-eta gradient (a) + coefficient scaling (b) ════════
ETA_VAL = {"1e-3": 1e-3, "5e-4": 5e-4, "3e-4": 3e-4, "1e-4": 1e-4}
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(6.8, 3.4), gridspec_kw={"width_ratios": [1.2, 1.0]})
sc = []
for nm, (col, src) in ETA_SRC.items():
    D, Y = load(src)
    if len(D) < 2: continue
    u = U(D); k, C0 = np.polyfit(u, Y, 1); sc.append((ETA_VAL[nm], C0, k))
    axL.scatter(D, Y, s=34, color=col, edgecolors=PALETTE["INK"], linewidths=0.4, zorder=3,
                label=fr"$\eta={ETA_LBL[nm]}$  ($C_1{{=}}{k:.2f}$)")
    xr = np.linspace(min(D) - 0.05, Hr, 40); axL.plot(xr, C0 + k * U(xr), "--", color=col, lw=1.2, alpha=0.7)
axL.set_xlabel(r"data-side $D$"); axL.set_ylabel(r"$\lambda^2-\lambda_0^2\ (\times10^{-4})$")
axL.text(0.03, 0.97, "(a)", transform=axL.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
axL.set_ylim(top=13)   # headroom so the legend clears the top red points (老丁 6-22)
axL.legend(loc="upper right", fontsize=9)
ev = np.array([s_[0] for s_ in sc]); C0v = np.array([s_[1] for s_ in sc]); kv = np.array([s_[2] for s_ in sc])
pC0 = np.polyfit(np.log10(ev), np.log10(np.abs(C0v)), 1); pk = np.polyfit(np.log10(ev), np.log10(np.abs(kv)), 1)
R2C0 = R2f(np.log10(np.abs(C0v)), pC0[1] + pC0[0] * np.log10(ev)); R2k = R2f(np.log10(np.abs(kv)), pk[1] + pk[0] * np.log10(ev))
axR.scatter(ev, C0v, s=44, color=PALETTE["FIT"], edgecolors=PALETTE["INK"], linewidths=0.4,
            label=fr"$C_0\propto\eta^{{{pC0[0]:.2f}}}$ ($R^2{{=}}{R2C0:.3f}$)")
axR.scatter(ev, kv, s=44, color=PALETTE["ACCENT"], marker="D", edgecolors=PALETTE["INK"], linewidths=0.4,
            label=fr"$C_1\propto\eta^{{{pk[0]:.2f}}}$ ($R^2{{=}}{R2k:.3f}$)")
xe = np.array([ev.min() * 0.9, ev.max() * 1.1])
axR.plot(xe, 10**(pC0[1] + pC0[0] * np.log10(xe)), ":", color=PALETTE["FIT"], lw=1.2)
axR.plot(xe, 10**(pk[1] + pk[0] * np.log10(xe)), ":", color=PALETTE["ACCENT"], lw=1.2)
axR.set_xscale("log"); axR.set_yscale("log"); axR.set_xlabel(r"learning rate $\eta$"); axR.set_ylabel("coefficient")
axR.text(0.03, 0.97, "(b)", transform=axR.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
axR.legend(loc="lower right", fontsize=7.5)
for o in ["03_paper/figures/p3_eta_scaling_convex.pdf", "03_paper/figures/p3_eta_scaling_convex.png"]:
    fig2.savefig(o); print("WROTE", o)
print(f"  eta_scaling: C0~eta^{pC0[0]:.2f}(R2={R2C0:.3f})  C1~eta^{pk[0]:.2f}(R2={R2k:.3f})")
