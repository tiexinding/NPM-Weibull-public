"""Cross-architecture figures (§6.3) — unified to p3_style (B2 6-22 Tier 2; was F149/turbo 6-19).
Three figs from one data pipeline (verbatim): ① p3_arch_compare_convex 7-point curves
② p3_arch_mechanism per-module C1 + k-gate ③ p3_arch_perlayer layer x module heatmap.
Tier-2 fixes: p3_style palette (Pythia blue / Llama orange, was cyan/red), NO bold in-figure titles
(守#76), NO callout arrow (守#82), compare-legend coefficients -> indicative (守#80, single seed),
diverging RdBu_r symmetric for signed C1 (守 6-19). Vector PDF + PNG to 03_paper/figures/.
"""
import json, numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from math import gamma
from p3_style import apply_style, PALETTE
apply_style()
PYC = PALETTE["ARCH_A"]; LLC = PALETTE["ARCH_B"]   # Pythia blue / Llama orange
INK = PALETTE["INK"]

V = 50304; Hr = 8.0; P = 0.59
TAGS = ["000","010","020","030","040","050","060"]
def D_bi(t, n=2400000):
    t = np.load(t)[:n].astype(np.int64); pr, nx = t[:-1], t[1:]
    def H(x): _, c = np.unique(x, return_counts=True); p = c/c.sum(); return -np.sum(p*np.log2(p))
    return float(H(pr*V+nx) - H(pr))
Dc = {t: D_bi(f"tokens_corrupt/corrupt_rho{t}_s0.npy") for t in TAGS}
Darr = np.array([Dc[t] for t in TAGS]); u7 = np.power(np.clip(Hr-Darr, 1e-9, None), P)
U = lambda D: np.power(np.clip(Hr-np.asarray(D), 1e-9, None), P)
def mtype(n):
    if "h_to_4h" in n or "up_proj" in n: return "FFN-up"
    if "4h_to_h" in n or "down_proj" in n: return "FFN-down"
    if "gate_proj" in n: return "FFN-gate"
    if "dense.weight" in n or "o_proj" in n: return "Attn-out"
    return None
def traj(a, t):
    j = json.load(open(f"_tmp_arch/scratch_{a}_rho{t}_s0/lambda_trajectory_continue.json"))
    return (j[-1]["lambda"]**2 - j[0]["lambda"]**2)*1e4
def budget(a, t):
    L = open(f"_tmp_arch/scratch_{a}_rho{t}_s0/v1b_perlayer_budget.jsonl").read().strip().split("\n")
    return json.loads(L[0])["layers"], json.loads(L[-1])["layers"]
def R2(y, yh): return 1 - np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
ARCH = [("pythia", "Pythia (GELU FFN)", PYC, "o"), ("llama", "Llama (SwiGLU FFN)", LLC, "D")]

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"03_paper/figures/{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print("WROTE", name)

# ════════ ① 7-point curves (legend coefficients -> indicative, 守#80) ════════
fig, ax = plt.subplots(figsize=(6.5, 4.3))
for a, lab, col, mk in ARCH:
    D = Darr; Y = np.array([traj(a, t) for t in TAGS]); u = U(D)
    C1, C0 = np.polyfit(u, Y, 1); r2 = R2(Y, C0+C1*u)
    ax.scatter(D, Y, s=70, color=col, marker=mk, edgecolor=INK, linewidth=0.5, zorder=3, label=lab)
    xr = np.linspace(D.min()-0.03, D.max()+0.03, 60); ax.plot(xr, C0+C1*U(xr), "--", color=col, lw=1.5, alpha=0.7)
    print(f"  {a}: C0={C0:.2f} C1={C1:.2f} R2={r2:.2f}")
ax.set_xlim(6.0, 8.15); ax.set_ylim(1.4, 2.72)   # margins so points clear the frame (老丁 6-23)
ax.axvline(Hr, color=PALETTE["NEUTRAL"], ls=":", lw=1.0)
ax.text(Hr-0.04, 1.45, r"$H_r{=}8.0$", ha="right", va="bottom", fontsize=8, color=PALETTE["NEUTRAL"])
ax.text(0.03, 0.05, "Llama steeper at all 7 $\\rho$ (7/7);\ncoefficients indicative (single seed)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
ax.set_xlabel(r"data-side $D$ = local conditional entropy")
ax.set_ylabel(r"$\lambda^2-\lambda_0^2$  ($\times10^{-4}$)")
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "p3_arch_compare_convex")

# ════════ ② mechanism: per-module C1 bars + k-gate (no titles, (a)/(b), no callout) ════════
order = ["Attn-out", "FFN-gate", "FFN-up", "FFN-down"]
C1m = {}; KS = {}
for a, *_ in ARCH:
    types = {}; ks = []
    for t in TAGS:
        f, l = budget(a, t)
        for nm in l:
            mt = mtype(nm)
            if mt is None: continue
            l2 = l[nm]["rms"]**2 / gamma(1+2/l[nm]["k80"]); f2 = f[nm]["rms"]**2 / gamma(1+2/f[nm]["k80"])
            types.setdefault(mt, {}).setdefault(t, []).append((l2 - f2)*1e4)
            ks.append(l[nm]["k80"])
    KS[a] = np.array(ks); C1m[a] = {}
    for mt in order:
        if mt not in types: C1m[a][mt] = np.nan; continue
        Y = np.array([np.mean(types[mt][t]) for t in TAGS]); C1m[a][mt] = np.polyfit(u7, Y, 1)[0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.4), gridspec_kw={"width_ratios": [1.7, 1]})
x = np.arange(len(order)); w = 0.38
for i, (a, lab, col, mk) in enumerate(ARCH):
    vals = [C1m[a][mt] for mt in order]
    a1.bar(x + (i-0.5)*w, [v if not np.isnan(v) else 0 for v in vals], w, color=col,
           label=lab.split(" ")[0], edgecolor="white", zorder=3)
    for xi, v in zip(x+(i-0.5)*w, vals):
        if not np.isnan(v): a1.text(xi, v+0.03, f"{v:.2f}", ha="center", fontsize=7.5)
a1.set_xticks(x); a1.set_xticklabels(order, fontsize=8); a1.set_ylabel(r"per-module $C_1$ ($\lambda^2$ sensitivity)")
a1.set_ylim(0, 1.05); a1.legend(loc="upper left", fontsize=8); a1.grid(axis="y", alpha=0.3)
a1.text(0.97, 0.97, "(a)", transform=a1.transAxes, ha="right", va="top", fontsize=11, fontweight="bold")
# (SwiGLU-gate = Llama-only note moved to the LaTeX caption, 守#76/老丁 6-22)
for a, lab, col, mk in ARCH:
    k = KS[a]; med = np.median(k)
    a2.errorbar([0 if a=="pythia" else 1], [med], yerr=[[med-np.percentile(k,25)],[np.percentile(k,75)-med]],
                fmt=mk, ms=11, color=col, mec=INK, mew=0.5, capsize=4, elinewidth=1.4)
kp, kl = np.median(KS["pythia"]), np.median(KS["llama"]); ratio = gamma(1+2/kl)/gamma(1+2/kp)
a2.text(0.5, 0.97, f"$k$: {kp:.3f} vs {kl:.3f}\n$\\Gamma(1{{+}}2/k)$ ratio $= {ratio:.3f}$\n"
        f"({abs(ratio-1)*100:.1f}%, negligible)\nreal $\\mathrm{{RMS}}^2$ dynamics",
        transform=a2.transAxes, ha="center", va="top", fontsize=7.5)
a2.set_xticks([0, 1]); a2.set_xticklabels(["Pythia", "Llama"], fontsize=8); a2.set_xlim(-0.6, 1.6)
a2.set_ylim(1.15, 1.27); a2.set_ylabel(r"Weibull shape $k$"); a2.grid(axis="y", alpha=0.3)
a2.text(0.05, 0.97, "(b)", transform=a2.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
save(fig, "p3_arch_mechanism")

# ════════ ③ layer x module heatmap (RdBu_r diverging, symmetric; arch name = non-bold identity) ════════
LAYERS = list(range(6))
def grid(a):
    g = {}
    for t in TAGS:
        f, l = budget(a, t)
        for nm in l:
            mt = mtype(nm)
            if mt is None: continue
            l2 = l[nm]["rms"]**2 / gamma(1+2/l[nm]["k80"]); f2 = f[nm]["rms"]**2 / gamma(1+2/f[nm]["k80"])
            g.setdefault((mt, l[nm]["layer"]), []).append((l2 - f2)*1e4)
    M = np.full((len(order), len(LAYERS)), np.nan)
    for i, mt in enumerate(order):
        for j, lay in enumerate(LAYERS):
            if (mt, lay) in g: M[i, j] = np.polyfit(u7, np.array(g[(mt, lay)]), 1)[0]
    return M
fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.7), gridspec_kw={"wspace": 0.10})
# viridis (sequential), unified with Fig16/Fig19 (老丁 6-27: 量级梯度故事优先 viridis;
# C1 的正负=浅层 inverse partly noise 非头条, 且每格数值已标出, 符号信息不丢). vmin/vmax span actual data.
grids = [np.ma.masked_invalid(grid(a)) for a, lab, col, mk in ARCH]
allv = np.concatenate([g.compressed() for g in grids])
VMIN, VMAX = float(allv.min()), float(allv.max())
cmap = mpl.colormaps["viridis"].copy(); cmap.set_bad("0.90")
for ax, (a, lab, col, mk), M in zip(axes, ARCH, grids):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=VMIN, vmax=VMAX, origin="lower")
    ax.set_xticks(LAYERS); ax.set_xticklabels([f"L{j}" for j in LAYERS], fontsize=8); ax.set_xlabel("block index")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    ax.set_title(lab, fontsize=9)            # non-bold (apply_style) architecture identity, not a description
    ax.grid(False)
    for i in range(len(order)):
        for j in range(len(LAYERS)):
            if not np.ma.is_masked(M[i, j]):
                frac = (M[i, j] - VMIN) / (VMAX - VMIN)
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if frac < 0.55 else INK)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
cb.set_label(r"data sensitivity $C_1$ (slope of $\lambda^2-\lambda_0^2$)", fontsize=8.5)
save(fig, "p3_arch_perlayer")
print("done: arch 3 figs -> p3_style")
