#!/usr/bin/env python
"""§7/appendix: second dimension = redundancy — matplotlib redraw of p3_2d_redundancy (B2 6-22).
Data pipeline verbatim (numbers unchanged); English, no in-figure title, p3_style. Slope -> C_1.
Color = repeated-bigram % (redundancy). 守 #19/#76/#47/#25(local).
"""
import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from p3_style import apply_style, PALETTE
V = 50304; Hr = 8.0

def feats(t, n=2400000):
    t = t[:n].astype(np.int64); prev = t[:-1]; nxt = t[1:]
    def H(x): _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    D = float(H(prev * V + nxt) - H(prev)); bg = prev * V + nxt
    return D, float(1 - len(np.unique(bg)) / len(bg)) * 100
def Y2_dir(d): j = json.load(open(d + "/lambda_trajectory_continue.json")); return (j[-1]["lambda"]**2 - j[0]["lambda"]**2) * 1e4
def Y2_json(f): j = json.load(open(f)); return (j[-1]["lambda"]**2 - j[0]["lambda"]**2) * 1e4

P = []
for r in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
    d = f"_tmp_corrupt/corrupt_rho{int(r*100):03d}_s0"; tk = f"tokens_corrupt/corrupt_rho{int(r*100):03d}_s0.npy"
    if os.path.exists(d + "/lambda_trajectory_continue.json"):
        D, rep = feats(np.load(tk)); P.append((f"shuffle r{r:.1f}", D, Y2_dir(d), rep, "shuffle"))
for k in [1,2,4,8,16,32]:
    d = f"_tmp_repeat/repeat_r{k}_s0"; tk = f"tokens_repeat/repeat_r{k}_s0.npy"
    if os.path.exists(d + "/lambda_trajectory_continue.json"):
        D, rep = feats(np.load(tk)); P.append((f"repeat r{k}", D, Y2_dir(d), rep, "repeat"))
for nm, tk, jf in [("code","tokens/code_30M.npy","_tmp_L3_code_s0.json"),
                   ("wikitext","tokens/wikitext_30M.npy","_tmp_L3_wikitext_s0.json"),
                   ("c4","tokens/c4_30M.npy","_tmp_L3_c4s0.json")]:
    if os.path.exists(tk) and os.path.exists(jf):
        D, rep = feats(np.load(tk)); P.append((nm, D, Y2_json(jf), rep, "natural"))
sh = [p for p in P if p[4] == "shuffle"]; Ds = np.array([p[1] for p in sh]); Ys = np.array([p[2] for p in sh])
k, C0 = np.polyfit(Hr - Ds, Ys, 1)
reps = np.array([p[3] for p in P]); norm = Normalize(reps.min(), reps.max()); cmap = "viridis"  # magnitude -> viridis, unified with arch-lam-maps (Fig 16)

apply_style()
fig, ax = plt.subplots(figsize=(6.6, 4.4))
xr = np.linspace(min(p[1] for p in P) - 0.1, Hr, 50)
ax.plot(xr, C0 + k * (Hr - xr), "--", color=PALETTE["NEUTRAL"], lw=1.6,
        label=fr"within-corpus law (shuffle): $\lambda^2{{-}}\lambda_0^2{{=}}{C0:.2f}{{+}}{k:.2f}(H_r{{-}}D)$")
MARK = {"shuffle": "o", "repeat": "X", "natural": "*"}; SZ = {"shuffle": 55, "repeat": 90, "natural": 230}
for grp in ["shuffle", "repeat", "natural"]:
    g = [p for p in P if p[4] == grp]
    ax.scatter([p[1] for p in g], [p[2] for p in g], c=[p[3] for p in g], cmap=cmap, norm=norm,
               marker=MARK[grp], s=SZ[grp], edgecolors=PALETTE["INK"], linewidths=0.5, zorder=3)
for p in P:
    if p[0] in ("code", "repeat r32"):
        ax.annotate(p[0], (p[1], p[2]), textcoords="offset points", xytext=(-6, -14),
                    ha="right", fontsize=8, color=PALETTE["ACCENT"])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.045, pad=0.02)
cb.set_label("repeated-bigram % (redundancy)", fontsize=8.5)
shape_leg = [Line2D([0],[0], marker=MARK[g], color="none", markerfacecolor=PALETTE["NEUTRAL"],
             markeredgecolor=PALETTE["INK"], markersize=8, label=lab)
             for g, lab in [("shuffle","shuffle (destroys mapping)"),("repeat","repeat (synthetic redundancy)"),("natural","natural corpora")]]
h, l = ax.get_legend_handles_labels()
ax.legend(h + shape_leg, l + [s.get_label() for s in shape_leg], loc="upper right", fontsize=7.2)
ax.set_xlabel(r"data-side $D$ = local conditional entropy")
ax.set_ylabel(r"$\lambda^2-\lambda_0^2\ (\times10^{-4})$")
for ext in ("pdf","png"): fig.savefig(f"03_paper/figures/p3_2d_redundancy.{ext}", bbox_inches="tight")
print(f"WROTE p3_2d_redundancy  (law C0={C0:.2f} C1={k:.2f})")
for l_, D, Y, rep, g in P:
    if l_ in ("code","wikitext","c4","repeat r32"): print(f"  {l_}: D={D:.2f} Y={Y:.2f} rep={rep:.0f}% [{g}]")
