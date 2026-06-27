#!/usr/bin/env python
"""§5 derivation figure — matplotlib/publication version (B2 6-22, redraw of p3_physics_fits_convex).

Identical data pipeline + fits to make_physics_fits_convex.py (numbers unchanged); only the
rendering is rebuilt: matplotlib + p3_style, English, NO in-figure suptitle/panel titles
(caption provides), (a)-(d) panel letters, vector PDF. 守 #19/#76/#25(data local). Spec:
03_paper/figure_style_spec.md.

A  training-side linear in S:   l2-l0^2 = a0 + a1*S
B  data-side saturation:        D = Hr - dH*S^p   (fits p~1.69)
C  composition -> convex law:    l2-l0^2 = C0 + C1*(Hr-D)^(1/p),  exponent 0.59 = 1/p
D  texture (high-order, no closed form): texture = a*exp(b*(1-S))
"""
import numpy as np, glob, os, json
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p3_style import apply_style, PALETTE

V = 50304

# ----------------- data pipeline (verbatim from make_physics_fits_convex.py) -----------------
def short(b):
    L = b.split(".")[2]; tt = "Wo" if "attention" in b else ("up" if "h_to_4h" in b else "dn")
    return (int(L), {"Wo": 0, "up": 1, "dn": 2}[tt])
def tex(jf):
    rows = [json.loads(l) for l in open(jf)]; steps = np.array([r["step"] for r in rows]); post = steps >= 300
    bo = sorted(rows[0]["layers"].keys(), key=short); Mp = np.array([[r["layers"][b]["align"] for r in rows] for b in bo])[:, post]
    return float(np.mean([np.mean(np.abs(np.diff(Mp[i]))) / np.mean(np.abs(Mp[i])) for i in range(Mp.shape[0]) if np.mean(np.abs(Mp[i])) > 0]))
def struct(f):
    a = np.load(f); n = (len(a) // 512) * 512
    o = np.load("tokens_corrupt/corrupt_rho000_s0.npy")[:n].reshape(-1, 512); c = a[:n].reshape(-1, 512)
    return float(((o[:, :-1] == c[:, :-1]) & (o[:, 1:] == c[:, 1:])).mean())
def D_bi(t, n=2400000):
    t = t[:n].astype(np.int64); prev = t[:-1]; nxt = t[1:]
    def H(x): _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    return float(H(prev * V + nxt) - H(prev))

S=[];T=[];li=[];le=[];D=[]
for f in sorted(glob.glob("tokens_corrupt/corrupt_rho*_s0.npy")):
    r=os.path.basename(f).split("rho")[1].split("_")[0]; d=f"_tmp_corrupt/corrupt_rho{r}_s0"
    S.append(struct(f)); T.append(tex(d+"/v1b_perlayer_budget.jsonl")); D.append(D_bi(np.load(f)))
    j=json.load(open(d+"/lambda_trajectory_continue.json")); li.append(j[0]["lambda"]); le.append(j[-1]["lambda"])
S,T,li,le,D=map(np.array,[S,T,li,le,D])
dl2=(le**2-li**2)*1e4
def R2(y,yh): return 1-np.sum((y-yh)**2)/np.sum((y-np.mean(y))**2)

pA,_=curve_fit(lambda s,a,b:a+b*s,S,dl2,p0=[1,5]); r2A=R2(dl2,pA[0]+pA[1]*S)
pB,_=curve_fit(lambda s,Hr,dH,p:Hr-dH*np.power(np.clip(s,1e-9,None),p),S,D,p0=[8.0,1.8,1.7],maxfev=20000)
Hr_f,dH_f,p_f=pB; r2B=R2(D,Hr_f-dH_f*np.power(np.clip(S,1e-9,None),p_f)); expo=1.0/p_f
u=np.power(np.clip(Hr_f-D,1e-9,None),expo)
pC,_=curve_fit(lambda x,C0,C1:C0+C1*x,u,dl2,p0=[0.5,1]); r2C=R2(dl2,pC[0]+pC[1]*u)
pT,_=curve_fit(lambda x,a,b:a*np.exp(b*x),1-S,T,p0=[0.18,1]); r2T=R2(T,pT[0]*np.exp(pT[1]*(1-S)))

# ----------------- render (matplotlib) -----------------
apply_style()
COL = {"A": PALETTE["ACCENT"], "B": PALETTE["FIT"], "C": "#2a9d5a", "D": PALETTE["ARCH_B"]}
FITLINE = "#555555"
fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.2))   # 3 panels a/b/c; texture panel (d) removed (老丁 6-27: texture 不属律, 正文不提)
fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.10, hspace=0.11)

def panel(ax, letter, x, y, xf, yf, col, xlabel, ylabel, fit_txt, txt_xy, ha, va):
    ax.plot(xf, yf, ls="--", lw=1.4, color=FITLINE, zorder=1)
    ax.scatter(x, y, s=34, color=col, edgecolors=PALETTE["INK"], linewidths=0.5, zorder=3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.text(0.025, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=11, fontweight="bold")
    ax.text(txt_xy[0], txt_xy[1], fit_txt, transform=ax.transAxes, ha=ha, va=va, fontsize=7.5,
            color=PALETTE["INK"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.65))

Ss = np.linspace(0, 1, 60)
# A: dl2 linear in S (increasing) -> annotate lower-right
panel(axes[0], "a", S, dl2, Ss, pA[0]+pA[1]*Ss, COL["A"],
      r"structure retained $S$", r"$\lambda^2-\lambda_0^2\ (\times10^{-4})$",
      f"$\\lambda^2{{-}}\\lambda_0^2 = a_0 + a_1 S$\n$a_0{{=}}{pA[0]:.2f},\\ a_1{{=}}{pA[1]:.2f},\\ R^2{{=}}{r2A:.3f}$",
      (0.97, 0.05), "right", "bottom")
# B: D saturation, decreasing in S -> annotate lower-left
panel(axes[1], "b", S, D, Ss, Hr_f-dH_f*np.power(Ss,p_f), COL["B"],
      r"structure retained $S$", r"$D$  (bits)",
      f"$D = H_r - \\Delta H\\, S^{{p}}$\n$H_r{{=}}{Hr_f:.2f},\\ p{{=}}{p_f:.2f},\\ R^2{{=}}{r2B:.4f}$",
      (0.03, 0.05), "left", "bottom")
axes[1].set_ylim(top=9.0)   # headroom so the (b) letter clears the curve at D=8
# C: dl2 linear in u (increasing) -> annotate lower-right
us = np.linspace(min(u), max(u), 60)
panel(axes[2], "c", u, dl2, us, pC[0]+pC[1]*us, COL["C"],
      rf"$(H_r-D)^{{{expo:.2f}}}$", r"$\lambda^2-\lambda_0^2\ (\times10^{-4})$",
      f"$\\lambda^2{{-}}\\lambda_0^2 = C_0 + C_1 (H_r{{-}}D)^{{{expo:.2f}}}$\n"
      f"$R^2{{=}}{r2C:.3f}$, exponent ${expo:.2f}{{=}}1/{p_f:.2f}$ (derived)",
      (0.97, 0.05), "right", "bottom")

for outp in ["03_paper/figures/p3_physics_fits_convex.pdf",
             "03_paper/figures/p3_physics_fits_convex.png"]:
    fig.savefig(outp)
    print("WROTE", outp)
print(f"  A: a0={pA[0]:.3f} a1={pA[1]:.3f} R2={r2A:.3f}")
print(f"  B: Hr={Hr_f:.3f} dH={dH_f:.3f} p={p_f:.3f} R2={r2B:.4f}  -> 1/p={expo:.3f}")
print(f"  C: C0={pC[0]:.3f} C1={pC[1]:.3f} R2={r2C:.3f}")
print(f"  D: texture R2={r2T:.3f}")
