#!/usr/bin/env python
"""Paper#2 新小节"训练数据对λ的塑造关系" 核心图.
2-panel: (a)数据决定λ水平 (b)continue-train因果. 出版级. """
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
PF = ""
def traj(f):
    r = json.load(open(f)); k="lambda_obs" if "lambda_obs" in r[0] else "lambda"
    return np.array([x["step"] for x in r]), np.array([x[k] for x in r])

fig = plt.figure(figsize=(12,4.8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.28)

# ---- (a) 数据决定 λ 水平 (bars) ----
axa = fig.add_subplot(gs[0])
labels = ["Real Pythia\n(Pile)", "Continue\n(Pythia\n+wikitext)", "Scratch\nwikitext\n118M", "Scratch\nwikitext\n8M"]
peaks  = [0.0352, 0.0598, 0.0688, 0.0763]
colors = ["#2c7fb8", "#7fcdbb", "#fec44f", "#d95f0e"]
bars = axa.bar(labels, peaks, color=colors, edgecolor="black", linewidth=0.8, width=0.7)
for b,p in zip(bars,peaks): axa.text(b.get_x()+b.get_width()/2, p+0.0012, f"{p:.3f}", ha="center", fontsize=9.5, fontweight="bold")
axa.axhline(0.0352, ls="--", c="#2c7fb8", alpha=0.45, lw=1)
axa.set_ylabel("Peak Weibull $\\lambda$ (Transmission Class)", fontsize=10.5)
axa.set_ylim(0, 0.088); axa.tick_params(labelsize=8.5)
axa.set_title("(a) Data distribution sets absolute $\\lambda$", fontsize=11, pad=8)

# ---- (b) continue-train 因果轨迹 ----
axb = fig.add_subplot(gs[1])
sc, lc = traj("derived_data/self_train/lambda_trajectory_continue.json")
axb.plot(sc, lc, "-o", ms=3.2, c="#1b7837", lw=1.8, label="Continue: real Pythia + wikitext", zorder=5)
axb.axhline(0.0352, ls="--", c="#2c7fb8", lw=1.3, label="Real Pythia (Pile) peak = 0.035")
axb.axhline(0.0688, ls="--", c="#fec44f", lw=1.3, label="Scratch wikitext 118M = 0.069")
axb.axhline(0.0232, ls=":", c="gray", alpha=0.8, lw=1.2, label="Start (Pile-trained init) = 0.023")
axb.set_xlabel("Continue-training step (on wikitext)", fontsize=10.5)
axb.set_ylabel("Weibull $\\lambda$", fontsize=10.5)
axb.legend(fontsize=8.2, loc="lower right", framealpha=0.9); axb.set_ylim(0.015, 0.082)
axb.tick_params(labelsize=8.5)
axb.set_title("(b) Causal test: same model, swap Pile$\\to$wikitext", fontsize=11, pad=8)
axb.annotate("climbs past Pile peak\ntoward wikitext level\n(2.6$\\times$ start, 1.7$\\times$ Pile)", xy=(13000,0.058), xytext=(5200,0.070),
             arrowprops=dict(arrowstyle="->",lw=1), fontsize=8.5)
plt.savefig(PF+"F_data_shaping_lambda.png", dpi=160, bbox_inches="tight"); plt.close()
print("F_data_shaping_lambda.png ✓ (2-panel, 新小节核心图)")
