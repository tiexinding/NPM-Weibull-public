#!/usr/bin/env python
"""Paper#2 §4 力预算图.
λwd0.01基线自训Pythia. 2-panel: (a)绝对力magnitude (b)份额%. """
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
PF = ""
F = "derived_data/self_train/baseline_force.jsonl"
rows = [json.loads(l) for l in open(F)]
step = np.array([r["step"] for r in rows])
al = np.array([abs(r["true_align_full"]) for r in rows])   # |F_align|
inj = np.array([r["inj_full"] for r in rows])              # F_inj (+)
dec = np.array([r["dec_full"] for r in rows])              # |F_decay|
tot = al + inj + dec
a_sh, i_sh, d_sh = 100*al/tot, 100*inj/tot, 100*dec/tot

fig = plt.figure(figsize=(12,4.6))
gs = gridspec.GridSpec(1, 2, wspace=0.26)

# (a) 绝对力 magnitude (log-y)
axa = fig.add_subplot(gs[0])
axa.semilogy(step, al, "-o", ms=3, c="#d95f0e", label="$|F_{\\mathrm{align}}|=2\\eta|\\langle W,\\hat u\\rangle|$")
axa.semilogy(step, dec, "-s", ms=3, c="#2c7fb8", label="$|F_{\\mathrm{decay}}|=2\\eta\\lambda_{wd}\\Sigma W^2$")
axa.semilogy(step, inj, "-^", ms=3, c="#7fbf7b", label="$F_{\\mathrm{inj}}=\\eta^2\\|\\hat u\\|^2$")
axa.set_xlabel("Training step", fontsize=10.5); axa.set_ylabel("Force magnitude (a.u.)", fontsize=10.5)
axa.legend(fontsize=9, loc="lower right"); axa.tick_params(labelsize=8.5)
axa.set_title("(a) Absolute forces: decay catches up to alignment", fontsize=11, pad=8)

# (b) 份额% (lines)
axb = fig.add_subplot(gs[1])
axb.plot(step, a_sh, "-o", ms=3, c="#d95f0e", label="alignment")
axb.plot(step, d_sh, "-s", ms=3, c="#2c7fb8", label="decay")
axb.plot(step, i_sh, "-^", ms=3, c="#7fbf7b", label="injection")
axb.axhline(50, ls=":", c="gray", alpha=0.6)
axb.set_xlabel("Training step", fontsize=10.5); axb.set_ylabel("Force share (%)", fontsize=10.5)
axb.legend(fontsize=9, loc="upper right"); axb.set_ylim(0,100); axb.tick_params(labelsize=8.5)
axb.set_title("(b) Force budget: alignment 93%$\\to$49%, balanced at saturation", fontsize=11, pad=8)
axb.annotate("rise:\nalign dominates", xy=(2500,90), xytext=(4000,75), fontsize=8.5, color="#d95f0e",
             arrowprops=dict(arrowstyle="->",color="#d95f0e"))
axb.annotate("saturation:\nalign$\\approx$decay\n(net$\\approx$0)", xy=(20000,49), xytext=(13000,30), fontsize=8.5,
             arrowprops=dict(arrowstyle="->"))
plt.savefig(PF+"F_force_budget.png", dpi=160, bbox_inches="tight"); plt.close()
print("F_force_budget.png ✓ (§4 力预算, 2-panel)")
print(f"基线: step{step[0]} align%={a_sh[0]:.1f} → step{step[-1]} align%={a_sh[-1]:.1f}/dec%={d_sh[-1]:.1f}")
