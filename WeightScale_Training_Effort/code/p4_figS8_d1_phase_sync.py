# -*- coding: utf-8 -*-
"""
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P4 = Path(__file__).resolve().parents[1]
S = json.load(open(P4 / 'data/p4_phase1_summary.json'))
D1 = S['d1_const_eta']
FIG = P4 / 'figures'
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})
CCOL = {'f000_r01': '#1a7837', 'f100_r64': '#b2182b'}
CLAB = {'f000_r01': 'clean (f=0, ρ=1)', 'f100_r64': 'noise ×64 (f=100%, ρ=64)'}

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.13, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=300)

# ---------- (a) phase-matched p_sync vs same-t p_eff ----------
ax = axes[0]; panel(ax, 'a')
for p in D1['pairs']:
    cell = p['cell']
    ps = p['phase_sync']['p_sync']
    ts = np.array(sorted(int(t) for t in ps)) / 1000.0
    ax.plot(ts, [ps[str(int(t * 1000))] for t in ts], 'o-', color=CCOL[cell], lw=1.6,
            ms=4.5, markeredgecolor='#333', markeredgewidth=0.5, zorder=4)
    pe = p['peff_t']
    tt = np.array([t for t in sorted(int(k) for k in pe) if t <= 16000]) / 1000.0
    ax.plot(tt, [pe[str(int(t * 1000))] for t in tt], 'o-', color=CCOL[cell], lw=1.1,
            ms=3, alpha=0.35, zorder=3)
ax.axhline(1.0, color='0.35', lw=1.1, ls=':')
ax.text(15.8, 0.93, 'equilibrium prediction p = 1', fontsize=7.5, color='0.35', ha='right')
ax.text(11, 1.30, 'phase-matched  p_sync(t) =\n−log₂[Δλ²(η/2, 2t) / Δλ²(η, t)]\n(same t/τ_wd; τ_wd = 1/(η·λ_wd))',
        fontsize=7.5, color='0.25', ha='center')
ax.text(11, 1.95, 'same-t ratio p_eff(t) — phase-mismatched\n(the pre-registered test statistic)',
        fontsize=7.5, color='0.55', ha='center')
ax.annotate('noise settles at 1.04 (6k–16k)', (10, 1.041), xytext=(8.5, 0.62),
            textcoords='data', fontsize=7.5, color=CCOL['f100_r64'],
            arrowprops=dict(arrowstyle='-', color=CCOL['f100_r64'], lw=0.7))
ax.annotate('clean rises to 1.06\n(monotone, not yet settled)', (16, 1.059), xytext=(13.4, 0.30),
            textcoords='data', fontsize=7.5, color=CCOL['f000_r01'],
            arrowprops=dict(arrowstyle='-', color=CCOL['f000_r01'], lw=0.7))
handles = [Line2D([], [], color=CCOL['f000_r01'], lw=1.6, marker='o', ms=4, label=CLAB['f000_r01']),
           Line2D([], [], color=CCOL['f100_r64'], lw=1.6, marker='o', ms=4, label=CLAB['f100_r64'])]
ax.legend(handles=handles, fontsize=7.5, loc='upper left', frameon=False)
ax.set_xlabel('phase time t (×1000; η run at t, η/2 run at 2t)')
ax.set_ylabel('η-scaling exponent')
ax.set_xlim(0, 17); ax.set_ylim(-0.15, 2.35)
spines(ax)

# ---------- (b) crossing phase synchrony ----------
ax = axes[1]; panel(ax, 'b')
cr = D1['crossing']
te = np.array(sorted(int(t) for t in cr['diff_eta'])) / 1000.0
de = [cr['diff_eta'][str(int(t * 1000))] for t in te]
th = np.array(sorted(int(t) for t in cr['diff_eta_half_2t'])) / 1000.0
dh = [cr['diff_eta_half_2t'][str(int(t * 1000))] for t in th]
ax.axhline(0, color='0.5', lw=0.9)
ax.plot(te, de, 'o-', color='0.2', lw=1.6, ms=4.2, markeredgecolor='#333',
        markeredgewidth=0.4, label='η:  diff at t')
ax.plot(th, dh, 's--', color='0.55', lw=1.4, ms=4.2, markeredgecolor='#333',
        markeredgewidth=0.4, label='η/2:  diff at 2t (plotted at phase t)')
ax.annotate('η crossing:\nstep 20k (observed)', (20, 0), xytext=(21.5, -0.62),
            fontsize=7.5, color='0.2', ha='center',
            arrowprops=dict(arrowstyle='->', color='0.2', lw=0.8))
ax.annotate('', (16, dh[-1]), xytext=(20, -0.02),
            arrowprops=dict(arrowstyle='-', color='0.55', lw=0.8, ls=':'))
ax.text(24.5, -1.10, 'η/2 tracks the same shape at 2× the time\n(amplitude ×≈0.37, stable 3k–12k)\n→ η/2 crossing predicted ≈ step 40k,\njust beyond the 32k horizon',
        fontsize=7.5, color='0.4', ha='center')
ax.set_xlabel('phase time t (×1000)')
ax.set_ylabel(r'clean − noise:  ΔΔλ²(t)   (×10⁻⁴)')
ax.set_xlim(0, 33.5); ax.set_ylim(-1.6, 0.55)
ax.legend(fontsize=7.5, loc='upper center', frameon=False)
spines(ax)

fig.tight_layout()
out = FIG / 'P4_Fs8_d1_phase_sync.png'
fig.savefig(out, bbox_inches='tight')
print('saved', out)
