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
FIG = P4 / 'figures'
S = json.load(open(P4 / 'data/p4_phase1_summary.json'))
D1 = S['d1_const_eta']
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})
CCOL = {'f000_r01': '#1a7837', 'f100_r64': '#b2182b'}
ETA_LS = {'e3e4': '-', 'e15e4': '--'}
CLAB = {'f000_r01': 'clean (f=0, ρ=1)', 'f100_r64': 'noise ×64 (f=100%, ρ=64)'}

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

def lam_traj(tag):
    tr = json.load(open(P4 / 'runs_d1' / tag / 'lambda_trajectory_continue.json'))
    st = np.array([e['step'] for e in tr]); lam = np.array([e['lambda'] for e in tr])
    return st, (lam**2 - lam[0]**2) * 1e4

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=300)

# ---------- (a) raw Δλ²(t), 4 runs ----------
ax = axes[0]; panel(ax, 'a')
for cell in CCOL:
    for et in ('e3e4', 'e15e4'):
        st, dl = lam_traj(f'p4d1_{cell}_const_{et}')
        ax.plot(st / 1000.0, dl, ETA_LS[et], color=CCOL[cell], lw=1.4, marker='o', ms=3.2,
                markeredgecolor='#333', markeredgewidth=0.4)
ax.annotate('η', (32.3, 12.54), fontsize=9, color='0.25', va='center')
ax.annotate('η/2', (32.3, 3.68), fontsize=9, color='0.25', va='center')
ax.axvline(20, color='0.6', lw=0.7, ls=':')
ax.text(20.4, 12.9, 'crossing (η): 20k', fontsize=7, color='0.4')
ax.set_xlabel('step (×1000)'); ax.set_ylabel(r'Δλ²(t)   (×10⁻⁴)')
ax.set_xlim(0, 35.5); ax.set_ylim(-0.3, 13.5)
handles = [Line2D([], [], color=CCOL['f000_r01'], lw=1.6, label=CLAB['f000_r01']),
           Line2D([], [], color=CCOL['f100_r64'], lw=1.6, label=CLAB['f100_r64']),
           Line2D([], [], color='0.35', lw=1.4, ls='-', label='η = 3e-4'),
           Line2D([], [], color='0.35', lw=1.4, ls='--', label='η/2 = 1.5e-4')]
ax.legend(handles=handles, fontsize=7.5, loc='upper left', frameon=False)
spines(ax)

# ---------- (b) phase-matched p_sync vs same-t p_eff ----------
ax = axes[1]; panel(ax, 'b')
for p in D1['pairs']:
    cell = p['cell']
    ps = p['phase_sync']['p_sync']
    ts = np.array(sorted(int(t) for t in ps)) / 1000.0
    lab_s = ('clean' if cell == 'f000_r01' else 'noise') + r' $p_{\mathrm{sync}}$ (phase-matched)'
    ax.plot(ts, [ps[str(int(t * 1000))] for t in ts], 'o-', color=CCOL[cell], lw=1.6,
            ms=4.5, markeredgecolor='#333', markeredgewidth=0.5, zorder=4, label=lab_s)
    pe = p['peff_t']
    tt = np.array([t for t in sorted(int(k) for k in pe) if t <= 16000]) / 1000.0
    lab_f = ('clean' if cell == 'f000_r01' else 'noise') + r' $p_{\mathrm{eff}}$ (same-$t$)'
    ax.plot(tt, [pe[str(int(t * 1000))] for t in tt], 'o--', color=CCOL[cell], lw=1.1,
            ms=3, alpha=0.45, zorder=3, label=lab_f)
ax.axhline(1.0, color='0.35', lw=1.1, ls=':', label=r'equilibrium $p = 1$')
ax.legend(fontsize=6.8, loc='lower right', framealpha=0.9)
ax.set_xlabel('phase time t (×1000)')
ax.set_ylabel('η-scaling exponent')
ax.set_xlim(0, 17); ax.set_ylim(-0.15, 2.35)
spines(ax)

# ---------- (c) crossing phase synchrony ----------
ax = axes[2]; panel(ax, 'c')
cr = D1['crossing']
te = np.array(sorted(int(t) for t in cr['diff_eta'])) / 1000.0
de = [cr['diff_eta'][str(int(t * 1000))] for t in te]
th = np.array(sorted(int(t) for t in cr['diff_eta_half_2t'])) / 1000.0
dh = [cr['diff_eta_half_2t'][str(int(t * 1000))] for t in th]
ax.axhline(0, color='0.5', lw=0.9)
ax.plot(te, de, 'o-', color='0.2', lw=1.6, ms=4.2, markeredgecolor='#333',
        markeredgewidth=0.4, label='η:  clean−noise at t')
ax.plot(th, dh, 's--', color='0.55', lw=1.4, ms=4.2, markeredgecolor='#333',
        markeredgewidth=0.4, label='η/2:  at 2t (plotted at phase t)')
ax.set_xlabel('phase time t (×1000)')
ax.set_ylabel(r'clean − noise:  ΔΔλ²(t)   (×10⁻⁴)')
ax.set_xlim(0, 33.5); ax.set_ylim(-1.6, 0.55)
ax.legend(fontsize=7.5, loc='upper center', frameon=False)
spines(ax)

fig.tight_layout()
out = FIG / 'P4_F7_eta_phase.png'
fig.savefig(out, bbox_inches='tight')
print('saved', out)
