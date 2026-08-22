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
cells = S['cells']
F_COL = {0.0: '#c6dbef', 0.25: '#6baed6', 0.50: '#2171b5', 1.0: '#08306b'}
SEED_MK = {0: 'o', 1: '^', 2: 's'}
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})
C = sorted(cells.values(), key=lambda c: (c['f'], c['rho']))

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=300)

#
ax = axes[0]; panel(ax, 'a')
JIT = {0: -0.18, 1: 0.0, 2: 0.18}
for i, c in enumerate(C):
    for run in sorted(c['runs'], key=lambda r: r['seed']):
        ax.scatter(i + JIT[run['seed']], run['dlam2'], s=34, marker=SEED_MK[run['seed']],
                   color=F_COL[c['f']], edgecolor='#333', linewidth=0.5, zorder=3)
ax.set_xticks(range(len(C)))
ax.set_xticklabels([f"f{int(c['f']*100)}\nρ{c['rho']}" for c in C], fontsize=6.2)
ax.set_ylabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$)')
leg = [Line2D([0], [0], marker=SEED_MK[s_], ls='', ms=7, mfc='0.82', mec='#333',
              label=f'seed {s_}') for s_ in (0, 1, 2)]
ax.legend(handles=leg, fontsize=8, loc='upper left', framealpha=0.92)
spines(ax)

#
ax = axes[1]; panel(ax, 'b')
by_seed = {s_: {} for s_ in (0, 1, 2)}
for c in C:
    key = (c['f'], c['rho'])
    for run in c['runs']:
        by_seed[run['seed']][key] = run['dlam2']
keys = sorted(by_seed[0])
x0 = np.array([by_seed[0][k] for k in keys])
lo, hi = x0.min() * 0.9, x0.max() * 1.05
ax.plot([lo, hi], [lo, hi], '--', color='0.7', lw=1, zorder=1)
rel_all = []
for s_, mk in ((1, '^'), (2, 's')):
    xs = np.array([by_seed[s_][k] for k in keys])
    rel_all += list(np.abs(xs - x0) / ((xs + x0) / 2) * 100)
    ax.scatter(x0, xs, marker=mk, s=40, color=[F_COL[k[0]] for k in keys],
               edgecolor='#333', linewidth=0.5, zorder=3, label=f'seed {s_} vs seed 0')
ax.set_xlabel(r'$\Delta\lambda^2$ (seed 0)'); ax.set_ylabel(r'$\Delta\lambda^2$ (seed 1 / seed 2)')
ax.legend(fontsize=8, loc='upper left', framealpha=0.92)
ax.text(0.97, 0.05, f'mean |rel. diff| = {np.mean(rel_all):.1f}%\nmax = {np.max(rel_all):.1f}% (f100·ρ1: near-zero base,\nabsolute diff only 1.9×10$^{-6}$)',
        transform=ax.transAxes, ha='right', fontsize=8, color='0.3')
spines(ax)

#
ax = axes[2]; panel(ax, 'c')
devs = []
for i, c in enumerate(C):
    for run in c['runs']:
        d = (run['dlam2'] - c['dlam2_mean']) * 100
        devs.append(abs(d))
        ax.scatter(i + JIT[run['seed']], d, s=22, marker=SEED_MK[run['seed']],
                   color=F_COL[c['f']], edgecolor='#333', linewidth=0.4, zorder=3)
    ax.errorbar(i, 0, yerr=c['dlam2_sd'] * 100, fmt='none', ecolor='0.45',
                elinewidth=0.8, capsize=2.5, zorder=2)
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xticks(range(len(C)))
ax.set_xticklabels([f"f{int(c['f']*100)}\nρ{c['rho']}" for c in C], fontsize=6.2)
ax.set_ylabel(r'deviation from cell mean  ($\times 10^{-6}$)')
ax.text(0.02, 0.95, f'all 48 runs; median |dev| = {np.median(devs):.1f}×10$^{{-6}}$'
        '\n(cell-to-cell effects are 100–2500× larger)',
        transform=ax.transAxes, ha='left', va='top', fontsize=7.5, color='0.3')
spines(ax)

fig.tight_layout()
out = P4 / 'figures/P4_Fs1_seed_reference.png'
fig.savefig(out); print('saved', out)
