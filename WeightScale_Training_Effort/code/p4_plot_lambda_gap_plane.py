# -*- coding: utf-8 -*-
"""P4 16-cell λ+gap classification plane (7-4, B2).

x = Δλ² (weight-scale growth, ×1e-4)   — "how much did weights move"
y = gap_matched (matched held − train) — "is it rote memorization"
color = f (shuffle fraction, sequential blues = D knob)
size  = ρ (repetition, marker area = R knob)

"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P4 = Path(__file__).resolve().parents[1]
rows = json.load(open(P4 / 'data/cloud_eval/p4_grid_eval.json'))

# sequential single-hue (Blues), light→dark with f; dark edge keeps light steps legible
F_COL = {0.0: '#c6dbef', 0.25: '#6baed6', 0.50: '#2171b5', 1.0: '#08306b'}
R_SIZE = {1: 55, 4: 110, 16: 200, 64: 330}

fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=150)

# quadrant guides: gap threshold (memorization) + Δλ² threshold (weights moved)
GAP_TH, DL_TH = 1.5, 1.0
ax.axhline(GAP_TH, color='0.75', lw=1.0, ls='--', zorder=1)
ax.axvline(DL_TH, color='0.75', lw=1.0, ls='--', zorder=1)

# region labels (recessive, corners)
ax.text(2.72, -0.9, 'LEARNED\n(weights grew, no gap)', color='0.45', fontsize=9,
        ha='center', va='bottom', style='italic')
ax.text(1.55, 15.3, 'MEMORIZED\n(weights grew, huge gap)', color='0.45', fontsize=9,
        ha='center', va='top', style='italic')
ax.text(0.42, -0.9, 'NOTHING TO LEARN\n(weights flat)', color='0.45', fontsize=9,
        ha='center', va='bottom', style='italic')

for r in rows:
    ax.scatter(r['dlam2'], r['gap_matched'], s=R_SIZE[r['rho']],
               color=F_COL[r['f']], edgecolor='#333333', linewidth=0.8,
               alpha=0.92, zorder=3)

#
def tag(name):
    return next(x for x in rows if x['name'].startswith(name))
a = tag('p4grid_f000_r01'); ax.annotate('clean, single pass', (a['dlam2'], a['gap_matched']),
        xytext=(a['dlam2'] - 0.55, a['gap_matched'] + 2.1), fontsize=8.5, color='0.15',
        arrowprops=dict(arrowstyle='-', color='0.55', lw=0.8))
b = tag('p4grid_f100_r01'); ax.annotate('pure noise, single pass', (b['dlam2'], b['gap_matched']),
        xytext=(b['dlam2'] + 0.05, b['gap_matched'] + 2.6), fontsize=8.5, color='0.15',
        arrowprops=dict(arrowstyle='-', color='0.55', lw=0.8))
c = tag('p4grid_f100_r64'); ax.annotate('Q4: memorized noise\n(largest Δλ², zero structure)',
        (c['dlam2'], c['gap_matched']),
        xytext=(c['dlam2'] + 0.08, c['gap_matched'] - 2.6), fontsize=8.5, color='0.15',
        ha='right', va='top',
        arrowprops=dict(arrowstyle='-', color='0.55', lw=0.8))

ax.set_xlabel(r'$\Delta\lambda^2$  (weight-scale growth, $\times 10^{-4}$)', fontsize=11)
ax.set_ylabel('generalization gap  (matched held $-$ train, nats)', fontsize=11)
ax.grid(alpha=0.22, lw=0.6)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# two legends: color = f (D knob), size = ρ (R knob)
leg_f = [Line2D([0], [0], marker='o', ls='', markersize=9, markerfacecolor=F_COL[f],
                markeredgecolor='#333333', label=f'f = {int(f*100)}%') for f in sorted(F_COL)]
l1 = ax.legend(handles=leg_f, title='shuffle fraction f\n(structure → D)', fontsize=8.5,
               title_fontsize=8.5, loc='upper left', bbox_to_anchor=(0.005, 0.995),
               framealpha=0.9, borderpad=0.7)
ax.add_artist(l1)
leg_r = [Line2D([0], [0], marker='o', ls='', markerfacecolor='0.82', markeredgecolor='#333333',
                markersize=(R_SIZE[r] ** 0.5) * 0.82, label=f'ρ = {r}') for r in sorted(R_SIZE)]
ax.legend(handles=leg_r, title='repetition ρ\n(→ R)', fontsize=8.5, title_fontsize=8.5,
          loc='upper left', bbox_to_anchor=(0.005, 0.66), framealpha=0.9,
          labelspacing=1.15, borderpad=0.8)

ax.set_xlim(-0.15, 3.75)
ax.set_ylim(-1.6, 16.2)
ax.text(0.995, 0.005, 'Pythia-70m from-scratch, 8000 steps, seed 0 (single-seed screening)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, color='0.55')
fig.tight_layout()
out = P4 / 'figures/p4_lambda_gap_plane.png'
fig.savefig(out, dpi=150)
print('saved', out)
