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
cells = S['cells']; FIG = P4 / 'figures'
F_COL = {0.0: '#c6dbef', 0.25: '#6baed6', 0.50: '#2171b5', 1.0: '#08306b'}
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

# ================= F6: BLiMP =================
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=300)
ax = axes[0]; panel(ax, 'a')
for f in (0.0, 0.25, 0.50, 1.0):
    row = [cells[f'f{int(f*100):03d}_r{r:02d}'] for r in (1, 4, 16, 64)]
    ax.errorbar([1, 4, 16, 64], [c['blimp_mean'] for c in row],
                yerr=[c['blimp_sd'] for c in row], fmt='o-', color=F_COL[f], ms=6, lw=1.3,
                capsize=2, markeredgecolor='#333', markeredgewidth=0.6, label=f'f = {int(f*100)}%')
ax.axhline(0.5, color='0.6', ls=':', lw=1)
ax.text(60, 0.503, 'chance', fontsize=7.5, color='0.45', ha='right')
ax.set_xscale('log', base=2); ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(['1', '4', '16', '64'])
ax.set_xlabel('repetition ρ  (log₂)'); ax.set_ylabel('BLiMP accuracy (67 subsets)')
ax.legend(fontsize=8, title='shuffle $f$', title_fontsize=8, loc='upper right')
spines(ax)
ax = axes[1]; panel(ax, 'b')
TH = json.load(open(P4 / 'data/p4_thresholds.json'))
lab = {r['cell']: r['label'] for r in TH['runs'] if r['set'] == 'main' and r['seed'] == 0}
groups = {'LEARNED': [], 'NOTHING': [], 'MEMORIZED': []}
for key, c in cells.items():
    groups[lab[key]].append((c['blimp_mean'], c['blimp_sd'], c['f']))
CL = {'LEARNED': '#1a7837', 'NOTHING': '#7b3294', 'MEMORIZED': '#b2182b'}
rng = np.random.default_rng(3)
for i, g in enumerate(('LEARNED', 'NOTHING', 'MEMORIZED')):
    vals = groups[g]
    xs = i + rng.uniform(-0.13, 0.13, len(vals))
    ax.errorbar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], fmt='none',
                ecolor='0.55', elinewidth=0.7, capsize=1.5, zorder=2)
    ax.scatter(xs, [v[0] for v in vals], s=52, color=[F_COL[v[2]] for v in vals],
               edgecolor=CL[g], linewidth=1.5, zorder=3)
    m = np.mean([v[0] for v in vals])
    ax.hlines(m, i - 0.24, i + 0.24, color=CL[g], lw=2.2, zorder=4)
    ax.text(i - 0.30, m, f'{m:.3f}', ha='right', va='center', fontsize=8.5, color=CL[g], fontweight='bold')
ax.axhline(0.5, color='0.6', ls=':', lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(['LEARNED\n(n=6 cells)', 'NOTHING\n(n=2)', 'MEMORIZED\n(n=8)'], fontsize=8.5)
ax.set_ylabel('BLiMP accuracy')
ax.text(0.98, 0.97, 'fill = shuffle f; edge = regime\nwithin-row (fixed f): MEMORIZED < LEARNED, 16/16',
        transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color='0.3')
spines(ax)
fig.tight_layout(); fig.savefig(FIG / 'P4_F6_blimp.png'); plt.close(fig)

# ================= Fs4: 32k drift falsification =================
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=300)
ax = axes[0]; panel(ax, 'a')
STY = {('f000_r01', '8k'): ('#4d4d4d', '-'), ('f000_r01', '32k'): ('#4d4d4d', '--'),
       ('f100_r64', '8k'): ('#08306b', '-'), ('f100_r64', '32k'): ('#08306b', '--')}
for cell in ('f000_r01', 'f100_r64'):
    tr8 = json.load(open(P4 / f'runs/p4grid_{cell}_s0/lambda_trajectory_continue.json'))
    ax.plot([e['step'] for e in tr8], [(e['lambda']**2 - tr8[0]['lambda']**2)*1e4 for e in tr8],
            STY[(cell, '8k')][1], color=STY[(cell, '8k')][0], lw=1.5,
            label=f"{cell.replace('_', '·').replace('f000·r01','clean ρ1').replace('f100·r64','shuffled ρ64')} (T=8k)")
    tr32 = json.load(open(P4 / f'runs_32k/p4grid_{cell}_32k_e3e4/lambda_trajectory_continue.json'))
    ax.plot([e['step'] for e in tr32], [(e['lambda']**2 - tr32[0]['lambda']**2)*1e4 for e in tr32],
            STY[(cell, '32k')][1], color=STY[(cell, '32k')][0], lw=1.5,
            label=f"same cell (T=32k schedule)")
ax.set_xlabel('step'); ax.set_ylabel(r'$\Delta\lambda^2(t)$  ($\times 10^{-4}$)')
#
ETA_C = '#e08214'
ax2 = ax.twinx()
for tag, ls in (('runs/p4grid_f000_r01_s0', '-'), ('runs_32k/p4grid_f000_r01_32k_e3e4', '--')):
    rows = [json.loads(l) for l in open(P4 / tag / 'v1b_spline_true.jsonl')]
    ax2.plot([r['step'] for r in rows], [r['eta'] * 1e4 for r in rows], ls, color=ETA_C, lw=1.1, alpha=0.8)
ax2.set_ylim(0, 3.3); ax2.set_ylabel(r'learning rate $\eta(t)$  ($\times 10^{-4}$)', color=ETA_C, fontsize=9)
ax2.tick_params(axis='y', labelcolor=ETA_C, labelsize=8)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_color(ETA_C)
ax2.text(8500, 0.42, r'$\eta$(t), T=8k' + '\n(floor by 8k)', fontsize=7, color=ETA_C, va='center')
ax2.text(21500, 1.15, r'$\eta$(t), T=32k (stretched)', fontsize=7, color=ETA_C)
ax.legend(fontsize=7.5, loc='upper left')
ax.text(0.97, 0.42, 'cosine schedule stretches with T:\nthe 32k run re-opens the high-LR\ntransient — Δλ² grows ~3.8× further\n(the 8k "plateau" is partly schedule-made)',
        transform=ax.transAxes, ha='right', fontsize=7.5, color='0.3')
spines(ax)
ax = axes[1]; panel(ax, 'b')
dr = S['drift_32k']
xs = np.arange(len(dr))
w = 0.32
ax.bar(xs - w/2, [d['p_eff_8k'] for d in dr], width=w*0.9, color='#4d4d4d',
       edgecolor='#222', lw=0.5, label='T = 8k')
ax.bar(xs + w/2, [d['p_eff_32k'] for d in dr], width=w*0.9, color='#8c96c6',
       edgecolor='#222', lw=0.5, label='T = 32k (scaled schedule)')
for x_, d in zip(xs, dr):
    ax.text(x_ - w/2, d['p_eff_8k'] + 0.03, f"{d['p_eff_8k']:.2f}", ha='center', fontsize=8)
    ax.text(x_ + w/2, d['p_eff_32k'] + 0.03, f"{d['p_eff_32k']:.2f}", ha='center', fontsize=8)
ax.axhline(1.0, color='#b2182b', ls='--', lw=1.1)
ax.text(1.42, 1.03, 'equilibrium limit p = 1\n(pre-registered drift target)', fontsize=7.5, color='#b2182b', ha='right')
ax.axhline(2.0, color='#b2182b', ls=':', lw=1.0)
ax.text(1.42, 1.92, 'pure-injection limit p = 2', fontsize=7.5, color='#b2182b', ha='right')
ax.set_xticks(xs); ax.set_xticklabels(['clean ρ1', 'shuffled ρ64'], fontsize=9)
ax.set_ylabel(r'effective η-exponent  $p_{eff} = -\log_2$ ratio')
ax.set_ylim(0, 2.25)
ax.legend(fontsize=8, loc='upper left')
ax.text(0.40, 0.985, 'pre-registered: p → 1 at longer T.\nTEST INVALID as run: schedule stretches with T,\nso any decaying schedule pins the ratio to the\nhigh-LR transient (zero power for the prediction).\nByproduct: exponents budget-robust (+5%) under\nscaled schedules. v2 test = constant-η horizons.',
        transform=ax.transAxes, fontsize=7.5, color='0.2', va='top',
        bbox=dict(fc='white', ec='0.8', alpha=0.92))
spines(ax)
fig.tight_layout(); fig.savefig(FIG / 'P4_Fs4_32k_drift.png'); plt.close(fig)
print('saved P4_F6_blimp.png + P4_Fs4_32k_drift.png')
