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
R_SIZE = {1: 45, 4: 90, 16: 160, 64: 260}
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.13, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

C = sorted(cells.values(), key=lambda c: (c['f'], c['rho']))

# ================= F1: classification plane (3-seed) =================
fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=300)
TH = json.load(open(P4 / 'data/p4_thresholds.json'))
ref = cells['f000_r01']['dlam2_mean']
TH_G = TH['theta_gap']; TH_L_ABS = TH['theta_lam_rel'] * ref
g_lo, g_hi = TH['robust_interval_gap']
l_lo, l_hi = TH['robust_interval_lam'][0] * ref, TH['robust_interval_lam'][1] * ref
#
ax.axhspan(g_lo, g_hi, color='#f0d488', alpha=0.28, zorder=0, lw=0)
ax.axvspan(l_lo, l_hi, color='#f0d488', alpha=0.28, zorder=0, lw=0)
ax.axhline(TH_G, color='0.62', lw=0.9, ls='--', zorder=1)
ax.axvline(TH_L_ABS, color='0.62', lw=0.9, ls='--', zorder=1)
for c in C:
    ax.errorbar(c['dlam2_mean'], c['gap_m_mean'], xerr=c['dlam2_sd'], yerr=c['gap_m_sd'],
                fmt='none', ecolor='0.45', elinewidth=0.9, capsize=2, zorder=2)
    ax.scatter(c['dlam2_mean'], c['gap_m_mean'], s=R_SIZE[c['rho']], color=F_COL[c['f']],
               edgecolor='#333', linewidth=0.7, zorder=3)
ax.text(2.72, -1.05, 'LEARNED', color='0.42', fontsize=9.5, ha='center', style='italic')
ax.text(1.62, 15.4, 'MEMORIZED', color='0.42', fontsize=9.5, ha='center', style='italic')
ax.text(0.40, -1.05, 'NOTHING\nTO LEARN', color='0.42', fontsize=9.5, ha='center', va='bottom', style='italic')
for tag, dx, dy, ha in (('f000_r01', -0.06, 1.15, 'right'), ('f100_r01', 0.10, 1.15, 'left'),
                        ('f100_r64', 0.14, -1.15, 'left')):
    c = cells[tag]
    lbl = {'f000_r01': 'clean, single pass', 'f100_r01': 'shuffled, single pass',
           'f100_r64': 'shuffled ×64'}[tag]
    ax.annotate(lbl, (c['dlam2_mean'], c['gap_m_mean']),
                xytext=(c['dlam2_mean'] + dx, c['gap_m_mean'] + dy), fontsize=8, ha=ha,
                arrowprops=dict(arrowstyle='-', color='0.55', lw=0.7))
leg_f = [Line2D([0], [0], marker='o', ls='', ms=8, mfc=F_COL[f], mec='#333',
                label=f'f = {int(f*100)}%') for f in sorted(F_COL)]
l1 = ax.legend(handles=leg_f, title='shuffle fraction $f$', fontsize=8, title_fontsize=8,
               loc='upper left', framealpha=0.92); ax.add_artist(l1)
leg_r = [Line2D([0], [0], marker='o', ls='', mfc='0.82', mec='#333', ms=(R_SIZE[r]**0.5)*0.75,
                label=f'ρ = {r}') for r in sorted(R_SIZE)]
ax.legend(handles=leg_r, title='repetition ρ', fontsize=8, title_fontsize=8,
          loc='upper left', bbox_to_anchor=(0.0, 0.60), framealpha=0.92, labelspacing=1.0)
ax.set_xlabel(r'$\Delta\lambda^2$  (weight-scale growth, $\times 10^{-4}$)')
ax.set_ylabel('generalization gap (matched held $-$ train, nats)')
ax.set_xlim(-0.15, 3.8); ax.set_ylim(-1.7, 16.3)
spines(ax); fig.tight_layout(); fig.savefig(FIG / 'P4_F1_classification_plane.png'); plt.close(fig)

# ================= F2: mechanism (a: identity, b: work per nat) =================
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), dpi=300)
ax = axes[0]; panel(ax, 'a')
al = np.array([c['I_align_mean'] for c in C]); dl = np.array([c['dlam2_mean'] for c in C])
for c in C:
    ax.scatter(c['I_align_mean'], c['dlam2_mean'], s=R_SIZE[c['rho']] * 0.7, color=F_COL[c['f']],
               edgecolor='#333', linewidth=0.6, zorder=3)
b1, b0 = np.polyfit(al, dl, 1)
xs = np.linspace(al.min() * 0.95, al.max() * 1.03, 10)
ax.plot(xs, b1 * xs + b0, '--', color='0.6', lw=1, zorder=1)
ax.set_xlabel(r'accumulated alignment work  $I_{align}$  (a.u.)')
ax.set_ylabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$)')
spines(ax)
ax = axes[1]; panel(ax, 'b')
w = 0.19
for i, f in enumerate((0.0, 0.25, 0.50, 1.0)):
    xs, ys = [], []
    for j, rho in enumerate((1, 4, 16, 64)):
        c = cells[f'f{int(f*100):03d}_r{rho:02d}']
        xs.append(j + (i - 1.5) * w); ys.append(c['work_per_nat'])
    ax.bar(xs, ys, width=w * 0.92, color=F_COL[f], edgecolor='#333', linewidth=0.5,
           label=f'f = {int(f*100)}%')
ax.set_xticks(range(4)); ax.set_xticklabels(['ρ = 1', 'ρ = 4', 'ρ = 16', 'ρ = 64'])
ax.set_ylabel('alignment work per nat of loss reduction (a.u.)')
ax.legend(fontsize=8, title='shuffle $f$', title_fontsize=8, loc='upper right',
          bbox_to_anchor=(1.0, 0.88), framealpha=0.92)
spines(ax); fig.tight_layout(); fig.savefig(FIG / 'P4_F2_alignment_work.png'); plt.close(fig)

# ================= F3: data coordinates (a: D rows + P3, b: S unification) =================
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), dpi=300)
ax = axes[0]; panel(ax, 'a')
p3 = S['paper3_ref']
ax.plot([r['D'] for r in p3], [r['dlam2'] for r in p3], 's--', color='#b2182b', ms=5.5, lw=1.2,
        label='window-shuffle (implicit rep ≈3.3×)', zorder=3)
for rho, mk, cc in ((1, 'o', '#4d4d4d'), (4, '^', '#1a7837')):
    row = [cells[f'f{int(f*100):03d}_r{rho:02d}'] for f in (0.0, 0.25, 0.50, 1.0)]
    ax.errorbar([c['D'] for c in row], [c['dlam2_mean'] for c in row],
                yerr=[c['dlam2_sd'] for c in row], fmt=mk + '-', color=cc, ms=6, lw=1.2,
                capsize=2, label=f'global shuffle, ρ = {rho}', zorder=3)
ax.set_xlabel('$D$  (bigram conditional entropy, bits)')
ax.set_ylabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$)')
ax.legend(fontsize=8, loc='lower left'); spines(ax)
ax = axes[1]; panel(ax, 'b')
RHO_STY = {1: ('#4d4d4d', 'o'), 4: ('#1a7837', '^'), 16: ('#7b3294', 'D'), 64: ('#e08214', 'v')}
for rho, (cc, mk) in RHO_STY.items():
    row = [cells[f'f{int(f*100):03d}_r{rho:02d}'] for f in (0.0, 0.25, 0.50, 1.0)]
    Sx = [c['S'] for c in row]; Dy = [c['dlam2_mean'] for c in row]
    ax.errorbar(Sx, Dy, yerr=[c['dlam2_sd'] for c in row], fmt=mk + '-', color=cc, ms=5.5,
                lw=1.2, capsize=2, label=f'ρ = {rho}', zorder=3)
    if rho == 4:
        Sw, pred, act = 0.16, float(np.interp(0.16, sorted(Sx), [y for _, y in sorted(zip(Sx, Dy))])), 1.675
ax.scatter([Sw], [act], marker='s', s=55, color='#b2182b', zorder=4,
           label='window-shuffle 0.6 (measured)')
ax.scatter([Sw], [pred], marker='x', s=70, color='#b2182b', zorder=4,
           label='S-interp prediction (ρ=4)')
ax.set_xlabel('$S$  (retained intact-bigram fraction)')
ax.set_ylabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$)')
ax.legend(fontsize=7.5, loc='lower right', ncol=2); spines(ax)
fig.tight_layout(); fig.savefig(FIG / 'P4_F3_data_coordinates.png'); plt.close(fig)

print('saved P4_F1..F3 to figures/  (F4/F5: p4_fig45_corpus_eta.py; Fs1: p4_fig_seed_reference.py)')
