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
F_COL = {0.0: '#c6dbef', 1.0: '#08306b'}
CORP_COL = {'wikitext': '#4d4d4d', 'c4': '#1a7837', 'code': '#b2182b'}
CORP_MK = {'wikitext': 'o', 'c4': 'D', 'code': 's'}
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})
CORNERS = [(0.0, 4), (1.0, 4), (0.0, 64), (1.0, 64)]
LBL = [f"f{int(f*100)}·ρ{r}" for f, r in CORNERS]

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.15, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

def corner_data(corp):
    out = {}
    if corp == 'wikitext':
        for f, r in CORNERS:
            c = cells[f'f{int(f*100):03d}_r{r:02d}']
            out[(f, r)] = dict(dlam2=c['dlam2_mean'], sd=c['dlam2_sd'], gap=c['gap_m_mean'], D=c['D'])
    elif corp == 'code' and 'corners_code_3seed' in S:
        for rr in S['corners_code_3seed']:
            out[(rr['f'], rr['rho'])] = dict(dlam2=rr['dlam2_mean'], sd=rr['dlam2_sd'],
                                             gap=rr['gap_m_mean'], D=rr['D'])
    elif corp == 'c4' and 'corners_c4_3seed' in S:
        for rr in S['corners_c4_3seed']:
            out[(rr['f'], rr['rho'])] = dict(dlam2=rr['dlam2_mean'], sd=rr['dlam2_sd'],
                                             gap=rr['gap_m_mean'], D=rr['D'])
    else:
        for rr in S['corners'][corp]:
            out[(rr['f'], rr['rho'])] = dict(dlam2=rr['dlam2'], sd=None, gap=rr['gap_m'], D=rr['D'])
    return out

CD = {corp: corner_data(corp) for corp in ('wikitext', 'c4', 'code')}

# ================= F4: cross-corpus =================
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), dpi=300)
# (a) plane
ax = axes[0]; panel(ax, 'a')
ax.axhline(1.18, color='0.78', lw=0.9, ls='--', zorder=1)
RS = {4: 55, 64: 140}
for corp in CD:
    for (f, r), v in CD[corp].items():
        ax.scatter(v['dlam2'], v['gap'], marker=CORP_MK[corp], s=RS[r], color=F_COL[f],
                   edgecolor=CORP_COL[corp], linewidth=1.4, zorder=3)
leg1 = [Line2D([0], [0], marker=CORP_MK[k], ls='', ms=8, mfc='0.85', mec=CORP_COL[k],
               mew=1.4, label=k) for k in CORP_MK]
leg2 = [Line2D([0], [0], marker='o', ls='', ms=8, mfc=F_COL[f], mec='#333',
               label=f'f = {int(f*100)}%') for f in (0.0, 1.0)]
l1 = ax.legend(handles=leg1, fontsize=8, loc='upper left', title='corpus', title_fontsize=8)
ax.add_artist(l1)
l2 = ax.legend(handles=leg2, fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 0.64))
ax.add_artist(l2)
leg3 = [Line2D([0], [0], marker='o', ls='', mfc='0.85', mec='#333',
               ms=(RS[r] ** 0.5) * 0.78, label=f'ρ = {r}') for r in (4, 64)]
ax.legend(handles=leg3, fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 0.40),
          title='repetition ρ', title_fontsize=8, labelspacing=0.9)
ax.text(2.9, 2.3, 'MEMORIZED ↑', fontsize=8, color='0.45', style='italic')
ax.set_xlabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$)'); ax.set_ylabel('gap (matched, nats)')
spines(ax)
# (b) Δλ² grouped bars
ax = axes[1]; panel(ax, 'b')
w = 0.26
for i, corp in enumerate(('wikitext', 'c4', 'code')):
    xs = np.arange(4) + (i - 1) * w
    ys = [CD[corp][c]['dlam2'] for c in CORNERS]
    err = [CD[corp][c]['sd'] if CD[corp][c]['sd'] else 0 for c in CORNERS]
    ax.bar(xs, ys, width=w * 0.9, color=CORP_COL[corp], alpha=0.82, edgecolor='#222',
           linewidth=0.5, yerr=err, capsize=2, error_kw=dict(lw=0.8), label=corp)
    for x_, c_ in zip(xs, CORNERS):
        if c_[0] == 0.0:
            ax.text(x_, 0.06, f"D\n{CD[corp][c_]['D']:.1f}", ha='center', fontsize=6, color='white')
ax.set_xticks(range(4)); ax.set_xticklabels(LBL, fontsize=8)
ax.set_ylabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$)')
ax.legend(fontsize=8, loc='upper left'); spines(ax)
# (c) gap grouped bars
ax = axes[2]; panel(ax, 'c')
for i, corp in enumerate(('wikitext', 'c4', 'code')):
    xs = np.arange(4) + (i - 1) * w
    ys = [CD[corp][c]['gap'] for c in CORNERS]
    ax.bar(xs, ys, width=w * 0.9, color=CORP_COL[corp], alpha=0.82, edgecolor='#222',
           linewidth=0.5, label=corp)
ax.axhline(1.18, color='0.6', ls='--', lw=0.9)
ax.text(3.45, 1.6, 'θ_gap = 1.18', fontsize=7, color='0.45')
ax.set_xticks(range(4)); ax.set_xticklabels(LBL, fontsize=8)
ax.set_ylabel('gap (matched, nats)')
ax.legend(fontsize=8, loc='upper left'); spines(ax)
fig.tight_layout(); fig.savefig(FIG / 'P4_F4_crosscorpus.png'); plt.close(fig)

# ================= F5: η control =================
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), dpi=300)
eta = {(r['f'], r['rho']): r for r in S['eta_control']}
ETA_CORNERS = [(0.0, 1), (1.0, 1), (0.0, 64), (1.0, 64)]
ELBL = [f"f{int(f*100)}·ρ{r}" for f, r in ETA_CORNERS]
# (a) paired Δλ² (log y)
ax = axes[0]; panel(ax, 'a')
for i, c in enumerate(ETA_CORNERS):
    e = eta[c]
    ax.bar(i - 0.17, e['dlam2_eta30'], width=0.3, color='#4d4d4d', edgecolor='#222', lw=0.5,
           label='η = 3e-4' if i == 0 else None)
    ax.bar(i + 0.17, e['dlam2_eta15'], width=0.3, color='#8c96c6', edgecolor='#222', lw=0.5,
           label='η = 1.5e-4' if i == 0 else None)
ax.set_yscale('log')
ax.set_xticks(range(4)); ax.set_xticklabels(ELBL, fontsize=8)
ax.set_ylabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$, log)')
ax.legend(fontsize=8, loc='upper left'); spines(ax)
# (b) ratios + two scaling refs
ax = axes[1]; panel(ax, 'b')
rats = [eta[c]['ratio'] for c in ETA_CORNERS]
ax.bar(range(4), rats, width=0.5, color='#8c96c6', edgecolor='#222', lw=0.5)
ax.axhline(2 ** -1.55, color='#b2182b', ls='--', lw=1.1)
ax.axhline(2 ** -1.75, color='#b2182b', ls=':', lw=1.1)
ax.text(3.55, 2 ** -1.55 + 0.008, r'$C_0$: $2^{-1.55}$', fontsize=7.5, color='#b2182b', ha='right')
ax.text(3.55, 2 ** -1.75 - 0.028, r'$C_1$: $2^{-1.75}$', fontsize=7.5, color='#b2182b', ha='right')
ax.set_xticks(range(4)); ax.set_xticklabels(ELBL, fontsize=8)
ax.set_ylabel(r'$\Delta\lambda^2(\eta/2)\,/\,\Delta\lambda^2(\eta)$')
ax.set_ylim(0, 0.5); spines(ax)
# (c) gap stability
ax = axes[2]; panel(ax, 'c')
for i, c in enumerate(ETA_CORNERS):
    key = f'f{int(c[0]*100):03d}_r{c[1]:02d}'
    ax.bar(i - 0.17, cells[key]['gap_m_mean'], width=0.3, color='#4d4d4d', edgecolor='#222', lw=0.5,
           label='η = 3e-4' if i == 0 else None)
    ax.bar(i + 0.17, eta[c]['gap_m'], width=0.3, color='#8c96c6', edgecolor='#222', lw=0.5,
           label='η = 1.5e-4' if i == 0 else None)
ax.set_xticks(range(4)); ax.set_xticklabels(ELBL, fontsize=8)
ax.set_ylabel('gap (matched, nats)')
ax.legend(fontsize=8, loc='upper left'); spines(ax)
fig.tight_layout(); fig.savefig(FIG / 'P4_F5_eta_control.png'); plt.close(fig)
print('saved P4_F4_crosscorpus.png + P4_F5_eta_control.png')
