# -*- coding: utf-8 -*-
"""
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / 'data/p4_phase1_summary.json', HERE / 'p4_phase1_summary.json'):
    if cand.exists():
        S = json.load(open(cand)); break
FIG = (HERE.parent / 'figures') if (HERE.parent / 'figures').exists() else HERE
D1 = S['d1_const_eta']; RULE = D1['rule']
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})

CCOL = {'f000_r01': '#1a7837', 'f100_r64': '#b2182b'}   # regime colors: LEARNED / MEMORIZED
CLAB = {'f000_r01': 'clean  (f=0, ρ=1)', 'f100_r64': 'noise ×64  (f=100%, ρ=64)'}

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=300,
                         gridspec_kw={'width_ratios': [1.25, 1.0, 0.95]})

# ---------- (a) p_eff(t) anchors, constant eta ----------
ax = axes[0]; panel(ax, 'a')
ax.axvspan(8, 32, color='#deebf7', alpha=0.55, zorder=0)
ax.text(28.2, 2.32, 'pre-registered window\n8k → 32k', ha='center', va='top',
        fontsize=8, color='#2171b5')
for p in D1['pairs']:
    cell = p['cell']
    st = np.array(sorted(int(k) for k in p['peff_t'])) / 1000.0
    y = np.array([p['peff_t'][str(int(s * 1000))] for s in st])
    ax.plot(st, y, 'o-', color=CCOL[cell], ms=4.5, lw=1.4,
            markeredgecolor='#333', markeredgewidth=0.5)
    ax.annotate(CLAB[cell], (st[-1], y[-1]),
                xytext=(6, 10 if cell == 'f000_r01' else -14), textcoords='offset points',
                fontsize=8.5, color=CCOL[cell], ha='right', fontweight='bold')
    ax.annotate(f"Δp = {p['drift']:+.3f}\nKendall τ = {p['kendall_tau_anchors']:+.2f}",
                (st[-1], y[-1]), xytext=(-58, 26 if cell == 'f000_r01' else -40),
                textcoords='offset points', fontsize=7.5, color=CCOL[cell])
ax.text(4.6, 1.46, 'post-warmup transient', fontsize=7, color='0.45', ha='center')
ax.plot([16, 16], [1.40, 2.22], color='0.45', lw=0.8, ls=':', zorder=1)
ax.text(16.4, 1.435, '16k: post-hoc sensitivity split', fontsize=6.5, color='0.45', ha='left')
ax.text(3.4, 2.355, 'constant η (warmup-only):\npair η = 3×10⁻⁴ vs η/2 = 1.5×10⁻⁴,  λ_wd = 0.05',
        fontsize=7, color='0.35', ha='left', va='top')
ax.set_xlabel('step (×1000)'); ax.set_ylabel(r'$p_{\rm eff}(t)=-\log_2[\Delta\lambda^2(\eta/2)/\Delta\lambda^2(\eta)]$')
ax.set_xlim(0, 34); ax.set_ylim(1.40, 2.38)
spines(ax)

# ---------- (b) drift verdict number line ----------
ax = axes[1]; panel(ax, 'b')
thr_s, thr_r = RULE['thr_confirm'], RULE['thr_refute']
ax.axvspan(-0.16, thr_s, color='#1a7837', alpha=0.10)
ax.axvspan(thr_s, thr_r, color='0.5', alpha=0.12)
ax.axvspan(thr_r, 0.16, color='#b2182b', alpha=0.08)
for x in (thr_s, thr_r):
    ax.axvline(x, color='0.35', lw=0.9, ls='--')
    ax.text(x, 0.10, f'{x:+.2f}', ha='center', fontsize=7, color='0.35')
ax.text(0.0, 2.88, 'verdict: INCONCLUSIVE (cells disagree)', ha='center', va='center',
        fontsize=9, fontweight='bold', color='0.25')
ax.text(-0.095, 2.32, 'SUPPORTED\n(both cells, + τ ≤ −0.5)', ha='center', fontsize=7.2, color='#1a7837')
ax.text((thr_s + thr_r) / 2, 2.50, 'gap', ha='center', fontsize=7.2, color='0.4')
ax.text(0.09, 2.42, 'REFUTED (both cells)', ha='center', fontsize=7.2, color='#b2182b')
ax.axvline(-0.12, color='#1a7837', lw=1.2, ls=':')
ax.text(-0.114, 0.52, 'mechanism\nexpectation\nΔp ≈ −0.12', ha='left', fontsize=7, color='#1a7837')
rows = {'f000_r01': 1.85, 'f100_r64': 1.35}
for p in D1['pairs']:
    cell = p['cell']
    ax.scatter(p['drift'], rows[cell], s=95, color=CCOL[cell], zorder=5,
               edgecolor='#333', linewidth=0.8)
    if cell == 'f000_r01':
        ax.text(p['drift'], rows[cell] + 0.13, f"{p['drift']:+.3f}", ha='center',
                fontsize=8, color=CCOL[cell], fontweight='bold')
    else:
        ax.text(p['drift'] - 0.014, rows[cell], f"{p['drift']:+.3f}", ha='right', va='center',
                fontsize=8, color=CCOL[cell], fontweight='bold')
# post-hoc sensitivity 16k→32k (open diamonds; excludes warmup recovery)
for p in D1['pairs']:
    dl = p['sens_16k_32k']['drift_late']
    ax.scatter(dl, rows[p['cell']] + 0.30, s=68, facecolor='none',
               edgecolor=CCOL[p['cell']], linewidth=1.4, marker='D', zorder=5)
# cosine cross-protocol fingerprint (open squares, values in caption)
for d in S['drift_32k']:
    dp = d['p_eff_32k'] - d['p_eff_8k']
    ax.scatter(dp, rows[d['cell']] - 0.32, s=70, facecolor='none',
               edgecolor=CCOL[d['cell']], linewidth=1.4, marker='s', zorder=5)
h1 = ax.scatter([], [], s=70, color='0.35', edgecolor='#333', linewidth=0.8,
                label='8k→32k, constant η (pre-registered)')
h3 = ax.scatter([], [], s=58, facecolor='none', edgecolor='0.35', linewidth=1.4, marker='D',
                label='16k→32k sensitivity (post-hoc)')
h2 = ax.scatter([], [], s=60, facecolor='none', edgecolor='0.35', linewidth=1.4, marker='s',
                label='cosine (stretch artifact)')
ax.legend(handles=[h1, h3, h2], fontsize=6.8, loc='lower right', frameon=False, handletextpad=0.3)
ax.set_xlim(-0.16, 0.16); ax.set_ylim(0.0, 3.05)
ax.set_yticks([])
ax.set_xlabel(r'drift  Δp = $p_{\rm eff}$(32k) − $p_{\rm eff}$(8k)')
spines(ax); ax.grid(False)

# ---------- (c) BLiMP schedule robustness ----------
ax = axes[2]; panel(ax, 'c')
b_cos = S['blimp_extra']['cosine_32k_at30k']
b_c32 = D1['blimp_32k']; b_c8 = D1['blimp_8k']
order = [('f000_r01', 'e3e4', 'η = 3e-4'), ('f000_r01', 'e15e4', 'η = 1.5e-4'),
         ('f100_r64', 'e3e4', 'η = 3e-4'), ('f100_r64', 'e15e4', 'η = 1.5e-4')]
ys = [3.4, 2.8, 1.6, 1.0]
for (cell, et, elab), y in zip(order, ys):
    c = CCOL[cell]
    v_cos = b_cos[f'p4grid_{cell}_32k_{et}']
    v_cst = b_c32[f'p4d1_{cell}_const_{et}']
    ax.plot([v_cos, v_cst], [y, y], '-', color='0.75', lw=1.2, zorder=2)
    ax.scatter(v_cos, y, s=64, facecolor='none', edgecolor=c, linewidth=1.5, marker='o', zorder=4)
    ax.scatter(v_cst, y, s=64, color=c, edgecolor='#333', linewidth=0.7, marker='o', zorder=4)
    ax.text(0.402, y, elab, fontsize=7.5, va='center', color='0.35')
ax.axvline(0.5, color='0.6', ls=':', lw=1)
ax.text(0.502, 3.85, 'chance', fontsize=7.5, color='0.45')
ax.text(0.635, 3.05, 'clean', color=CCOL['f000_r01'], fontsize=8.5, fontweight='bold', ha='center')
ax.text(0.60, 1.3, 'noise ×64', color=CCOL['f100_r64'], fontsize=8.5, fontweight='bold', ha='center')
ax.scatter([], [], s=64, facecolor='none', edgecolor='0.3', linewidth=1.5, label='cosine @30k')
ax.scatter([], [], s=64, color='0.3', label='constant η @32k')
ax.legend(fontsize=7.5, loc='lower right', frameon=False)
ax.set_xlim(0.40, 0.68); ax.set_ylim(0.5, 4.1)
ax.set_yticks([])
ax.set_xlabel('BLiMP accuracy (67 subsets)')
spines(ax)

fig.tight_layout()
out = FIG / 'P4_Fs6_d1_verdict.png'
fig.savefig(out, bbox_inches='tight')
print('saved', out)
