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
T = json.load(open(P4 / 'data/p4_thresholds.json'))
runs = T['runs']
TH_G, TH_L = T['theta_gap'], T['theta_lam_rel']
GI, LI = T['robust_interval_gap'], T['robust_interval_lam']

def gmm2(x, iters=200):
    x = np.asarray(x, float)
    mu = np.percentile(x, [25, 75]).astype(float)
    sd = np.array([x.std() / 2 + 1e-6] * 2); pi = np.array([.5, .5])
    for _ in range(iters):
        pdf = np.stack([pi[k]/(sd[k]*np.sqrt(2*np.pi))*np.exp(-.5*((x-mu[k])/sd[k])**2) for k in (0,1)])
        g = pdf/pdf.sum(0, keepdims=True); nk = g.sum(1)
        mu = (g*x).sum(1)/nk; sd = np.sqrt((g*(x-mu[:,None])**2).sum(1)/nk)+1e-9; pi = nk/len(x)
    return mu, sd, pi

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})
fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), dpi=150)
for ax, l in zip(axes, 'abc'):
    ax.text(-0.14, 1.02, l, transform=ax.transAxes, fontsize=13, fontweight='bold')

#
ax = axes[0]
g_main = np.array([r['gap'] for r in runs if r['set'] == 'main'])
z = np.log1p(g_main)
mu, sd, pi = gmm2(z)
ax.hist(z, bins=24, color='#9ecae1', edgecolor='#333', lw=0.4, density=True, alpha=0.85)
xs = np.linspace(z.min()-0.2, z.max()+0.2, 400)
for k, cc, lb in ((0, '#1a7837', 'component 1 (no-memorization)'), (1, '#b2182b', 'component 2 (memorized)')):
    kk = int(np.argsort(mu)[k])
    ax.plot(xs, pi[kk]/(sd[kk]*np.sqrt(2*np.pi))*np.exp(-.5*((xs-mu[kk])/sd[kk])**2), color=cc, lw=1.6, label=lb)
ax.axvline(np.log1p(TH_G), color='k', ls='--', lw=1.2)
ax.axvspan(np.log1p(GI[0]), np.log1p(GI[1]), color='gold', alpha=0.18)
ax.text(np.log1p(TH_G)+0.05, ax.get_ylim()[1]*0.92, f'θ_gap = {TH_G:.2f} nats', fontsize=8.5)
ax.text(np.log1p(1.6), ax.get_ylim()[1]*0.70,
        f'empty interval\n({GI[0]:.2f}, {GI[1]:.2f})\nno data points here:\nany threshold inside\ngives identical labels',
        fontsize=7.5, color='0.25', ha='center')
ax.set_xlabel('log(1 + gap)  [48 main-grid runs]'); ax.set_ylabel('density')
ax.legend(fontsize=7.5, loc='upper right')

#
ax = axes[1]
l_nm = np.array([r['lam_rel'] for r in runs if r['set'] == 'main' and r['gap'] <= TH_G])
z2 = np.log(l_nm)
mu2, sd2, pi2 = gmm2(z2)
ax.hist(z2, bins=18, color='#9ecae1', edgecolor='#333', lw=0.4, density=True, alpha=0.85)
xs2 = np.linspace(z2.min()-0.3, z2.max()+0.3, 400)
for k, cc, lb in ((0, '#7b3294', 'component 1 (weights static)'), (1, '#1a7837', 'component 2 (weights grew)')):
    kk = int(np.argsort(mu2)[k])
    ax.plot(xs2, pi2[kk]/(sd2[kk]*np.sqrt(2*np.pi))*np.exp(-.5*((xs2-mu2[kk])/sd2[kk])**2), color=cc, lw=1.6, label=lb)
ax.axvline(np.log(TH_L), color='k', ls='--', lw=1.2)
ax.axvspan(np.log(LI[0]), np.log(LI[1]), color='gold', alpha=0.18)
ax.text(np.log(TH_L)+0.06, ax.get_ylim()[1]*0.92, f'θ_λrel = {TH_L:.2f}', fontsize=8.5)
ax.set_xlabel('log λ_rel = log(Δλ² / Δλ²_clean-ref)  [24 non-memorized runs]')
ax.set_ylabel('density')
ax.legend(fontsize=7.5, loc='upper left')

#
ax = axes[2]
MK = {'main': 'o', 'c4': 'D', 'code': 's', 'eta15': '^'}
CL = {'LEARNED': '#1a7837', 'MEMORIZED': '#b2182b', 'NOTHING': '#7b3294'}
for r in runs:
    ax.scatter(r['lam_rel'], r['gap'] + 0.03, marker=MK[r['set']], s=34,
               color=CL[r['label']], edgecolor='#333', lw=0.4, alpha=0.9, zorder=3)
ax.set_xscale('log'); ax.set_yscale('log')
ax.axhline(TH_G, color='k', ls='--', lw=1.1)
ax.axhspan(GI[0], GI[1], color='gold', alpha=0.15)
ax.axvline(TH_L, color='k', ls='--', lw=1.1)
ax.axvspan(LI[0], LI[1], color='gold', alpha=0.15)
ax.text(0.05, 8, 'MEMORIZED\n(gap-primary:\napplies at any λ)', fontsize=7.8, color=CL['MEMORIZED'])
ax.text(1.05, 0.06, 'LEARNED', fontsize=8.5, color=CL['LEARNED'])
ax.text(0.045, 0.06, 'NOTHING\nTO LEARN', fontsize=8.5, color=CL['NOTHING'])
leg = ([Line2D([0],[0], marker=MK[k], ls='', ms=6, mfc='0.8', mec='#333', label=k) for k in MK])
ax.legend(handles=leg, fontsize=7, loc='center right', title='run set', title_fontsize=7)
ax.set_xlabel('λ_rel (log)'); ax.set_ylabel('gap + 0.03 nats (log)')
for a_ in axes:
    for s_ in ('top', 'right'): a_.spines[s_].set_visible(False)
    a_.grid(alpha=0.2, lw=0.5); a_.set_axisbelow(True)
fig.text(0.99, 0.01, 'thresholds: 2-component GMM decision boundaries; gold bands = empty intervals (label-invariant zones); 60 runs',
         ha='right', fontsize=7.5, color='0.55')
fig.tight_layout()
fig.savefig(P4 / 'figures/p4_threshold_method.png')
print('saved figures/p4_threshold_method.png')
