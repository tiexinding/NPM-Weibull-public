# -*- coding: utf-8 -*-
"""
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('NPM_ROOT', '.'))
P4 = ROOT / '50_data/05_paper4'
REF = ROOT / '50_data/03_paper/Data_Predictability_WeightScale/derived_data'

grid = json.load(open(P4 / 'data/cloud_eval/p4_grid_eval.json'))
side = json.load(open(REF / 'data_side_stats.json'))

#
p3 = []
for rw in ['000', '010', '020', '030', '040', '050', '060']:
    f_ = REF / f'_tmp_arch/scratch_pythia_rho{rw}_s0/lambda_trajectory_continue.json'
    tr = json.load(open(f_))
    lam0, lamf = tr[0]['lambda'], tr[-1]['lambda']
    D = side['corruption_levels'][f'{int(rw)/100:.2f}']['D']
    p3.append(dict(rho_w=int(rw) / 100, D=D, dlam2=(lamf**2 - lam0**2) * 1e4))
print('===== Q1: Paper#3 from-scratch reference (window shuffle, wikitext_30M tiled to 98M, implicit repetition ~3.3x) =====')
for r in p3:
    print(f"  rho_w={r['rho_w']:.1f}  D={r['D']:.3f}  Δλ²={r['dlam2']:+.3f}")

print('\n===== Q1: this grid ρ=1 and rho=4 rows (f shuffle) =====')
for rho in (1, 4):
    row = sorted([g for g in grid if g['rho'] == rho], key=lambda g: g['f'])
    for g in row:
        print(f"  ρ={rho:>2} f={g['f']:.2f}  D={g['D']:.3f}  Δλ²={g['dlam2']:+.3f}")

#
def slope(pts):
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return np.polyfit(x, y, 1)[0]
p3_seg = [(r['D'], r['dlam2']) for r in p3 if r['D'] < 7.75]
g1_seg = [(g['D'], g['dlam2']) for g in grid if g['rho'] == 1 and g['D'] < 7.75]
g4_seg = [(g['D'], g['dlam2']) for g in grid if g['rho'] == 4 and g['D'] < 7.75]
print(f"\n  unsaturated segment dΔλ²/dD:  Paper#3={slope(p3_seg):+.3f}  this gridρ=1={slope(g1_seg):+.3f}  ρ=4={slope(g4_seg):+.3f}")

#
def mine(tag):
    rows = [json.loads(l) for l in open(P4 / f'runs/{tag}/v1b_spline_true.jsonl')]
    steps = np.array([r['step'] for r in rows]); loss = np.array([r['loss'] for r in rows])
    t_mem = next((int(s) for s, l in zip(steps, loss) if l < 1.0), None)
    #
    alive = float((loss > 1.0).mean())
    return t_mem, alive, float(loss[-1])

print('\n===== Q3/Q4: Δλ² vs ρ per f + memorization time t_mem (first step with train loss < 1.0) =====')
print(f"{'':>8}" + ''.join(f"{'ρ='+str(r):>18}" for r in (1, 4, 16, 64)))
mined = {}
for f_ in (0.0, 0.25, 0.50, 1.0):
    cells = {g['rho']: g for g in grid if g['f'] == f_}
    line1 = f"f={f_:.2f}  "; line2 = ' ' * 8
    for rho in (1, 4, 16, 64):
        g = cells[rho]
        t_mem, alive, lf = mine(g['name'])
        mined[g['name']] = (t_mem, alive)
        line1 += f"Δλ²={g['dlam2']:+.2f}        "
        line2 += f"t_mem={str(t_mem) if t_mem else '—':>5} survival{alive*100:3.0f}%  "
    print(line1); print(line2)

print('\n  [Q4 criterion] high-rho cells: harder memorization (later t_mem / longer gradient survival) -> larger dlam2?')
hi = [(g['name'], g['f'], g['dlam2'], mined[g['name']][1]) for g in grid if g['rho'] >= 16]
xa = np.array([h[3] for h in hi]); ya = np.array([h[2] for h in hi])
print(f"  ρ≥16 8 cells: corr(gradient-survival fraction, dlam2) = {np.corrcoef(xa, ya)[0,1]:+.3f}")
for h in sorted(hi, key=lambda x: -x[2]):
    print(f"    {h[0][7:-3]:>12}  Δλ²={h[2]:+.2f}  survival={h[3]*100:3.0f}%")

#
print('\n===== Q5: token budget check =====')
import os
sizes = {p.name: p.stat().st_size for p in (P4 / 'token_data/p4grid').glob('p4grid_*.npy')}
u = set(sizes.values())
print(f"  16-cell file byte-size set = {u} (uint16: (bytes-128)/2 tokens)")
print(f"  → tokens per cell = {(list(u)[0]-128)//2:,} (= 8000 steps × bs24 × seq512 = {8000*24*512:,}) constant OK")

#
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=150)
F_COL = {0.0: '#c6dbef', 0.25: '#6baed6', 0.50: '#2171b5', 1.0: '#08306b'}

ax = axes[0]
ax.plot([r['D'] for r in p3], [r['dlam2'] for r in p3], 's--', color='#b2182b', ms=7,
        lw=1.4, label='Paper#3 scratch (window-shuffle, implicit rep≈3.3×)', zorder=3)
for rho, mk, c in ((1, 'o', '#4d4d4d'), (4, '^', '#1a7837')):
    row = sorted([g for g in grid if g['rho'] == rho], key=lambda g: g['D'])
    ax.plot([g['D'] for g in row], [g['dlam2'] for g in row], mk + '-', color=c, ms=8,
            lw=1.4, label=f'P4 grid ρ={rho} (f-shuffle)', zorder=3)
ax.set_xlabel('D  (bigram conditional entropy, bits)')
ax.set_ylabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$)')
ax.legend(fontsize=8.5, loc='lower left'); ax.grid(alpha=0.25); ax.set_axisbelow(True)
ax.text(0.02, 0.62, 'Q1: D-driven decline reproduces;\nρ=4 row ≈ Paper#3 (matched repetition);\nsplit at D≈7.9 = bigram-D saturation\n(window-shuffle keeps 40% structure)',
        transform=ax.transAxes, va='top', fontsize=8.5, color='0.35', style='italic')

ax = axes[1]
for f_ in (0.0, 0.25, 0.50, 1.0):
    row = sorted([g for g in grid if g['f'] == f_], key=lambda g: g['rho'])
    ax.plot([g['rho'] for g in row], [g['dlam2'] for g in row], 'o-', color=F_COL[f_],
            ms=8, lw=1.5, markeredgecolor='#333', markeredgewidth=0.7,
            label=f'f = {int(f_*100)}%')
ax.set_xscale('log', base=2); ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(['1', '4', '16', '64'])
ax.set_xlabel('repetition ρ  (log₂)')
ax.set_ylabel(r'$\Delta\lambda^2$  ($\times 10^{-4}$)')
ax.legend(fontsize=8.5, title='shuffle f', title_fontsize=8.5, loc='lower right')
ax.grid(alpha=0.25); ax.set_axisbelow(True)
ax.text(0.02, 0.97, 'Q3/Q4: f=0 inverted-U (easy → gradient dies);\nf=100% monotone (rote noise = longest effort)',
        transform=ax.transAxes, va='top', fontsize=8.5, color='0.35', style='italic')
for a in axes:
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
fig.text(0.995, 0.01, 'Pythia-70m from-scratch, seed 0 (single-seed screening)',
         ha='right', fontsize=7.5, color='0.55')
fig.tight_layout()
out = P4 / 'figures/p4_16grid_deepdive.png'
fig.savefig(out, dpi=150)
print('\nsaved', out)
