# -*- coding: utf-8 -*-
"""
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P4 = Path(__file__).resolve().parents[1]
FIG = P4 / 'figures'
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})

CELLS = [('f000_r01', 'clean ρ1', '#4d4d4d'), ('f000_r64', 'clean ρ64', '#1a7837'),
         ('f100_r01', 'shuffled ρ1', '#c6dbef'), ('f100_r64', 'shuffled ρ64', '#08306b')]
ETA_DIR = {'η': 'runs', 'η/2': 'runs_eta15'}

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=300)
#
ax = axes[0]; panel(ax, 'a')
for cell, lbl, col in CELLS:
    for tag, ls in (('η', '-'), ('η/2', '--')):
        tr = json.load(open(P4 / ETA_DIR[tag] / f'p4grid_{cell}_s0/lambda_trajectory_continue.json'))
        ax.plot([e['step'] for e in tr], [(e['lambda']**2 - tr[0]['lambda']**2) * 1e4 for e in tr],
                ls, color=col, lw=1.5, label=f'{lbl} ({tag})')
ax.set_xlabel('step'); ax.set_ylabel(r'$\Delta\lambda^2(t)$  ($\times 10^{-4}$)')
ax.legend(fontsize=8, loc='upper left', ncol=2,
          title=r'solid: η = 3×10$^{-4}$    dashed: η/2 = 1.5×10$^{-4}$', title_fontsize=8)
ax.text(0.97, 0.46, 'λ² plateaus by ~6k steps at BOTH η\n(last 2k steps: 0.6–1.9% of growth)\n→ the η-ratio is a plateau-level ratio,\nnot a truncation artifact',
        transform=ax.transAxes, ha='right', fontsize=7.5, color='0.3')
spines(ax)
#
ax = axes[1]; panel(ax, 'b')
for cell, lbl, col in CELLS:
    for tag, ls in (('η', '-'), ('η/2', '--')):
        rows = [json.loads(l) for l in open(P4 / ETA_DIR[tag] / f'p4grid_{cell}_s0/v1b_spline_true.jsonl')]
        ax.plot([r['step'] for r in rows], [r['loss'] for r in rows], ls, color=col, lw=1.3)
ax.set_xlabel('step'); ax.set_ylabel('train loss (nats)')
ax.text(0.97, 0.72, 'loss is NOT equally converged at η/2\n(clean: 3.91 vs 3.61; rote: 0.41 vs 0.22)\n→ λ dynamics settle faster than loss\n(consistent with Paper#2 overshoot→relax)',
        transform=ax.transAxes, ha='right', fontsize=7.5, color='0.3')
spines(ax)
fig.tight_layout(); fig.savefig(FIG / 'P4_Fs2_eta_trajectories.png'); plt.close(fig)
print('saved P4_Fs2_eta_trajectories.png')
