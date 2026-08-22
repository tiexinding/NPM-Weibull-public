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
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10})

CCOL = {'f000_r01': '#1a7837', 'f100_r64': '#b2182b'}
ETA_LS = {'e3e4': '-', 'e15e4': '--'}
ETA_LAB = {'e3e4': 'η = 3e-4', 'e15e4': 'η/2 = 1.5e-4'}
CLAB = {'f000_r01': 'clean (f=0, ρ=1)', 'f100_r64': 'noise ×64 (f=100%, ρ=64)'}

def spines(ax):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5); ax.set_axisbelow(True)

def panel(ax, letter):
    ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight='bold')

def spline(run_root, tag):
    return [json.loads(l) for l in open(P4 / run_root / tag / 'v1b_spline_true.jsonl')]

def lam_traj(run_root, tag):
    tr = json.load(open(P4 / run_root / tag / 'lambda_trajectory_continue.json'))
    st = np.array([e['step'] for e in tr]); lam = np.array([e['lambda'] for e in tr])
    return st, (lam**2 - lam[0]**2) * 1e4

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=300)

# ---------- (a) recorded η(t): constant pair vs cosine-32k pair ----------
ax = axes[0]; panel(ax, 'a')
for et, col in (('e3e4', '0.15'), ('e15e4', '0.55')):
    rows = spline('runs_d1', f'p4d1_f000_r01_const_{et}')
    st = np.array([r['step'] for r in rows]) / 1000.0
    ax.plot(st, [r['eta'] * 1e4 for r in rows], '-', color=col, lw=1.6,
            label=f'constant, {ETA_LAB[et]}')
    rows = spline('runs_32k', f'p4grid_f000_r01_32k_{et}')
    st = np.array([r['step'] for r in rows]) / 1000.0
    ax.plot(st, [r['eta'] * 1e4 for r in rows], '--', color=col, lw=1.3,
            label=f'cosine 32k, {ETA_LAB[et]}')
ax.set_xlabel('step (×1000)'); ax.set_ylabel('recorded η(t)  (×10⁻⁴)')
ax.set_xlim(0, 33); ax.set_ylim(0, 3.3)
ax.legend(fontsize=7.5, loc='upper right', frameon=False)
ax.text(1.1, 0.18, 'warmup', fontsize=7, color='0.45')
ax.text(24.5, 0.62, 'cosine decays to 0.1η —\nthe Fig. S4 stretch confound', fontsize=7, color='0.45', ha='center')
ax.text(21, 1.72, 'constant η: flat after warmup\n(schedule independent of T)', fontsize=7, color='0.3', ha='center')
spines(ax)

# ---------- (b) raw Δλ²(t), 4 runs ----------
ax = axes[1]; panel(ax, 'b')
for cell in CCOL:
    for et in ('e3e4', 'e15e4'):
        st, dl = lam_traj('runs_d1', f'p4d1_{cell}_const_{et}')
        ax.plot(st / 1000.0, dl, ETA_LS[et], color=CCOL[cell], lw=1.4, marker='o', ms=3.4,
                markeredgecolor='#333', markeredgewidth=0.4)
ax.annotate('η', (32.3, 12.54), fontsize=9, color='0.25', va='center')
ax.annotate('η/2', (32.3, 3.68), fontsize=9, color='0.25', va='center')
ax.set_xlabel('step (×1000)'); ax.set_ylabel(r'Δλ²(t) = λ²(t) − λ²(0)   (×10⁻⁴)')
ax.set_xlim(0, 35.5); ax.set_ylim(-0.3, 13.5)
handles = [Line2D([], [], color=CCOL['f000_r01'], lw=1.6, label=CLAB['f000_r01']),
           Line2D([], [], color=CCOL['f100_r64'], lw=1.6, label=CLAB['f100_r64']),
           Line2D([], [], color='0.35', lw=1.4, ls='-', label='η = 3e-4'),
           Line2D([], [], color='0.35', lw=1.4, ls='--', label='η/2 = 1.5e-4')]
ax.legend(handles=handles, fontsize=7.5, loc='upper left', frameon=False)
ax.text(19, 5.0, 'no plateau at either η:\nboth cells keep growing\n(cf. 8k cosine plateau = schedule floor)',
        fontsize=7, color='0.45', ha='center')
spines(ax)

# ---------- (c) high-resolution Δ(RMS²)(t), 137 pts, log-x ----------
ax = axes[2]; panel(ax, 'c')
for cell in CCOL:
    for et in ('e3e4', 'e15e4'):
        rows = spline('runs_d1', f'p4d1_{cell}_const_{et}')
        st = np.array([r['step'] for r in rows])
        rms = np.array([r['rms'] for r in rows])
        ax.plot(st, (rms**2 - rms[0]**2) * 1e4, ETA_LS[et], color=CCOL[cell], lw=1.3)
ax.set_xscale('log')
ax.axvspan(8000, 32000, color='#deebf7', alpha=0.45, zorder=0)
ax.text(16000, 0.55, 'verdict\nwindow', fontsize=7, color='#2171b5', ha='center')
ax.axvline(500, color='0.6', lw=0.7, ls=':')
ax.text(540, 10.6, 'warmup end', fontsize=6.5, color='0.45', rotation=90, va='top')
ax.set_xlabel('step (log)'); ax.set_ylabel(r'Δ(RMS²)(t)   (×10⁻⁴, 137-pt full resolution)')
ax.set_xlim(1, 40000); ax.set_ylim(-0.3, 17.5)
ax.text(60, 13.5, 'smooth throughout —\nno kinks between λ anchors;\nnoise transient = early-time\nshape, not an artifact', fontsize=7, color='0.45')
spines(ax)

fig.tight_layout()
out = FIG / 'P4_Fs7_d1_raw_curves.png'
fig.savefig(out, bbox_inches='tight')
print('saved', out)
