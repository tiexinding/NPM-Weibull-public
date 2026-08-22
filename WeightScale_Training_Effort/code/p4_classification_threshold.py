# -*- coding: utf-8 -*-
"""
"""
import json
from pathlib import Path
import numpy as np

P4 = Path(__file__).resolve().parents[1]
S = json.load(open(P4 / 'data/p4_phase1_summary.json'))

#
#
main = []
for key, c in S['cells'].items():
    for r in c['runs']:
        main.append(dict(set='main', cell=key, seed=r['seed'], dlam2=r['dlam2'], gap=r['gap_m']))
ref_main = {r['seed']: r['dlam2'] for r in S['cells']['f000_r01']['runs']}
for r in main:
    r['lam_rel'] = r['dlam2'] / ref_main[r['seed']]

#
#
corners = []
for corp, fn in (('c4', 'p4_corner_c4_eval_s012.json'), ('code', 'p4_corner_code_eval_s012.json')):
    rows = json.load(open(P4 / 'data/cloud_eval' / fn))
    ref = {r['seed']: r['dlam2'] for r in rows if r['f'] == 0.0 and r['rho'] == 4}
    for x in rows:
        corners.append(dict(set=corp, cell=f"f{int(x['f']*100):03d}_r{x['rho']:02d}", seed=x['seed'],
                            dlam2=x['dlam2'], gap=x['held_matched'] - x['train_loss'],
                            lam_rel=x['dlam2'] / ref[x['seed']]))

#
eta = []
ref_eta = next(x['dlam2_eta15'] for x in S['eta_control'] if x['f'] == 0.0 and x['rho'] == 1)
for x in S['eta_control']:
    eta.append(dict(set='eta15', cell=f"f{int(x['f']*100):03d}_r{x['rho']:02d}", seed=0,
                    dlam2=x['dlam2_eta15'], gap=x['gap_m'], lam_rel=x['dlam2_eta15'] / ref_eta))

#
def gmm2_boundary(x, iters=200, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    mu = np.percentile(x, [25, 75]).astype(float)
    sd = np.array([x.std() / 2 + 1e-6] * 2)
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        pdf = np.stack([pi[k] / (sd[k] * np.sqrt(2 * np.pi)) *
                        np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) for k in (0, 1)])
        g = pdf / pdf.sum(0, keepdims=True)
        nk = g.sum(1)
        mu = (g * x).sum(1) / nk
        sd = np.sqrt((g * (x - mu[:, None]) ** 2).sum(1) / nk) + 1e-9
        pi = nk / len(x)
    lo, hi = (0, 1) if mu[0] < mu[1] else (1, 0)
    #
    grid = np.linspace(mu[lo], mu[hi], 2001)
    p_lo = pi[lo] / sd[lo] * np.exp(-0.5 * ((grid - mu[lo]) / sd[lo]) ** 2)
    p_hi = pi[hi] / sd[hi] * np.exp(-0.5 * ((grid - mu[hi]) / sd[hi]) ** 2)
    b = grid[int(np.argmin(np.abs(p_lo - p_hi)))]
    return b, (mu[lo], sd[lo], pi[lo]), (mu[hi], sd[hi], pi[hi])

#
gaps = np.array([r['gap'] for r in main])
bg, glo, ghi = gmm2_boundary(np.log1p(gaps))
TH_GAP = float(np.expm1(bg))
#
lam_nm = np.array([r['lam_rel'] for r in main if r['gap'] <= TH_GAP])
bl, llo, lhi = gmm2_boundary(np.log(lam_nm))
TH_LAM = float(np.exp(bl))

def classify(r):
    if r['gap'] > TH_GAP: return 'MEMORIZED'
    if r['lam_rel'] < TH_LAM: return 'NOTHING'
    return 'LEARNED'

print(f"===== data-driven thresholds (48-run main grid, 2-GMM decision boundary) =====")
print(f"θ_gap = {TH_GAP:.3f} nats (log1p-GMM: low cluster μ={np.expm1(glo[0]):.2f}, high cluster μ={np.expm1(ghi[0]):.2f})")
print(f"θ_λrel = {TH_LAM:.3f}      (log-GMM on non-memorized {len(lam_nm)} run: low cluster μ={np.exp(llo[0]):.3f}, high cluster μ={np.exp(lhi[0]):.3f})")

#
all_runs = main + corners + eta
for r in all_runs:
    r['label'] = classify(r)
for r in all_runs:
    r['borderline'] = False

print(f"\n===== per-cell labels (main grid 16 cells x 3 seeds consistency) =====")
incons = 0
for key in sorted(S['cells']):
    labs = [r['label'] for r in main if r['cell'] == key]
    mark = '' if len(set(labs)) == 1 else ' 🔴 seed-inconsistent'
    if len(set(labs)) > 1: incons += 1
    bl_ = any(r['borderline'] for r in main if r['cell'] == key)
    print(f"  {key}: {labs[0]}{mark}{'  (borderline)' if bl_ else ''}")
print(f"  seed-inconsistent cells: {incons}/16")

print(f"\n===== corner cells + η/2 (calibration check) =====")
for r in corners + eta:
    print(f"  [{r['set']:>5}] {r['cell']}: λ_rel={r['lam_rel']:.2f} gap={r['gap']:.2f} → {r['label']}"
          + ('  (borderline)' if r['borderline'] else ''))

#
print(f"\n===== robustness: threshold gap intervals (all {len(all_runs)} run) =====")
g_all = np.array([r['gap'] for r in all_runs]); l_all = np.array([r['lam_rel'] for r in all_runs])
g_lo = g_all[g_all <= TH_GAP].max(); g_hi = g_all[g_all > TH_GAP].min()
l_lo = l_all[l_all < TH_LAM].max(); l_hi = l_all[l_all >= TH_LAM].min()
print(f"  θ_gap ∈ ({g_lo:.2f}, {g_hi:.2f}) nats → identical labels for all runs (width {g_hi/g_lo:.1f}×)")
print(f"  θ_λrel ∈ ({l_lo:.2f}, {l_hi:.2f}) → identical labels for all runs (width {l_hi/l_lo:.1f}×)")
GAP_INT, LAM_INT = (float(g_lo), float(g_hi)), (float(l_lo), float(l_hi))

out = dict(theta_gap=TH_GAP, theta_lam_rel=TH_LAM, robust_interval_gap=GAP_INT, robust_interval_lam=LAM_INT,
           method='2-component 1D GMM decision boundary; gap in log1p space on 48 main runs; '
                  'lam_rel in log space on non-memorized runs; reference cell = cleanest structured '
                  'cell of same corpus & eta (main/eta: f000_r01; corners: f000_r04, +6% bias noted)',
           runs=[{k: r[k] for k in ('set', 'cell', 'seed', 'dlam2', 'gap', 'lam_rel', 'label', 'borderline')}
                 for r in all_runs])
with open(P4 / 'data/p4_thresholds.json', 'w') as fh:
    json.dump(out, fh, indent=1)
print(f"\nsaved data/p4_thresholds.json ({len(all_runs)} runs)")
