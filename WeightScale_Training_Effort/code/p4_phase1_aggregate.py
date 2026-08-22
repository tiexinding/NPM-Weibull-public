# -*- coding: utf-8 -*-
"""
"""
import json, glob
from pathlib import Path
import numpy as np

P4 = Path(__file__).resolve().parents[1]
EV = P4 / 'data/cloud_eval'

def integrate(run_dir):
    f = run_dir / 'v1b_spline_true.jsonl'
    if not f.exists(): return None
    rows = [json.loads(l) for l in open(f)]
    st = np.array([r['step'] for r in rows]); dt = np.diff(st, prepend=0)
    al = float((-np.array([r['true_align_full'] for r in rows]) * dt).sum())
    ij = float((np.array([r['inj_full'] for r in rows]) * dt).sum())
    de = float((np.array([r['dec_full'] for r in rows]) * dt).sum())
    loss = np.array([r['loss'] for r in rows])
    return dict(I_align=al, I_inj=ij, I_dec=de, dloss=float(loss[0] - loss[-1]),
                max_neg_align=float(max(-r['true_align_full'] for r in rows)))

def lam_info(run_dir):
    f = run_dir / 'lambda_trajectory_continue.json'
    tr = json.load(open(f))
    return dict(lam0=tr[0]['lambda'], lamf=tr[-1]['lambda'], k_final=tr[-1]['k'],
                lam_traj=[(e['step'], e['lambda']) for e in tr])

#
main = json.load(open(EV / 'p4_grid_eval_s012.json'))
names = {r['name'] for r in main}
s0 = [r for r in json.load(open(EV / 'p4_grid_eval.json')) if r['name'] not in names]
main = main + s0
assert len(main) == 48, f'main grid expects 48 runs, got {len(main)}'
runs_extra = {}
for r in main:
    d = P4 / 'runs' / r['name']
    ex = integrate(d)
    if ex: runs_extra[r['name']] = {**ex, **{k: v for k, v in lam_info(d).items() if k != 'lam_traj'}}

cells = {}
for r in main:
    key = f"f{int(r['f']*100):03d}_r{r['rho']:02d}"
    c = cells.setdefault(key, dict(f=r['f'], rho=r['rho'], D=r['D'], R=r['R'], runs=[]))
    ex = runs_extra.get(r['name'], {})
    c['runs'].append(dict(seed=r['seed'], dlam2=r['dlam2'], gap=r['gap'], gap_m=r['gap_matched'],
                          held=r['held_loss'], train=r['train_loss'], **ex))
for key, c in cells.items():
    for q in ('dlam2', 'gap_m', 'held', 'train', 'I_align', 'I_inj', 'I_dec', 'dloss'):
        vals = [x[q] for x in c['runs'] if q in x]
        c[f'{q}_mean'] = float(np.mean(vals)); c[f'{q}_sd'] = float(np.std(vals)); c['n'] = len(vals)
    c['work_per_nat'] = c['I_align_mean'] / c['dloss_mean'] if c['dloss_mean'] else None
    #
    c['S'] = (1 - c['f']) ** 2

#
corners = {}
for corp in ('c4', 'code'):
    rows = json.load(open(EV / f'p4_corner_{corp}_eval.json'))
    corners[corp] = []
    for r in rows:
        d = P4 / f'runs_{corp}' / r['name']
        ex = integrate(d) or {}
        corners[corp].append(dict(f=r['f'], rho=r['rho'], D=r['D'], R=r['R'], dlam2=r['dlam2'],
                                  gap_m=r['gap_matched'], held=r['held_loss'], train=r['train_loss'],
                                  **{k: ex[k] for k in ex if k.startswith('I_')}))

#
eta = []
base16 = {(r['f'], r['rho']): r['dlam2'] for r in json.load(open(EV / 'p4_grid_eval.json'))}
for r in json.load(open(EV / 'p4_eta15_eval.json')):
    eta.append(dict(f=r['f'], rho=r['rho'], dlam2_eta15=r['dlam2'],
                    dlam2_eta30=base16.get((r['f'], r['rho'])),
                    ratio=r['dlam2'] / base16[(r['f'], r['rho'])] if base16.get((r['f'], r['rho'])) else None,
                    gap_m=r['gap_matched'], held=r['held_loss']))

#
REF = P4.parents[0] / '03_paper/Data_Predictability_WeightScale/derived_data'
side = json.load(open(REF / 'data_side_stats.json'))
p3 = []
for rw in ['000', '010', '020', '030', '040', '050', '060']:
    tr = json.load(open(REF / f'_tmp_arch/scratch_pythia_rho{rw}_s0/lambda_trajectory_continue.json'))
    lv = side['corruption_levels'][f'{int(rw)/100:.2f}']
    p3.append(dict(rho_w=int(rw) / 100, D=lv['D'], S=lv['S'],
                   dlam2=(tr[-1]['lambda']**2 - tr[0]['lambda']**2) * 1e4))


# ---------- BLiMP (7-5 P2 session) ----------
blimp = {}
bl_raw = json.load(open(EV / 'blimp_cloud.json'))
import collections
bl_main = collections.defaultdict(list)
for k, v in bl_raw.items():
    parts = k.split('/')
    if parts[0] == 'p4_runs':
        bl_main[parts[1][7:-3]].append(v['overall'])
for cell_key, vals in bl_main.items():
    if cell_key in cells:
        cells[cell_key]['blimp_mean'] = float(np.mean(vals))
        cells[cell_key]['blimp_sd'] = float(np.std(vals))
blimp['corners'] = {k: v['overall'] for k, v in bl_raw.items() if 'p4_runs_c4' in k or 'p4_runs_code' in k}
blimp['eta15'] = {k.split('/')[-1]: v['overall'] for k, v in bl_raw.items() if 'eta15' in k}

#
code3 = collections.defaultdict(list)
for x in json.load(open(EV / 'p4_corner_code_eval_s012.json')):
    code3[(x['f'], x['rho'])].append(x)
corners_code3 = []
for k, xs in sorted(code3.items()):
    corners_code3.append(dict(f=k[0], rho=k[1], n=len(xs),
        D=xs[0]['D'], R=xs[0]['R'],
        dlam2_mean=float(np.mean([x['dlam2'] for x in xs])), dlam2_sd=float(np.std([x['dlam2'] for x in xs])),
        gap_m_mean=float(np.mean([x['gap_matched'] for x in xs])), gap_m_sd=float(np.std([x['gap_matched'] for x in xs]))))

#
def dl32(tag):
    tr = json.load(open(P4 / f'runs_32k/{tag}/lambda_trajectory_continue.json'))
    return (tr[-1]['lambda']**2 - tr[0]['lambda']**2) * 1e4
drift = []
for cell, r8 in (('f000_r01', 0.311), ('f100_r64', 0.344)):
    d_full = dl32(f'p4grid_{cell}_32k_e3e4'); d_half = dl32(f'p4grid_{cell}_32k_e15e4')
    r32 = d_half / d_full
    drift.append(dict(cell=cell, dlam2_eta=d_full, dlam2_eta_half=d_half,
                      ratio_32k=r32, p_eff_32k=float(-np.log2(r32)),
                      ratio_8k=r8, p_eff_8k=float(-np.log2(r8))))

#
corners_c4_3 = []
c4s_p = EV / 'p4_corner_c4_eval_s012.json'
if c4s_p.exists():
    c43 = collections.defaultdict(list)
    for x in json.load(open(c4s_p)):
        c43[(x['f'], x['rho'])].append(x)
    for k, xs in sorted(c43.items()):
        corners_c4_3.append(dict(f=k[0], rho=k[1], n=len(xs),
            D=xs[0]['D'], R=xs[0]['R'],
            dlam2_mean=float(np.mean([x['dlam2'] for x in xs])), dlam2_sd=float(np.std([x['dlam2'] for x in xs])),
            gap_m_mean=float(np.mean([x['gap_matched'] for x in xs])), gap_m_sd=float(np.std([x['gap_matched'] for x in xs])),
            held_mean=float(np.mean([x['held_loss'] for x in xs])), held_sd=float(np.std([x['held_loss'] for x in xs]))))

#
def kendall_tau(y):
    c = 0; t = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            s = y[j] - y[i]
            c += (s > 0) - (s < 0); t += 1
    return c / t if t else None

d1 = None
d1_p = EV / 'd1_peff_traj.json'
if d1_p.exists():
    d1 = json.load(open(d1_p))
    #
    for p in d1['pairs']:
        ks = sorted(int(k) for k in p['peff_t'])
        y = [p['peff_t'][str(k)] for k in ks if k >= 16000]
        p['sens_16k_32k'] = dict(p_eff_16k=y[0], drift_late=round(y[-1] - y[0], 4),
                                 kendall_tau_late=round(kendall_tau(y), 3),
                                 note='post-hoc sensitivity, not pre-registered; excludes warmup recovery segment')
    #
    def _dl_map(tag):
        tr = json.load(open(P4 / 'runs_d1' / tag / 'lambda_trajectory_continue.json'))
        return {e['step']: (e['lambda']**2 - tr[0]['lambda']**2) * 1e4 for e in tr}
    T_SYNC = [1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
    for pr in d1['pairs']:
        hi, lo = _dl_map(pr['tag_full']), _dl_map(pr['tag_half'])
        ratio = {t: lo[2 * t] / hi[t] for t in T_SYNC}
        pr['phase_sync'] = dict(
            t_anchors=T_SYNC,
            collapse_ratio={str(t): round(ratio[t], 4) for t in T_SYNC},
            p_sync={str(t): round(float(-np.log2(ratio[t])), 4) for t in T_SYNC},
            note='post-hoc; eta/2@2t vs eta@t = matched t/tau phase (tau_wd = 1/(eta*lwd))')
    hiC, hiN = _dl_map('p4d1_f000_r01_const_e3e4'), _dl_map('p4d1_f100_r64_const_e3e4')
    loC, loN = _dl_map('p4d1_f000_r01_const_e15e4'), _dl_map('p4d1_f100_r64_const_e15e4')
    d1['crossing'] = dict(
        diff_eta={str(t): round(hiC[t] - hiN[t], 4) for t in T_SYNC + [20000, 24000, 32000]},
        diff_eta_half_2t={str(t): round(loC[2 * t] - loN[2 * t], 4) for t in T_SYNC},
        crossing_eta_step=20000, crossing_eta_half_predicted=40000,
        note='clean-noise raw-curve crossing; eta/2 diff at 2t tracks eta diff at t (~0.37x), crossing predicted ~40k beyond horizon')
    for bl_tag, key in (('blimp_d1_8k', 'blimp_8k'), ('blimp_d1_32k', 'blimp_32k')):
        f = EV / f'{bl_tag}.json'
        if f.exists():
            d1[key] = {k.split('/')[-1]: v['overall'] for k, v in json.load(open(f)).items()}

#
blimp30k_p = EV / 'blimp_32k_30k.json'
if blimp30k_p.exists():
    blimp['cosine_32k_at30k'] = {k.split('/')[-1]: v['overall']
                                 for k, v in json.load(open(blimp30k_p)).items()}

#
ext_p = P4 / 'data/p4_extraction_s0.json'
if ext_p.exists():
    ext = json.load(open(ext_p))
    for k, v in ext.items():
        if k in cells:
            cells[k]['extract_rate'] = v['extract_rate']; cells[k]['verbatim64'] = v['verbatim_full']

out = dict(

    meta=dict(date='2026-07-05', model='Pythia-70m from-scratch', steps=8000, bs=24, seq=512,
              T=98_304_000, eta='3e-4 (main), 1.5e-4 (control)', lwd=0.01, seeds=[0, 1, 2],
              corpus='wikitext103 full (main), c4_30M/code_30M (corners)',
              held='wikitext103[110M:+128k] disjoint + structure-matched held_f{025,050,100}',
              instability_excluded=0, source_evals=['p4_grid_eval_s012.json',
              'p4_corner_c4_eval.json', 'p4_corner_code_eval.json', 'p4_eta15_eval.json',
              'p4_corner_c4_eval_s012.json', 'd1_peff_traj.json', 'blimp_32k_30k.json',
              'blimp_d1_8k.json', 'blimp_d1_32k.json'],
              updated='2026-07-07 (D1 const-eta verdict + E1 blimp@30k + E2 c4 3-seed)'),
    cells=cells, corners=corners, corners_code_3seed=corners_code3,
    corners_c4_3seed=corners_c4_3, eta_control=eta,
    paper3_ref=p3, blimp_extra=blimp, drift_32k=drift, d1_const_eta=d1)
with open(P4 / 'data/p4_phase1_summary.json', 'w') as fh:
    json.dump(out, fh, indent=1)
print(f"cells={len(cells)} (runs={sum(c['n'] for c in cells.values())}), corners={sum(len(v) for v in corners.values())}, eta={len(eta)}, p3_ref={len(p3)}, c4_3seed={len(corners_c4_3)}, d1={'ok' if d1 else 'MISSING'}")
print('saved data/p4_phase1_summary.json')
