import os
#!/usr/bin/env python
"""


    R = 1 − unique_windows/total_windows)



"""
import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('NPM_ROOT', '.'))
BASE_NPY = ROOT / '40_Dynamics/sims/data/wikitext103_full_pythia_tokens.npy'
P4 = ROOT / '50_data/05_paper4'
TOK_OUT = P4 / 'token_data/p4grid'
GATE_OUT = P4 / 'gate_out'

T_FULL = 8000 * 24 * 512
FS = [0.0, 0.25, 0.50, 1.0]
RHOS = [1, 4, 16, 64]
KGRAM = 32
DN = 2_000_000
DBLOCK = 4_000_000
HELD_START = 110_000_000
HELD_N = 128_000
SEED0 = 20260702


def ftag(f):
    return f'f{int(round(f * 100)):03d}'


#
def Dbi(t, n=DN):
    t = t[:n].astype(np.int64)
    V = int(t.max()) + 1
    pr, nx = t[:-1], t[1:]
    def H(x):
        _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    return float(H(pr * V + nx) - H(pr))


#
def r_novelty_vec(stream, k=KGRAM):
    n = len(stream)
    if n <= k:
        return 0.0
    s = stream.astype(np.uint64) + np.uint64(1)
    BASE = np.uint64(1099511628211)
    nw = n - k + 1
    with np.errstate(over='ignore'):
        h = s[:nw].copy()
        for j in range(1, k):                 # Horner: h = h*B + s[j:j+nw]
            h *= BASE
            h += s[j:j + nw]
    uniq = len(np.unique(h))
    return 1.0 - uniq / nw


#
def partial_shuffle(block, f, rng):
    if f <= 0:
        return block
    if f >= 1.0:
        return rng.permutation(block)
    sel = np.nonzero(rng.random(len(block), dtype=np.float32) < f)[0]
    out = block.copy()
    out[sel] = out[sel][rng.permutation(len(sel))]
    return out


def tile_to(block, total):
    reps = total // len(block) + 1
    return np.tile(block, reps)[:total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='T=2M smoke test, no corpus written')
    a = ap.parse_args()
    T = 2_000_000 if a.smoke else T_FULL
    write_corpora = not a.smoke
    GATE_OUT.mkdir(exist_ok=True)
    if write_corpora:
        TOK_OUT.mkdir(parents=True, exist_ok=True)

    full = np.load(BASE_NPY, mmap_mode='r')
    print(f'base: {full.shape[0]:,} tokens ({BASE_NPY.name})  T={T:,}  '
          f'{"SMOKE(no write)" if a.smoke else "FULL"}', flush=True)
    assert HELD_START >= T + 5_000_000, 'held-out must be disjoint from training base; nt'

    #
    if write_corpora:
        held_clean = np.asarray(full[HELD_START:HELD_START + HELD_N])
        np.save(TOK_OUT / 'held_clean.npy', held_clean)
        tail_blk = np.asarray(full[HELD_START:HELD_START + DBLOCK])
        for f in FS:
            if f <= 0:
                continue
            rng_h = np.random.default_rng(SEED0 + 900 + int(f * 100))
            hb = partial_shuffle(tail_blk.copy(), f, rng_h)[:HELD_N]
            np.save(TOK_OUT / f'held_{ftag(f)}.npy', hb)
        print(f'held-out saved: held_clean + {[ftag(f) for f in FS if f > 0]} '
              f'(each {HELD_N:,} tok, start {HELD_START:,})', flush=True)

    #
    dblk = np.asarray(full[:DBLOCK])
    D_intr = {}
    for f in FS:
        rng = np.random.default_rng(SEED0 + int(f * 100))
        D_intr[f] = Dbi(partial_shuffle(dblk.copy(), f, rng))
        print(f'  D_intrinsic[{ftag(f)}] = {D_intr[f]:.3f}  (fixed {DBLOCK:,}-tok block)', flush=True)

    #
    rows = []
    for f in FS:
        for rho in RHOS:
            t0 = time.time()
            U = T // rho
            rng = np.random.default_rng(SEED0 + 1000 * rho + int(f * 100))
            blk = partial_shuffle(np.asarray(full[:U]).copy(), f, rng)
            stream = tile_to(blk, T)
            name = f'p4grid_{ftag(f)}_r{rho:02d}_s0'
            if write_corpora:
                np.save(TOK_OUT / f'{name}.npy', stream)
            D_ss = Dbi(blk)
            R = r_novelty_vec(stream)
            del blk, stream
            rows.append(dict(name=name, f=f, rho=rho, U=int(U), T=int(T),
                             D=D_intr[f], D_smallsample=D_ss, R=R))
            print(f'  [{ftag(f)} ρ={rho:>2}] U={U:>10,} D={D_intr[f]:6.3f} '
                  f'(ss {D_ss:5.3f}) R={R:.4f}  {time.time() - t0:.0f}s', flush=True)

    tag = 'smoke' if a.smoke else 'trainlevel'
    mpath = GATE_OUT / f'grid_{tag}_metrics.json'
    with open(mpath, 'w') as fh:
        json.dump(dict(T=T, held_start=HELD_START, held_n=HELD_N, base=str(BASE_NPY),
                       seed0=SEED0, kgram=KGRAM, rows=rows), fh, indent=2)
    print(f'saved {mpath}', flush=True)
    make_figure(rows, GATE_OUT / f'p4_grid_{tag}.png')
    verdict(rows)


def verdict(rows):
    D = np.array([r['D'] for r in rows]); R = np.array([r['R'] for r in rows])
    fv = np.array([r['f'] for r in rows]); rv = np.array([r['rho'] for r in rows])
    dmed, rmed = np.median(D), np.median(R)
    quads = {('hi' if d > dmed else 'lo') + 'D_' + ('hi' if r > rmed else 'lo') + 'R'
             for d, r in zip(D, R)}
    frank = np.array([sorted(FS).index(x) for x in fv])
    rrank = np.array([sorted(RHOS).index(x) for x in rv])
    c_Drho = np.corrcoef(D, rrank)[0, 1]
    c_Rf = np.corrcoef(R, frank)[0, 1]
    c_Df = np.corrcoef(D, frank)[0, 1]
    c_Rrho = np.corrcoef(R, rrank)[0, 1]
    Dlv = [np.mean(D[fv == f]) for f in sorted(set(fv))]
    gaps = np.diff(Dlv)
    print('\n===== train-level GATE verdict =====', flush=True)
    print(f'  quadrant coverage (median split D={dmed:.2f} R={rmed:.2f}): {sorted(quads)}  ({len(quads)}/4)', flush=True)
    print(f'  corr(D, ρ-rank) = {c_Drho:+.2f} (expect ~0) | corr(R, f-rank) = {c_Rf:+.2f} (expect ~0)', flush=True)
    print(f'  corr(D, f-rank) = {c_Df:+.2f} (expect >0) | corr(R, ρ-rank) = {c_Rrho:+.2f} (expect >0)', flush=True)
    print(f'  D gradient (f 4levels): {[f"{d:.3f}" for d in Dlv]}  adjacent spacing {[f"{g:.3f}" for g in gaps]}', flush=True)
    ok_quad = len(quads) == 4
    ok_orth = abs(c_Drho) < 0.1 and abs(c_Rf) < 0.1
    ok_grad = bool(np.all(gaps >= 0.15))
    ok = ok_quad and ok_orth and ok_grad
    print(f'  a) all four quadrants covered: {"✅" if ok_quad else "❌"}   b) decoupled/orthogonal: {"✅" if ok_orth else "❌"}'
          f'   c) D gradient resolution(≥0.15 bits/levels): {"✅" if ok_grad else "❌"}', flush=True)
    print(f'\n  GATE {"PASS ✅ 16 cells ready for training harness" if ok else "FAIL: adjust construction and re-test"}', flush=True)


def make_figure(rows, out):
    f_color = {0.0: '#2ca02c', 0.25: '#9acd32', 0.50: '#ff7f0e', 1.0: '#d62728'}
    rho_size = {1: 60, 4: 130, 16: 240, 64: 380}
    D = np.array([r['D'] for r in rows]); R = np.array([r['R'] for r in rows])
    dmed, rmed = np.median(D), np.median(R)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for r in rows:
        ax.scatter(r['D'], r['R'], s=rho_size[r['rho']], c=f_color[r['f']],
                   edgecolor='k', linewidth=0.6, alpha=0.85, zorder=3)
        ax.annotate(f"{ftag(r['f'])}·ρ{r['rho']}", (r['D'], r['R']),
                    fontsize=6, xytext=(4, 4), textcoords='offset points')
    ax.axvline(dmed, ls='--', color='gray', lw=1); ax.axhline(rmed, ls='--', color='gray', lw=1)
    ax.text(0.02, 0.97, 'Q3 hiD·loR', transform=ax.transAxes, va='top', fontsize=7, color='gray')
    ax.text(0.98, 0.97, 'Q4 hiD·hiR (memorizable noise)', transform=ax.transAxes,
            va='top', ha='right', fontsize=7, color='crimson')
    ax.text(0.02, 0.03, 'Q1 loD·loR', transform=ax.transAxes, va='bottom', fontsize=7, color='gray')
    ax.text(0.98, 0.03, 'Q2 loD·hiR', transform=ax.transAxes, va='bottom', ha='right', fontsize=7, color='gray')
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markeredgecolor='k',
                  label=f'f={f:.2f}', markersize=9) for f, c in f_color.items()]
    ax.legend(handles=leg, title='shuffle fraction f (D knob)', fontsize=8, loc='center left')
    ax.set_xlabel('D = bigram conditional entropy, fixed pre-tiling block (bits)')
    ax.set_ylabel(f'R = ∞-gram novelty coverage on tiled stream (k={KGRAM})')
    ax.set_title('Paper#4 train-level grid: (D,R) decoupling, 4×4\n(marker size = repetition ρ)')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'saved {out}', flush=True)


if __name__ == '__main__':
    main()
