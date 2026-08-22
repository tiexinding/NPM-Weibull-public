#!/usr/bin/env python
"""



"""
import sys, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLEAN = Path('<SET_PATH>')
OUT = Path(__file__).resolve().parent.parent / 'gate_out'
OUT.mkdir(exist_ok=True)

T = 4_000_000
RHOS = [1, 4, 16]
SEQ = 512
B_BLK = 16
KGRAM = 32
DN = 2_000_000


#
def Dbi(t, n=DN):
    t = t[:n].astype(np.int64)
    V = int(t.max()) + 1
    pr, nx = t[:-1], t[1:]
    def H(x):
        _, c = np.unique(x, return_counts=True); p = c / c.sum(); return -np.sum(p * np.log2(p))
    return float(H(pr * V + nx) - H(pr))


#
def r_novelty(stream, k=KGRAM):
    n = len(stream)
    if n <= k:
        return 0.0
    s = stream.astype(np.int64)
    mask = (1 << 64) - 1
    BASE = 1099511628211
    Bk = pow(BASE, k - 1, 1 << 64)
    h = 0
    for j in range(k):
        h = (h * BASE + int(s[j]) + 1) & mask
    seen = {h}
    covered = 0
    for i in range(k, n):
        h = (((h - ((int(s[i - k]) + 1) * Bk & mask)) & mask) * BASE + int(s[i]) + 1) & mask
        if h in seen:
            covered += 1
        else:
            seen.add(h)
    return covered / (n - k + 1)


#
def rep_bigram_tiled(stream):
    bg = stream[:-1].astype(np.int64) * (int(stream.max()) + 1) + stream[1:].astype(np.int64)
    uniq = len(np.unique(bg))
    return 1.0 - uniq / len(bg)


#
def apply_structure(block, level, rng):
    if level == 'clean':
        return block
    if level == 'fullshuffle':
        return rng.permutation(block)
    if level == 'blkshuf16':
        n = (len(block) // SEQ) * SEQ
        rows = block[:n].reshape(-1, SEQ).copy()
        nblk = SEQ // B_BLK
        for i in range(rows.shape[0]):
            blocks = rows[i].reshape(nblk, B_BLK)
            perm = rng.permutation(nblk)
            rows[i] = blocks[perm].reshape(-1)
        return np.concatenate([rows.reshape(-1), block[n:]])
    raise ValueError(level)


def tile_to(block, total):
    reps = total // len(block) + 1
    return np.tile(block, reps)[:total]


def main():
    clean = np.load(CLEAN)
    print(f'clean base: {clean.size:,} tokens', flush=True)
    structures = ['clean', 'blkshuf16', 'fullshuffle']
    #
    D_intrinsic = {}
    for st in structures:
        rng = np.random.default_rng(20260702 + {'clean':0,'blkshuf16':1,'fullshuffle':2}[st])
        D_intrinsic[st] = Dbi(apply_structure(clean[:T].copy(), st, rng))
        print(f'  D_intrinsic[{st:>11}] = {D_intrinsic[st]:.3f} (on fixed {T:,}-tok block)', flush=True)
    rows = []
    for st in structures:
        for rho in RHOS:
            t0 = time.time()
            U = T // rho
            base = clean[:U].copy()
            rng = np.random.default_rng(20260702 + {'clean':0,'blkshuf16':1,'fullshuffle':2}[st])
            blk = apply_structure(base, st, rng)
            stream = tile_to(blk, T)
            D = D_intrinsic[st]
            D_smallsample = Dbi(blk)
            R = r_novelty(stream)
            R_naive = rep_bigram_tiled(stream)
            rows.append(dict(structure=st, rho=rho, U=int(U),
                             D=D, D_smallsample=D_smallsample, R=R, R_naive_repbigram=R_naive))
            print(f'  [{st:>11} ρ={rho:>2}] U={U:>9,} D={D:6.3f} (small-samp {D_smallsample:5.3f}) '
                  f'R(novelty)={R:.3f} R_naive(repbg)={R_naive:.3f}  {time.time()-t0:.1f}s', flush=True)

    with open(OUT / 'gate_metrics.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'saved {OUT/"gate_metrics.json"}', flush=True)
    make_figure(rows)
    verdict(rows)


def verdict(rows):
    D = np.array([r['D'] for r in rows]); R = np.array([r['R'] for r in rows])
    dmed, rmed = np.median(D), np.median(R)
    quads = set()
    for r in rows:
        q = ('hi' if r['D'] > dmed else 'lo') + 'D_' + ('hi' if r['R'] > rmed else 'lo') + 'R'
        quads.add(q)
    # D ⊥ ρ ? R ⊥ structure ?
    print('\n===== GATE verdict =====', flush=True)
    print(f'  quadrant coverage (median split D={dmed:.2f} R={rmed:.2f}): {sorted(quads)}  ({len(quads)}/4)', flush=True)
    #
    print('  decoupling checks:', flush=True)
    print(f'   corr(D, ρ)        = {np.corrcoef(D,[r["rho"] for r in rows])[0,1]:+.2f}  (expect ~0: D invariant to repetition)', flush=True)
    rho_code = {1:0,4:1,16:2}
    print(f'   corr(R, ρ-rank)   = {np.corrcoef(R,[rho_code[r["rho"]] for r in rows])[0,1]:+.2f}  (expect >0: R rises with repetition)', flush=True)
    st_code = {'clean':0,'blkshuf16':1,'fullshuffle':2}
    print(f'   corr(D, struct)   = {np.corrcoef(D,[st_code[r["structure"]] for r in rows])[0,1]:+.2f}  (expect >0: D rises with shuffling)', flush=True)
    print(f'   corr(R, struct)   = {np.corrcoef(R,[st_code[r["structure"]] for r in rows])[0,1]:+.2f}  (expect ~0: R invariant to structure)', flush=True)
    ok = len(quads) == 4
    print(f'\n  GATE {"PASS ✅ all four quadrants covered, ready for training" if ok else "FAIL: quadrants incomplete, adjust construction"}', flush=True)


def make_figure(rows):
    st_color = {'clean': '#2ca02c', 'blkshuf16': '#ff7f0e', 'fullshuffle': '#d62728'}
    rho_size = {1: 70, 4: 150, 16: 300}
    D = np.array([r['D'] for r in rows]); R = np.array([r['R'] for r in rows])
    dmed, rmed = np.median(D), np.median(R)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for r in rows:
        ax.scatter(r['D'], r['R'], s=rho_size[r['rho']], c=st_color[r['structure']],
                   edgecolor='k', linewidth=0.6, alpha=0.85, zorder=3)
        ax.annotate(f"{r['structure'][:4]}·ρ{r['rho']}", (r['D'], r['R']),
                    fontsize=6.5, xytext=(4, 4), textcoords='offset points')
    ax.axvline(dmed, ls='--', color='gray', lw=1); ax.axhline(rmed, ls='--', color='gray', lw=1)
    ax.text(0.02, 0.97, 'Q3 hiD·loR\n(destroyed unique)', transform=ax.transAxes, va='top', fontsize=7, color='gray')
    ax.text(0.98, 0.97, 'Q4 hiD·hiR\n(memorizable noise)', transform=ax.transAxes, va='top', ha='right', fontsize=7, color='crimson')
    ax.text(0.02, 0.03, 'Q1 loD·loR\n(clean structured)', transform=ax.transAxes, va='bottom', fontsize=7, color='gray')
    ax.text(0.98, 0.03, 'Q2 loD·hiR\n(memorizable text)', transform=ax.transAxes, va='bottom', ha='right', fontsize=7, color='gray')
    # legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markeredgecolor='k', label=s, markersize=9)
           for s, c in st_color.items()]
    ax.legend(handles=leg, title='structure (D knob)', fontsize=8, loc='center left')
    ax.set_xlabel('D  =  bigram conditional entropy on pre-tiling block (bits)')
    ax.set_ylabel(f'R  =  ∞-gram novelty coverage on tiled stream (k={KGRAM})')
    ax.set_title('Paper#4 offline gate: (D,R) decoupling — 4-quadrant coverage\n(marker size = repetition ρ)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT / 'p4_gate_decouple.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'saved {out}', flush=True)


if __name__ == '__main__':
    main()
