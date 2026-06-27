#!/usr/bin/env python
"""Paper#3 v2 实验1 (B2 6-16): 腐蚀梯度数据生成器 = make-or-break 主实验.
取干净 token, 按 ρ=0→1 逐级在每个 seq_len 窗口内随机打乱 ρ 比例的位置.
单轴干净: 打乱=重排 → unigram 分布精确不变 + 序列长度不变 → 只破坏可学 X→Y 结构.
机制预期: ρ↑ → 结构崩 → 跨 batch 梯度不相干 → 对齐力崩 → λ 不长.
用法: python gen_corruption_gradient.py <in_npy> <seq_len> <out_dir> [seed] [n_levels]
产出: <out_dir>/corrupt_rho{000..100}_s{seed}.npy
"""
import sys
import numpy as np


def corrupt(tokens, seq_len, rho, rng):
    n = (len(tokens) // seq_len) * seq_len
    arr = tokens[:n].reshape(-1, seq_len).copy()
    k = int(round(rho * seq_len))
    if k >= 2:
        for row in arr:
            idx = rng.choice(seq_len, size=k, replace=False)
            row[idx] = row[rng.permutation(idx)]
    return arr.reshape(-1)


def struct_score(tokens, seq_len):
    """可学结构代理: 与原序相邻 bigram 的保留率(高=结构在). 仅诊断."""
    n = (len(tokens) // seq_len) * seq_len
    return float(np.mean(tokens[:n][:-1] == tokens[:n][:-1]))  # placeholder, replaced below


def adj_match(orig, cor, seq_len):
    """腐蚀后相邻对仍 = 原相邻对的比例(结构保留率, ρ=0→1.0, ρ=1→低)."""
    n = (len(orig) // seq_len) * seq_len
    o = orig[:n].reshape(-1, seq_len)
    c = cor[:n].reshape(-1, seq_len)
    same = (o[:, :-1] == c[:, :-1]) & (o[:, 1:] == c[:, 1:])
    return float(same.mean())


def main(in_npy, seq_len, out_dir, seed=0, n_levels=11):
    tokens = np.load(in_npy)
    _n = (len(tokens) // seq_len) * seq_len
    ug0 = np.bincount(tokens[:_n].astype(np.int64))  # 基准=截断后原序(corrupt 也截到 seq_len 整数倍)
    levels = [round(i / (n_levels - 1), 4) for i in range(n_levels)]
    print(f"[corrupt] in={in_npy} tokens={len(tokens):,} seq_len={seq_len} levels={levels}")
    for rho in levels:
        rng = np.random.default_rng(seed * 100003 + int(rho * 10000))
        out = corrupt(tokens, seq_len, rho, rng).astype(np.uint16)
        ug = np.bincount(out.astype(np.int64), minlength=len(ug0))
        ug_ok = np.array_equal(ug[: len(ug0)], ug0)  # unigram 精确不变?
        struct = adj_match(tokens, out, seq_len)       # 结构保留率
        fn = f"{out_dir}/corrupt_rho{int(rho * 100):03d}_s{seed}.npy"
        np.save(fn, out)
        print(f"  ρ={rho:.2f}  unigram_preserved={ug_ok}  struct_retain={struct:.3f}  -> {fn}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), a[3], int(a[4]) if len(a) > 4 else 0, int(a[5]) if len(a) > 5 else 11)
