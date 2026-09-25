#!/usr/bin/env python
"""q/k pair-profile correlation and its within-layer permutation null (paired RoPE base vs NoPE run).

Input : nope_analysis_v1.json (released as DATA/external/nope_analysis_v1.json),
        key qk_structure.{base,nope}['30000'] -> per-layer q_pair_static / k_pair_static (32 pairs each).
Output: p5v3_nope_pair_corr_null_v1.json (released as DATA/p5v3_nope_pair_corr_null_v1.json).

Protocol (Sec. 4.3.3, App. B.3.2):
  per layer, observed r = Pearson r(q_pair_static, k_pair_static);
  null = 5,000 permutations of the k profile within the layer, numpy default_rng(0),
         one generator shared by both arms (base first, then nope), layers in order, permutations inner;
  per-layer q975 of the permuted r;
  median statistic = median over the six layers of the per-layer permuted r (one value per permutation index);
  its q975 / q999 and the one-sided p-value (count(null >= observed median) + 1) / (B + 1).

The original numbers were computed inline (2026-09-24); this script reproduces the released JSON byte for byte.
"""
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--inp", default=str(HERE.parent / "DATA/external/nope_analysis_v1.json"))
ap.add_argument("--out", default=str(HERE.parent / "DATA/p5v3_nope_pair_corr_null_v1.json"))
ap.add_argument("--n_perm", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

src = json.load(open(a.inp))
B = a.n_perm
rng = np.random.default_rng(a.seed)          # shared across arms, as in the original computation


def pearson(x, y):
    return np.corrcoef(x, y)[0, 1]


def arm_null(arm):
    rows = sorted(src["qk_structure"][arm]["30000"], key=lambda x: x["layer"])
    assert [x["layer"] for x in rows] == list(range(6)), "expected six layers"
    Q = [np.array(x["q_pair_static"]) for x in rows]
    K = [np.array(x["k_pair_static"]) for x in rows]
    assert all(len(q) == len(k) == 32 for q, k in zip(Q, K)), "expected 32 rotary pairs"
    obs = [float(pearson(q, k)) for q, k in zip(Q, K)]
    null = np.zeros((B, len(Q)))
    for l in range(len(Q)):                  # layer outer, permutation inner
        for b in range(B):
            null[b, l] = pearson(Q[l], rng.permutation(K[l]))
    med_null = np.median(null, axis=1)
    med_obs = float(np.median(obs))
    return {
        "per_layer_r": obs,
        "median_r": med_obs,
        "per_layer_null_q975": [float(v) for v in np.quantile(null, 0.975, axis=0)],
        "median_stat_null_q975": float(np.quantile(med_null, 0.975)),
        "median_stat_null_q999": float(np.quantile(med_null, 0.999)),
        "p_median": float((np.sum(med_null >= med_obs) + 1) / (B + 1)),
        "n_pairs": len(Q[0]),
    }


out = {
    "source": "10_p5batch_cloud/results/nope_analysis_v1.json qk_structure.*.30000 q_pair_static/k_pair_static",
    "protocol": ("Pearson r of q and k pair profiles per layer; null = 5000 permutations of the k profile within "
                 "layer (rng seed 0); median-stat null = median over the six layers of per-layer permuted r"),
}
for arm in ("base", "nope"):
    out[arm] = arm_null(arm)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
for arm in ("base", "nope"):
    r = out[arm]
    print(f"{arm:5s} median r {r['median_r']:.3f}  null q975 {r['median_stat_null_q975']:.3f}  "
          f"q999 {r['median_stat_null_q999']:.3f}  p {r['p_median']:.2e}")
