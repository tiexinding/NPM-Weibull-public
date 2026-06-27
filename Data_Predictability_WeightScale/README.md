# Data Predictability Shapes Weibull Weight-Scale Growth in Transformer Training

Companion code and derived data for the paper:

> **Data Predictability Shapes Weibull Weight-Scale Growth in Transformer Training**
> Tiexin Ding (Independent Researcher). arXiv:XXXX.XXXXX.

This is the Paper #3 companion in the unified `NPM-Weibull-public` repository (see the
top-level README for the Paper #1 framework `arXiv:2605.18898` and Paper #2 training
dynamics `arXiv:2606.19367`).

## What this paper shows

Within controlled corruption families, the growth of the Weibull weight-scale parameter
follows a learning-rate-conditioned law in a single training-free corpus statistic — the
bigram conditional entropy `D = H(next | prev)`:

```
λ² − λ₀² = C₀(η) + C₁(η) · (Hr − D)^0.59
```

The convex exponent `0.59 = 1/p` is inherited from an independently measured data-side
saturation relation, not fitted to the growth. After removing the two per-η coefficients,
23 runs collapse onto `(Hr − D)^0.59` with unit slope (R² = 0.941). The law is validated
end-to-end (5.7% held-out within-family error), holds at model and per-layer levels and
across two architectures, and breaks on code — implicating redundancy as a second axis of
a broader Φ(D, R, A, H) data-to-weight framework.

## Layout

```
Data_Predictability/
├── code/                 figure- and fit-generating scripts (+ shared style p3_style.py)
├── derived_data/         per-run Weibull λ trajectories + pre-computed data-side stats
└── requirements.txt
```

## Figure → script map

| Figure | Script |
|---|---|
| Fig 1 (η-scaling), Fig 2 (collapse) | `make_convex_mpl.py` |
| Fig 3 (stable-phase stepwise law)   | `make_law_vs_step_mpl.py` |
| Fig 4 (convex derivation)           | `make_physics_fits_convex_mpl.py` |
| Fig 5 (end-to-end self-validation)  | `make_selfval_mpl.py` (numbers: `make_selfval.py`) |
| Fig 6/7 (architecture compare/mechanism) | `make_arch_figs_mpl.py` |
| Fig 8 (pooled vs per-layer)         | `make_perlayer_robust_mpl.py` |
| Fig 9 (redundancy second axis)      | `make_2d_redundancy_mpl.py` |
| Fig A1 (cross-seed)                 | `make_crossseed_mpl.py` |
| Fig A2/A3 (per-block C₁ / per-ρ λ maps) | `make_arch_figs_mpl.py`, `make_arch_lambda_maps.py` |
| Fig A4 (depth profile)              | `make_arch_perlayer_lambda.py` |
| Fig A5 (repetition / multi-epoch)   | `make_repeat_epoch_mpl.py` |
| exponent-robustness checks          | `verify_saturation_exponent.py` |

## Reproduction

```
pip install -r requirements.txt
cd code && python make_convex_mpl.py     # etc.
```

- **Weight side** — the Weibull λ trajectories used by every figure are in
  `derived_data/` (one JSON per run); figures regenerate from these directly.
- **Data side** — `derived_data/data_side_stats.json` holds the pre-computed bigram
  conditional entropy `D` and struct-retain fraction `S` for every corruption level and
  natural corpus, so the data-side statistics need not be recomputed from raw tokens.
- **Raw tokens** are not redistributed (they derive from the public WikiText / C4 /
  CodeParrot corpora). The corruption construction (`code/gen_corruption_gradient.py`)
  regenerates the corrupted-token files from a base corpus for full from-scratch
  reproduction.

## License

CC BY 4.0 (see the repository-root `LICENSE`).
