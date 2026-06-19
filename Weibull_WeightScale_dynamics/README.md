# Paper #2: Weibull Weight-Scale Dynamics under AdamW Training

Companion code + derived data for **"Weibull Weight-Scale Parameter Evolution under AdamW Training Dynamics"** (Tiexin Ding), the follow-up to Paper #1 (*A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions*, arXiv:2605.18898).

> This folder lives inside the `NPM-Weibull-public` repo (Paper #1) and **reuses `npm_weibull`** (the Paper #1 package) for Weibull fitting. arXiv ID: 2606.19367.

## Overview
Paper #1 found that transformer Transmission-class weights keep a stable Weibull shape k≈1.20 while the scale λ evolves. This paper explains *why* λ rises and relaxes during AdamW training, via a **leading-order three-force decomposition** of the squared weight norm (alignment / injection / decay), measured on self-trained transformers and recovered from sparse public checkpoints by a **spline displacement** method.

## Structure
```
Weibull_WeightScale_dynamics/
├── three_force_spline.py     # core: three-force decomposition + spline recovery + per-component (Q/K) forces + k-lock
├── analysis/                 # figure/analysis scripts
│   ├── make_force_budget_fig.py / make_bridge_sensitivity_fig.py
│   ├── make_eta_sweep_fig.py / make_4seed_fig.py / make_bsel_fig.py
│   ├── make_data_shaping_fig.py / make_perlayer_heatmap_en.py / hero_merged_en.py
│   ├── realpythia_perblock_k.py / compute_lambda_from_ckpts.py
├── run/                      # reproduce training/sweeps
│   ├── run_revision_cloud.sh # η sweep + extra seeds
│   ├── run_relax.sh / run_continue.sh / run_bigdata.sh
└── derived_data/             # λ trajectories, force-budget logs, per-block/-layer fits (small, ~700KB)
    ├── self_train/  real_pythia/  revision/
```

## Reproduce
1. **Self-train** (single RTX 4090, ~40 min/run): `python three_force_spline.py --arch pythia --steps 20000 --rec 50 --lwd 0.01 --eta 1e-3 --seed 1 --tokens <wikitext.npy> --config pythia70m_config.json --out <dir>`. Logs three-force budget + per-component forces to `<dir>/v1b_spline_true.jsonl`.
2. **Sweeps**: `run/run_revision_cloud.sh` (η ∈ {3e-4,1e-3,3e-3}, seeds 3–4); `run/run_relax.sh` (λ_wd ∈ {0.05,0.1,0.2}).
3. **Spline on real Pythia**: download EleutherAI Pythia checkpoints, run the spline recovery (see `three_force_spline.py` displacement identity).
4. **Figures**: `analysis/make_*.py` regenerate paper figures from `derived_data/`.

## Data
- `derived_data/` ships the **fitted Weibull λ trajectories, force-budget logs, and per-block/-layer k,λ fits** used in the paper figures (small JSON/JSONL).
- **Model checkpoints are not hosted** (reproducible from the scripts + seeds above). Real Pythia checkpoints are public: EleutherAI `pythia-{70m,160m,410m,1b}` (arXiv:2304.01373).
- Corpus: wikitext-103 (public).

## Dependency
Requires `npm_weibull` (this repo's Paper #1 package, PyPI `npm-weibull-py` v0.4) for the middle-80% Weibull probability-plot fit.

## Caveat (Selection class)
The Selection class (Q/K projections, Paper #1 k=0.28–0.51) does **not** follow the k≈1.20 Weibull regime; for it the three-force budget describes RMS dynamics, not a Weibull λ (see `make_bsel_fig.py`).

## License
CC BY 4.0 (same as Paper #1).
