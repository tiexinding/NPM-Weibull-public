# Weight-Scale Growth Tracks Training Effort, Not Learning Quality

Companion code and derived data for **Paper #4** of the NPM-Weibull series:
*"Weight-Scale Growth Tracks Training Effort, Not Learning Quality"* (Ding, 2026).

The paper trains Pythia-70M from scratch on a controlled 4x4 factorial grid over
wikitext-103 -- shuffle fraction f in {0, 25, 50, 100}% (local predictability, D) x
repetition factor rho in {1, 4, 16, 64} (redundancy, R) -- with three seeds per cell
(48 main-grid runs), plus three-seed corner-cell transfer checks on C4 and Python code
(24 runs) and a 2x learning-rate control (4 runs): 76 runs total.

## Layout

- `code/` -- construction, training, evaluation, classification, and figure scripts.
  - `p4_build_grid_corpora.py` builds the 16-cell grid corpora (+ GATE checks).
  - `run_p4_grid.sh` -> `v1b_spline_verify.py` (training + per-step force logging +
    checkpoints) -> `compute_lambda_from_ckpts.py` (Weibull lambda trajectories).
  - `p4_grid_heldout_eval.py`, `p4_blimp_eval.py` -- held-out / matched-gap / BLiMP eval.
  - `p4_classification_threshold.py` -- 2-component GMM thresholds, labels, and the
    empty-interval robustness check over all 76 runs.
  - `p4_phase1_aggregate.py` -- builds `derived_data/p4_phase1_summary.json`.
  - `p4_paper_figs.py`, `p4_fig45_corpus_eta.py`, `p4_fig6_blimp_figS4_32k.py`,
    `p4_fig7_eta_phase.py`, `p4_figS*.py` -- all paper and supplementary figures.
  - Paths are parameterized: set `NPM_ROOT` (or edit the `<SET_PATH>` placeholders)
    to your data root before running.
- `derived_data/` -- machine-readable results.
  - `p4_phase1_summary.json` -- aggregated per-cell statistics (main grid, corners,
    eta-control, window-shuffle reference, BLiMP, 32k drift, constant-eta probe).
  - `p4_thresholds.json` -- frozen thresholds, per-run labels, and robust intervals
    (76 runs).
  - `cloud_eval/` -- per-run evaluation records (main grid, corner cells with all
    three seeds, eta-control, BLiMP, constant-eta trajectories).
  - `trajectories/` -- lambda(t) trajectories for all 76 runs.
  - `gate_metrics.json`, `grid_trainlevel_metrics.json` -- corpus-construction GATE
    records (orthogonality, quadrant coverage).
- `figures_supplementary/` -- supplementary figures S1-S8 referenced by the paper's
  appendix (seed reference, eta trajectories, 32k drift, extraction, constant-eta
  diagnostics).

## Reproduction

1. Tokenize wikitext-103 with the Pythia (GPT-NeoX) vocabulary and set `NPM_ROOT`.
2. `python code/p4_build_grid_corpora.py` (GATE must PASS).
3. `bash code/run_p4_grid.sh` for the 16 cells x 3 seeds (single RTX-4090-class GPU;
   ~30 min per run).
4. `python code/p4_grid_heldout_eval.py`, then `p4_phase1_aggregate.py`, then
   `p4_classification_threshold.py`.
5. Figures: `p4_paper_figs.py` and the other `p4_fig*.py` scripts.

Training and force-logging internals (three-force decomposition, spline verification)
are shared with Paper #2/#3; see `Weibull_WeightScale_dynamics/` and
`Data_Predictability_WeightScale/` in this repository.

## Related papers

- Paper #1: arXiv:2605.18898 (Weibull framework)
- Paper #2: arXiv:2606.19367 (AdamW three-force dynamics)
- Paper #3: doi:10.5281/zenodo.22056013 (data-predictability law)
