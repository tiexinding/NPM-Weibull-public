#!/bin/bash
# Relaxation-regime validation. Self-train with weight decay lambda_wd=0.05 so that
# T/tau = steps * eta * lambda_wd = 20000 * 1e-3 * 0.05 = 1.0, pushing training into the
# relaxation regime, then validate predicted-vs-observed lambda against ground truth.
# Run from the Weibull_WeightScale_dynamics/ directory.
set -e
PY=${PY:-python}
TOK=${TOK:-data/wikitext_pythia_tokens.npy}; CFG=${CFG:-pythia70m_config.json}
OUT=${OUT:-runs/relax_lwd05}; CKPT=${CKPT:-runs/relax_ckpts}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
mkdir -p "$OUT" "$CKPT"

echo "=== [1/4] self-train Pythia-arch, lambda_wd=0.05 (T/tau=1.0), save checkpoints (~40 min on one GPU) ==="
$PY three_force_spline.py --arch pythia --steps 20000 --rec 50 --lwd 0.05 \
  --tokens "$TOK" --config "$CFG" --spacings 250,500,1000 --save_ckpt --ckpt_dir "$CKPT" --out "$OUT"

echo "=== [2/4] fit ground-truth Weibull lambda(t) from checkpoints ==="
$PY analysis/compute_lambda_from_ckpts.py --ckpt_dir "$CKPT" --arch pythia --out "$OUT/lambda_trajectory_relax.json"

echo "=== [3/4] predicted vs observed lambda (expect a relaxation phase, t/peak < 1) ==="
$PY analysis/make_predicted_lambda_fig.py --label "Pythia-arch lwd0.05 (T/tau=1)" \
  --true "$OUT/v1b_spline_true.jsonl" --lam "$OUT/lambda_trajectory_relax.json" --lwd 0.05 \
  --out "$OUT/F_lambda_pred_vs_obs_relax.png"

echo "=== [4/4] three-force decomposition + error breakdown (decay takes over near saturation) ==="
$PY analysis/make_force_decomp_error.py --label "Pythia-arch lwd0.05 (T/tau=1)" \
  --true "$OUT/v1b_spline_true.jsonl" --lam "$OUT/lambda_trajectory_relax.json" --lwd 0.05 \
  --out "$OUT/F_force_decomp_error_relax.png"
echo "=== DONE -> $OUT ==="
