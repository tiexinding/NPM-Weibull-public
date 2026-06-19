#!/bin/bash
# Continue-training control: initialize from the real public Pythia-70M weights and continue
# training on wikitext, to test whether lambda climbs toward the wikitext-from-scratch level
# (a data-driven effect) or stays near the pretrained value.
# NOTE: this run downloads the real Pythia-70M weights, so it does NOT set HF_HUB_OFFLINE.
# Run from the Weibull_WeightScale_dynamics/ directory.
set -e
PY=${PY:-python}
CFG=${CFG:-pythia70m_config.json}
TOK=${TOK:-data/wikitext103_full_pythia_tokens.npy}
export OMP_NUM_THREADS=4
OUT=runs/continue_run; CKPT=runs/continue_ckpts; mkdir -p "$OUT" "$CKPT"

echo "##### continue-train: real Pythia-70M (final) init + wikitext (same config) #####"
$PY three_force_spline.py --arch pythia --init_from pythia-70m --init_rev main \
  --steps 20000 --rec 50 --lwd 0.01 --eta 1e-3 --beta2 0.999 \
  --tokens "$TOK" --config "$CFG" --spacings 250,500,1000 --save_ckpt --ckpt_dir "$CKPT" --out "$OUT"
$PY analysis/compute_lambda_from_ckpts.py --ckpt_dir "$CKPT" --arch pythia --out "$OUT/lambda_trajectory_continue.json"
echo "=== DONE -> $OUT ==="
