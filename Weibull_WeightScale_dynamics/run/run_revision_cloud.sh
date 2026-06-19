#!/bin/bash
# Learning-rate sweep + extra seeds (paper Sec 4.4 / 4.5). ~4 runs x 40 min.
# Run from the Weibull_WeightScale_dynamics/ directory. Checkpoints are deleted after
# the Weibull lambda(t) is computed (only trajectories + force logs are kept).
set -e
PY=${PY:-python}
TOK=${TOK:-data/wikitext_pythia_tokens.npy}; CFG=${CFG:-pythia70m_config.json}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
COM(){ $PY analysis/compute_lambda_from_ckpts.py --ckpt_dir "$1" --arch pythia --out "$2"; }

# ---- learning-rate sweep (eta = 3e-4, 3e-3; 1e-3 is the baseline) ----
for ETA in 3e-4 3e-3; do
  TAG=${ETA/e-/e}; OUT=runs/eta_${TAG}; CKPT=${OUT}_ckpts; mkdir -p "$OUT" "$CKPT"
  echo "##### eta=$ETA (lambda_wd=0.01, seed 1) #####"
  $PY three_force_spline.py --arch pythia --steps 20000 --rec 50 --lwd 0.01 --eta $ETA --seed 1 --beta2 0.999 \
    --tokens "$TOK" --config "$CFG" --spacings 250,500,1000 --save_ckpt --ckpt_dir "$CKPT" --out "$OUT"
  COM "$CKPT" "$OUT/lambda_trajectory_eta_${TAG}.json"; rm -rf "$CKPT"
done

# ---- extra seeds (seeds 1,2 already done; add 3,4 for a 4-seed range) ----
for SEED in 3 4; do
  OUT=runs/seed_${SEED}; CKPT=${OUT}_ckpts; mkdir -p "$OUT" "$CKPT"
  echo "##### seed=$SEED (eta=1e-3, lambda_wd=0.01) #####"
  $PY three_force_spline.py --arch pythia --steps 20000 --rec 50 --lwd 0.01 --eta 1e-3 --seed $SEED --beta2 0.999 \
    --tokens "$TOK" --config "$CFG" --spacings 250,500,1000 --save_ckpt --ckpt_dir "$CKPT" --out "$OUT"
  COM "$CKPT" "$OUT/lambda_trajectory_seed_${SEED}.json"; rm -rf "$CKPT"
done
echo "=== ALL DONE ===  # each run produces ~8.5 GB of checkpoints transiently; ensure disk space"
