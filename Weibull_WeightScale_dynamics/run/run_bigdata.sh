#!/bin/bash
# Larger-corpus control: train on a bigger token set (e.g. full wikitext-103, ~100M tokens,
# ~2.5 epochs vs ~31 epochs for the 8M baseline) at the same config, to test whether the
# lambda peak drops when over-repetition is reduced.
# Run from the Weibull_WeightScale_dynamics/ directory.
set -e
PY=${PY:-python}
BIGTOK=${BIGTOK:-data/wikitext103_full_pythia_tokens.npy}; CFG=${CFG:-pythia70m_config.json}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
[ -e "$BIGTOK" ] || { echo "missing $BIGTOK (generate with pretok_local.py)"; exit 1; }
OUT=runs/bigdata_run; CKPT=runs/bigdata_ckpts; mkdir -p "$OUT" "$CKPT"

echo "##### big-corpus training (rec 50, lambda_wd 0.01, eta 1e-3, beta2 0.999) #####"
$PY three_force_spline.py --arch pythia --steps 20000 --rec 50 --lwd 0.01 --eta 1e-3 --beta2 0.999 \
  --tokens "$BIGTOK" --config "$CFG" --spacings 250,500,1000 --save_ckpt --ckpt_dir "$CKPT" --out "$OUT"
$PY analysis/compute_lambda_from_ckpts.py --ckpt_dir "$CKPT" --arch pythia --out "$OUT/lambda_trajectory_bigdata.json"
echo "=== DONE -> $OUT ==="
