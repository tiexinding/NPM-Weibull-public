#!/bin/bash
#
#   TOK=.../05_paper4/token_data/p4grid OUT=/path/p4_runs bash run_p4_grid.sh
#   SIMS=<SET_PATH> TOK=<SET_PATH> OUT=<SET_PATH> \
#     PY=/root/miniconda3/bin/python bash run_p4_grid.sh
#   STEPS=8 BS=2 SEQ=64 REC=2 SMOKE=1 bash run_p4_grid.sh
set -u
ROOT_DEFAULT="<SET_PATH>"
SIMS=${SIMS:-$ROOT_DEFAULT/40_Dynamics/sims}
TOK=${TOK:-$ROOT_DEFAULT/50_data/05_paper4/token_data/p4grid}
OUT=${OUT:-$ROOT_DEFAULT/50_data/05_paper4/runs}
PY=${PY:-python3}
STEPS=${STEPS:-8000}; ETA=${ETA:-3e-4}   # moderate eta (high eta can produce spurious AdamW spikes in memorization cells)
BS=${BS:-24}; SEQ=${SEQ:-512}; REC=${REC:-100}
SEEDS=${SEEDS:-"0"}
FS=${FS:-"000 025 050 100"}; RHOS=${RHOS:-"01 04 16 64"}
SMOKE=${SMOKE:-0}
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
mkdir -p "$OUT"

run_one () {
  local f=$1 rho=$2 seed=$3
  local tag="p4grid_f${f}_r${rho}_s${seed}"
  local o="$OUT/$tag" c="$OUT/$tag/ckpts"
  [ -f "$o/lambda_trajectory_continue.json" ] && { echo "SKIP $tag"; return; }
  local tok="$TOK/p4grid_f${f}_r${rho}_s0.npy"   # corpus fixed at s0 (common random numbers); seed only varies init/sampling
  [ -f "$tok" ] || { echo "MISSING $tok"; return; }
  mkdir -p "$o"; echo "##### P4 $tag ($(date +%m-%d\ %H:%M)) #####"
  $PY "$SIMS/v1b_spline_verify.py" --arch pythia --seed "$seed" \
    --steps "$STEPS" --rec "$REC" --lwd 0.01 --eta "$ETA" --beta2 0.999 --bs "$BS" --seq "$SEQ" \
    --tokens "$tok" --config "$SIMS/pythia70m_config.json" \
    --save_ckpt --ckpt_dir "$c" --out "$o" || { echo "FAIL $tag"; return; }
  $PY "$SIMS/compute_lambda_from_ckpts.py" --ckpt_dir "$c" --arch pythia \
    --out "$o/lambda_trajectory_continue.json" || { echo "FAIL_LAM $tag"; return; }
  mkdir -p "$o/W_ckpts" && mv "$c/step${STEPS}.pt" "$o/W_ckpts/" && rm -rf "$c"
  $PY - "$o/v1b_spline_true.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
mna = max(-r["true_align_full"] for r in rows)
peak = max(r["loss"] for r in rows[len(rows)//4:]) if len(rows) > 4 else float("nan")
print(f"[guard] max(-align)={mna:.3f} {'🔴 unstable!' if mna > 10 else 'ok'} | late loss peak={peak:.3f}")
EOF
  echo "##### DONE $tag ($(date +%H:%M)) #####"
}

for seed in $SEEDS; do for f in $FS; do for rho in $RHOS; do
  run_one "$f" "$rho" "$seed"
  [ "$SMOKE" = "1" ] && { echo "SMOKE: single cell only, stopping"; exit 0; }
done; done; done
echo "##### P4_GRID_DONE ($(date +%m-%d\ %H:%M)) #####"
