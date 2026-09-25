# Training code, configurations and run records

All LLaMA-style runs use `llama70m_config.json` (6 layers, d = 512, 8 heads, head dim 64, SwiGLU FFN 1376,
RoPE theta 1e4, untied embeddings, vocabulary 50,304), AdamW (beta1 0.9, beta2 0.999), peak learning rate 1e-3,
200-step warm-up, cosine decay to 0.1 x peak over a 30,000-step horizon, decoupled weight decay 0.1, fp32,
sequence length 512. Scripts are released byte-identical to the versions that ran; the `script_sha256_16`
recorded in a run record is the first 16 hex digits of `sha256sum` of the script file.

## Corpus

`data/ea_tokens.npy`: the first 1e8 tokens of the WikiText-103 training split, tokenized with the Pythia
(GPT-NeoX) tokenizer. Rebuild with `ea_pretok.py` (expects the WikiText-103 train parquet shards as
`data/wt103_train*.parquet` and a cached Pythia `tokenizer.json`); the expected shape, id range and SHA-256 are
in `ea_tokens.npy.meta.json`. Batching: `ea_train.py` and `ivn_*` draw random 512-token windows with
`default_rng(seed)`; `tclock_train.py` reads a constructed sequential stream (below).

## Evidence streams

| stream (paper id) | script (sha256_16) | key flags | corpus / batching | run records |
|---|---|---|---|---|
| Controlled data grid, R1 (12 runs) | `tclock_train.py` (f5a733333aa5c681) | `--arm {D1,D2,D3,D4} --seed {1,2,3} --steps 5000` (horizon 30000, bs 24); D1 (f=0, rho=1), D2 (f=1, rho=1), D3 (f=1, rho=64), D4 (f=0, rho=64); paired init across arms (`init_sha_16` e08d3c397b8e2069 for seed 1); checkpoints used 800/1600/3200/5000 | `ea_tokens.npy` -> `partial_shuffle` (fraction f of positions in the unique block of U = T/rho tokens permuted, data seeds 9900/9911) then `tile_to` rho times; sequential stream, held-out tail 0.5M tokens | `run_configs/tclock_D{1..4}_s{1..3}.json`; D2 was later extended to 8,000 steps (`--extend_base`), record `tclock_D2_s*_ext8000.json`, not used in the paper |
| Initialization families, R2 (9 runs) | `ea_train.py` (0f12a8474934f3f2) | `--family {gaussian,laplace,uniform} --seed {1,2,3} --steps 30000 --horizon 30000` (bs 24, sigma 0.02; rank-coupled inverse-CDF init with exact zero mean and sigma, per-matrix seed `SeedSequence([seed, 7700, idx])`) | random windows, `default_rng(seed)`; same batch sequence across families of one seed | `run_configs/ea_{family}_s{1..3}.json` |
| Paired RoPE reassignment, R3 (base, ropeperm) | `ivn_v1/ivn_train.py` (8068aad9398daff7) | `--arm {base,ropeperm} --family gaussian --seed 1` (30000 steps, bs 24, checkpoints 0,800,3200,5000,10000,15000,20000,25000,30000); ropeperm applies the fixed permutation `PI` of the 32 `inv_freq` entries after initialization (recorded as `pi` in the run record) | random windows, seed 1 (batch hashes equal to the Gaussian family seed 1) | `run_configs/ivn_{base,ropeperm}.json` |
| RoPE multi-seed, R4 (ropeperm seeds 2, 3) | `ivn_v2/ivn_train_v2.py` (recorded 3f9a01cc370beab1; see note) + `scalefield_logger.py` (020e8fbb61b6a080, not active) | `--arm ropeperm --family gaussian --seed {2,3} --steps 30000 --ckpt_steps 0,200,400,800,1600,3200,5000,10000,30000`; paired base = Gaussian family seeds 2, 3 | random windows, seed 2 / 3 | `run_configs/ropeperm_s{2,3}.json` |
| No-position run, R4 (NoPE) | `ivn_v2/ivn_train_v2.py` (bff0f26e70398675, the released file) | `--arm nope --family gaussian --seed 1 --steps 30000 --ckpt_steps 0,200,400,800,1600,3200,5000,10000,30000`; rotary forward returns cos=1, sin=0 (preflight S5 recorded) | random windows, seed 1 | `run_configs/nope_s1.json` |
| Optimizer-state dumps, R5 (10k dump run) | `ivn_v3_260912/ivn_train_v3.py` (3bbb7defb0dde2d0) + `scalefield_logger.py` (020e8fbb61b6a080) | phase 1: `--arm base --family gaussian --seed 1 --steps 800 --horizon 30000 --dump_tensors 200,400,800 --logger --log_every 200 --log_win 20 --ckpt_steps 0,200,400,800`; phase 2: same with `--steps 10000 --resume --dump_tensors 3200,10000 --ckpt_steps 0,200,400,800,1600,3200,5000,10000` (bs 24) | random windows, seed 1 (same batch sequence as the seed-1 Gaussian configuration) | `run_configs/dump_early.json` (phase-2 record, `resumed_from` 800) |
| Axis-wise logger run, R5 (30k, bs 16) | `ivn_v3_260913/ivn_train_v3.py` (2094bb2c66d37390) + `scalefield_logger.py` (7015da5731546dc3) | `--arm base --family gaussian --seed 1 --steps 30000 --horizon 30000 --bs 16 --seq 512 --warm 200 --ckpt_steps 0,200,400,800,1600,3200,5000,10000,30000 --logger --log_every 200 --log_win 20 --resume_at 800,3200,10000 --no_full_snapshots --save_final --loss_every 50` | random windows, seed 1, batch 16 (own window sequence) | `run_configs/R0_base_snap.json` |
| Seed-1 logger base (cloud replicate of R3 base) | `ivn_v2/ivn_train_v2.py` (recorded 3f9a01cc370beab1; see note) + `scalefield_logger.py` (020e8fbb61b6a080) | `--arm base --family gaussian --seed 1 --steps 30000 --logger --log_every 200 --log_win 20 --ckpt_steps 0,200,400,800,1600,3200,5000,10000,30000` | random windows, seed 1 | `run_configs/logger_base_s1.json` |
| Pythia-protocol size series, R6 | `pythia_series/run_p4_grid.sh` -> `v1b_spline_verify.py`, `compute_lambda_from_ckpts.py`; corpora `p4_build_grid_corpora.py` (see note) | GPT-NeoX 70M/160M/410M, arm f000_r01, seed 0, 8,000 steps, bs 24 x 512, peak lr 3e-4, weight decay 0.01 (full protocol in `DATA_PROVENANCE.md` §5); `FS="000 100" RHOS="01 64" P5_KEEP="512,2000,5000,8000"` with `CONFIG=pythia{70m,160m,410m}_config.json PFX=p4grid{70m,160m,410m}` in turn; model configs in `pythia_series/` | corpora built by `p4_build_grid_corpora.py` from WikiText-103 | per-matrix read-outs `DATA/pythia_series/*_p5_extract_v1.json` |

## Script versions and consistency checks

Where the exact file a run recorded was not kept, the released file is checked against the stored records on CPU,
without any training step.

- **Controlled data grid (tclock_train.py).** Rebuilding the D1-D4 corpora for seeds 1-3 with the released
  `tclock_train.py` (its own `partial_shuffle` / `tile_to` code, seeds 9900/9911) reproduces the recorded
  `stream_sha256_16`, `held_sha256_16` and `mother_sha256_16`, the unique-block size U and the token count of all
  12 R1 run records (12/12 match).
- **IVN-family trainers (all eight runs of R3-R5 and the seed-1 logger base).** `verify/verify_ivn.py` executes the
  released trainer up to its preflight (model construction, three-family initialization, data sampler) with the
  recorded flags, on CPU (torch 2.11.0, transformers 5.5.0), and compares it with the stored records; per-run
  results are in `verify/*.json`. For all eight runs the first five batch hashes equal `batch_hashes_first5`, and the
  forward pass on the recorded preflight batch reproduces the recorded step-0 loss to within 2e-6 and the per-layer
  hidden-state RMS to within a relative 6e-7, which covers the tensors outside the analysis set (embeddings, norms,
  output layer). Where a step-0 checkpoint is stored (six runs; not for the 10k dump run and the 30k logger run),
  all 42 analysis matrices equal it bit for bit. For the three ropeperm runs the permutation `PI` and the permuted
  `inv_freq` table equal the run record exactly. This includes the three runs whose recorded script sha
  (3f9a01cc370beab1) differs from the released `ivn_train_v2.py` (bff0f26e70398675).
- **fp16 snapshot hashes.** The sha256 of the re-serialized fp16 full-model snapshot differs from the recorded
  step-0 snapshot hash for every run that recorded one, including the NoPE run, whose recorded script sha equals the
  released file. The difference therefore comes from serialization under a different library version and device,
  not from the script revision; the forward-pass check above is the whole-model comparison.

## Notes

- **Script versions.** `ivn_train_v3.py` exists in two versions: `ivn_v3_260912/` (sha 3bbb7defb0dde2d0, used by the
  10k dump run) and `ivn_v3_260913/` (sha 2094bb2c66d37390, used by the 30k logger run). The later version only adds
  three options (`--v_window`, `--resume_at`, `--rebranch_seed`) and their code paths; with those options unset the
  update is unchanged. Each directory also holds the `scalefield_logger.py` recorded for that run
  (020e8fbb61b6a080 vs 7015da5731546dc3; the later logger adds record fields). Keep each trainer next to its logger:
  the trainer imports `scalefield_logger` by module name.
- **ivn_train.py.** The R3 runs predate a metadata-only edit of 2026-09-03 (two lines adding `config`, `rope_theta`,
  `hidden_size`, `num_hidden_layers` to the run record); `ivn_v1/ivn_train.py` is the pre-edit version, consistent with
  the R3 run records, which lack those keys. The run records do not store a script hash.
- **ivn_train_v2.py.** The ropeperm seed-2/3 and logger-base seed-1 runs recorded sha 3f9a01cc370beab1, an earlier
  revision of the same script from the same cloud session that was not retained; the released file (bff0f26e70398675)
  is the revision used by the NoPE run. For the controlled data grid, `run_configs/tclock_grid_manifest_v1.json`
  records `tclock_train.py` sha256 371574efc80636c4...; that file was not kept. The released `tclock_train.py`
  (f5a733333aa5c681) is the revision edited later the same day to add two further repetition arms (D5, D6, not
  used in the paper) and the `--extend_base` mode used for the D2 extension. Rebuilding the D1-D4 corpora with the
  released file reproduces the recorded `stream_sha256_16`, `held_sha256_16`, `mother_sha256_16`, U and token
  count of all 12 R1 run records.
- **Batch pairing** (checked from `batch_hashes_first5`): every family of one seed, the R3 pair, the NoPE run, the
  10k dump run and the seed-1 logger base share the seed-1 window sequence; ropeperm seeds 2/3 share the windows of the
  Gaussian family seeds 2/3; the 30k logger run (batch 16) has its own sequence.
- **Pythia series (`pythia_series/`).** These are the versions used for the size series. Relative to the earlier
  copies published with the companion study (`WeightScale_Training_Effort/code/`), the trainer
  `v1b_spline_verify.py` adds one option, `--ckpt_ana_only`, which drops the embedding and output matrices from
  non-final checkpoints; it changes what is saved, not the training. `run_p4_grid.sh` adds the model size
  (`CONFIG`), the run-name prefix (`PFX`), explicit checkpoint steps (`CKPT_STEPS`), the checkpoints kept for the
  scale-field read-outs (`P5_KEEP`, moved to `<run>/p5_ckpts/`) and `ANA_ONLY`. `p4_build_grid_corpora.py` and
  `compute_lambda_from_ckpts.py` are the same code as the earlier copies. Apart from these additions the released
  files differ from the files that ran only in comments, printed messages and default paths.
- **Launch scripts** (queue runners and watchdogs) are infrastructure only and are not released; the flags above are
  the complete per-run command lines.
- **Not released**: checkpoints, AdamW moment dumps and the logger window files (size); `DATA/` holds every read-out
  computed from them.
