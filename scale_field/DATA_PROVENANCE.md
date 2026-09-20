# Data provenance for *A Mesoscopic View of Transformer Weights Through Row and Column Scale Fields*

Release location: <https://github.com/tiexinding/NPM-Weibull-public>, directory `scale_field/` (tag to be set at posting).

Every figure, table and load-bearing number in the paper traces through three layers: **raw runs and checkpoints → read-out JSON (extraction script) → figure or table (plotting script)**. Layers 2 and 3 are released with the paper; layer 1 is released by size tier. In this release directory: `DATA/` holds the read-out JSON files (the four read-outs that lived outside the draft folder are under `DATA/external/` with their original file names), `scripts/` the extraction scripts, `FS/` the figure and table scripts; `PC/` refers to the manuscript compile folder, which is not part of the release. Raw-source paths in §1 are the author's working layout. MD5 sums of every released file are in `MANIFEST.txt`.

## 1. Raw sources

| id | location | content | used in |
|---|---|---|---|
| R1 | `04_tclock_runs/wt_ckpts/tclock_D{1..4}_s{1..3}/` | controlled data grid, 12 runs, 4 checkpoints each | Sec. 4.1-4.3, App. B.1-B.3 |
| R2 | `04_ea_runs/ckpt_key/ea_{gaussian,laplace,uniform}_s{1..3}/ckpt/` | three initialization families, 9 runs, 7 checkpoints each | Sec. 4.1, 4.3, 4.4, App. B.1-B.4 |
| R3 | `05_ivn_runs/ivn_{base,ropeperm}/ckpt/` | paired RoPE reassignment, 9 checkpoints each | Sec. 4.3, 4.6, App. A.4, B.3, B.5 |
| R4 | `10_p5batch_cloud/results/` | RoPE multi-seed and no-position runs | Sec. 4.3, App. B.3 |
| R5 | `11_p5mech_cloud/results/` | AdamW second-moment dumps | Sec. 4.5, App. B.5 |
| R6 | companion-study Pythia size series, per-matrix read-outs only | cross-size frequency profile | Fig. 5(d) |
| R7 | public Pythia 70M/160M/410M/1B final checkpoints | 348 matrices | Fig. 2(a) |
| R8 | `data_local/ea_tokens.npy` | pre-tokenized stream, 1e8 tokens | Sec. 4.6 evaluation slices |

Training and offline analysis code: `cloud_bundle/` (`ea_offline_analysis.py` implements the middle-80% Weibull-plot fit).

## 2. Read-out JSON files

| file | extraction script | raw input | consumed by |
|---|---|---|---|
| `DATA/p5v2_figure_readouts_v1.json` | `scripts/p5v2_fig1_v1.py` | R1, R2, R7 | Fig. 2 |
| `DATA/p5v3_core_quantile_profile_v1.json` | `scripts/p5v3_core_quantile_profile_v1.py` | R2 | Fig. 2(b) |
| `DATA/p5_public_pythia_knorm_v1.json` | `scripts/p5v2_public_knorm_refit_v1.py` | R7 | Fig. 2(a), App. A.2 |
| `06_hrow/kmix_bridge_v1_1.json` | `scripts/kmix_bridge_v1_1.py` | R1 | Fig. 3, Fig. 9(a), App. B.2 (hold-out check) |
| `DATA/p5v2_twoaxis_bridge_v1.json` | `scripts/p5v2_twoaxis_bridge_v1.py` | R1 | Fig. 3, Fig. 9(b,c), App. B.2.1-B.2.2, Table 5 |
| `DATA/p5v2_ea_kbi_trajectory_v1.json` | `scripts/p5v2_ea_kbi_trajectory_v1.py` | R2 | Fig. 2(c), Fig. 6, Fig. 13, App. A.2 |
| `DATA/p5v3_fig4_field_evolution_v1.json` | `scripts/p5v3_fig4_field_evolution_v1.py` | previous row | Fig. 6 |
| `DATA/p5v2_identity_transfer_v1.json` | `scripts/p5v2_identity_transfer_v1.py` | R1, R2 | Fig. 4, App. B.3.1 |
| `DATA/p5v2_path_gain_balance_v1.json` | `scripts/p5v2_path_gain_balance_v1.py` | R1, R2 | Fig. 14, App. B.6 |
| `DATA/p5v2_fig2a_profiles_v1.json` | `scripts/p5v2_fig2a_profiles_v1.py` | R3, R6 | Fig. 5(a,d), Fig. 11 |
| `DATA/p5v2_fig2c_hw_sd_readouts_v1.json` | `scripts/p5v2_fig2c_hw_sd_v1.py` | R3 | Fig. 5(c), Fig. 11 |
| `DATA/p5_kmix_on_ivn_v1.json` | `scripts/p5_kmix_on_ivn_v1.py` | R3 | Fig. 5, App. A.2 |
| `05_ivn_runs/ivn_j1_j3_hw_v1.json` | `scripts/ivn_j1_j3_hw_v1.py` | R3 | Fig. 5, App. A.2 (IQR read-out) |
| `10_p5batch_cloud/results/j3_multiseed_v1.json` | `scripts/p5v2_j3_multiseed_v1.py` | R2 (Gaussian), R4 | Fig. 5(b) |
| `10_p5batch_cloud/results/nope_analysis_v1.json` | `scripts/p5v2_nope_analysis_v1.py` | R4 | Fig. 12, App. B.3.2 |
| `DATA/p5v2_field_distribution_v1.json` | `scripts/p5v2_field_distribution_v1.py` | R1, R2 | Fig. 10, App. B.2.3 |
| `DATA/p5t1_chain_localize_v1.json` | `scripts/p5t1_chain_localize_v1.py` | R5 (base run dumps) | App. B.5 (lock-in and pairing steps of W versus V-hat and the gradient chain) |
| `DATA/p5t1_V_factor_matrix_v1.json` | `scripts/p5t1_V_factor_matrix_v1.py` | R5 | Fig. 7, App. B.5 |
| tab:vfactors values (removed from PDF) | `scripts/p5t1_V_vs_W_axes_v1.py` | R5 | App. B.5 |
| `DATA/p5v3_gain_edit_eval_v1.json`, `_slice2.json` | `scripts/p5v3_gain_edit_eval_v1.py`, `_slice2.py` | R3 base at 30k, R8 | Fig. 8, Table 7, Sec. 4.6 |
| `DATA/p5v3_gain_edit_matched_controls_v1.json` | `scripts/p5v3_gain_edit_matched_controls_v1.py` | R3 base at 30k, R8 | Fig. 8(b), Table 6 |
| `DATA/p5v3_gain_edit_perturbation_v1.json` | `scripts/p5v3_gain_edit_perturbation_v1.py` | R3 base at 30k | Fig. 8, Table 7 |
| `DATA/p5v3_lambda_scale_equivariance_v1.json` | `scripts/p5v3_lambda_scale_equivariance_v1.py` | R3 base, 9 checkpoints | App. A.4, Discussion |

## 3. Figures and tables

| item | label | file | plotting script | inputs |
|---|---|---|---|---|
| Fig. 1 | fig:overview | `P5v3_F1_decoder_training_loop_v4_260915.png` | hand-drawn SVG, `08_paper5_draft/figures/Figure1/` | schematic |
| Fig. 2 | fig:core | `P5v3_F1_normalized_core_v3.png` | `FS/p5v3_fig2_core_v3.py` | figure_readouts, core_quantile_profile |
| Fig. 3 | fig:bridge | `P5v3_F2_scale_field_bridge_v4.png` | `FS/p5v3_fig2_scale_field_bridge_v4.py` | kmix_bridge_v1_1, twoaxis_bridge, ea_kbi_trajectory |
| Fig. 4 | fig:transfer | `P5v2_S10_identity_transfer_v4.png` | `FS/p5v2_identity_transfer_plot_v4.py` | identity_transfer |
| Fig. 5, 11, 12 | fig:rope, fig:rope_readout, fig:nope | `P5v3_F5_rope_v5.png`, `P5v3_SB_rope_readout_v3.png`, `P5v3_SB_nope_v4.png` | `FS/p5v3_fig5_rope_v7.py` | fig2a_profiles, fig2c_hw_sd, kmix_on_ivn, ivn_j1_j3_hw, j3_multiseed, nope_analysis, R6 |
| Fig. 6 | fig:evolution | `P5v3_F4_field_evolution_v3.png` | `FS/p5v3_fig4_field_evolution_v3.py` | fig4_field_evolution, ea_kbi_trajectory |
| Fig. 7 | fig:topology | `P5v2_S19_factor_topology_v6.png` | `FS/p5t1_factor_matrix_timeline_v6.py` | p5t1_V_factor_matrix |
| Fig. 8 | fig:gainedit | `P5v3_F8_gain_edit_v2.png` | `FS/p5v3_fig8_gain_edit_v2.py` | gain_edit_eval_v1, _slice2, matched_controls, perturbation |
| Fig. 9 | fig:twoaxis | `P5v3_SB_bridge_coefficients_v2.png` | `FS/p5v3_figB_bridge_coefficients_v2.py` | kmix_bridge_v1_1, twoaxis_bridge |
| Fig. 14 | fig:gain | `P5v2_S13_path_gain_balance_v5.png` | `FS/p5v2_path_gain_balance_plot_v5.py` | path_gain_balance |
| Fig. 10 | fig:fielddist | `P5v2_S12_field_distribution_v4.png` | `FS/p5v2_field_distribution_plot_v4.py` | field_distribution |
| Fig. 13 | fig:timing | `P5v2_S11_timing_chain_v6.png` | `FS/p5v2_figS11_timing_plot_v6.py` | ea_kbi_trajectory |
| Tables 1-4 | definitions, evidence roles, identity-axis addresses, protocols | hand-written | no data | - |
| Table 5 | tab:sides | generated `app/tab_sides_body.tex` | `FS/p5v3_tabB_sides_v1.py` | twoaxis_bridge |
| (tab:vfactors, removed from the PDF 2026-09-19) | tab:vfactors | per-kind R^2 quoted in App. B.5 | `scripts/p5t1_V_vs_W_axes_v1.py` | `DATA/p5t1_V_vs_W_axes_v1.json` |
| Table 7 (data package only) | tab:gainedit_full | generated `app/tab_gainedit_full.tex`, not in the PDF | `FS/p5v3_tabB_gain_edit_full_v1.py` | gain-edit JSONs |
| Table 6 | tab:gainedit_matched | generated `.tex` | `FS/p5v3_tabB_gain_edit_matched_v2.py` | gain-edit JSONs |

Every plotting script reads only the JSON files listed here, asserts the counts and identities it depends on, and prints the numbers quoted in its caption.

## 4. Release tiers

- **With the paper**: all read-out JSON files with the manifest, the extraction and plotting scripts above, the fitting-protocol code, and the Figure 1 SVG source (about 15 MB).
- **Checkpoints and the token stream** (R1-R5, R8) are archived locally and not uploaded; the read-outs and scripts above reproduce every figure and number from the archived JSON files.
- **Not released**: intermediate backups and branches not used in the paper.

## 5. Details referenced from Appendix A

**Fit counts.** Controlled data grid: 12 runs x 4 checkpoints x 6 layers = 288 matrices per kind; 1,440 row-side fits on q, k, v, gate, up and 2,016 two-sided fits on all seven kinds. Initialization families: 9 runs x 7 checkpoints x 42 matrices = 2,646 fits. Paired RoPE reassignment: 4 arms recorded (base, permuted, plus two intervention arms not used in the paper) x 9 checkpoints x 42 matrices; per-matrix read-outs in `DATA/p5_kmix_on_ivn_v1.json`. Public Pythia endpoints: 348 matrices.

**Data-arm construction (f, rho).** For a run with token budget T = steps x 24 x 512, the first U = T/rho tokens of the pre-tokenized corpus form the unique block; a fraction f of the positions in that block (chosen without replacement) have their tokens permuted among themselves (`partial_shuffle`); the block is then tiled rho times to fill T (`tile_to`), i.e. shuffle first, tile second. The held-out tail after the block receives the same shuffle rule. Implementation: `cloud_bundle/ivn_train_v2.py` (`partial_shuffle`, `tile_to`, seeds 9900/9911 derived from the data seed and f). Arms: D1 (f=0, rho=1), D2 (1, 1), D3 (1, 64), D4 (0, 64); three seeds each, paired across arms. Construction script: companion-study data loader in `cloud_bundle/`.

**Field-edit evaluation slices.** 48 sequences x 512 tokens read from `data_local/ea_tokens.npy` at offsets 9e7 (slice 1, used for Figure 8) and 6e7 (slice 2); batch 8; base loss 3.347 / 3.093.

**Fits below the R^2 gate (0.99).** Initialization families: 133 of 2,646, all uniform initialization: step 0: 126, step 800: 7; minimum R^2 0.981. Public Pythia endpoints: two raw q fits, 70M layer 3 and 160M layer 6 (`DATA/p5_public_pythia_knorm_v1.json`, key `fit_gate_below_0.99`). Paired RoPE runs: all fits above 0.997 (`DATA/p5_kmix_on_ivn_v1.json`, key `fit_adequacy`). Per-fit list: `DATA/p5v2_ea_kbi_trajectory_v1.json`, key `fits_below_r2_gate`.

**Histogram lower edge.** The fit floors |W| at 1e-13 and bins log10|W| on [-12, 2]; entries below 1e-12 would be excluded. Checked on three checkpoints (paired-RoPE base at steps 0 and 30,000; uniform family seed 1 at step 800; 5.7e7 entries): no entry below 1e-12 and no exact zero; smallest non-zero magnitudes 5e-10 to 9e-9.

**sd versus IQR read-out on the paired RoPE run** (permuted against base, median over steps 20,000/25,000/30,000 and six layers, row level): v -9.7% (sd) vs +0.6% (IQR); q +3.5% vs +7.3%; k +1.1% vs +9.5%. Sources: sd from `DATA/p5_kmix_on_ivn_v1.json` (`per_matrix_readouts.hw`), IQR from `05_ivn_runs/ivn_j1_j3_hw_v1.json` (`H_W`).

**Scale equivariance ledger** (`DATA/p5v3_lambda_scale_equivariance_v1.json`, kind medians over six layers):

| kind | C_lambda min-max over steps | log s: step 0 -> 30k | log C_lambda: step 0 -> 30k |
|---|---|---|---|
| q_proj | 0.863-0.887 | +1.26 | -0.027 |
| k_proj | 0.845-0.887 | +1.26 | -0.048 |
| v_proj | 0.850-0.887 | +0.74 | -0.025 |
| o_proj | 0.856-0.887 | +0.77 | -0.017 |
| gate_proj | 0.869-0.887 | +1.04 | -0.002 |
| up_proj | 0.869-0.887 | +0.94 | -0.002 |
| down_proj | 0.867-0.887 | +0.94 | -0.005 |

Refit of W/s reproduces lambda_P/s to 0.2% (histogram rebinning). Ratio lambda_P / (s / sqrt(Gamma(1+2/k))) = 1.03-1.09. Proxy check on the nine initialization-family runs with mu = mean_i log r_i in place of s (`06_hrow/ea_mulhk_v1.json`, `06_hrow/ea_lambda_fit_v1.json`): per-segment |Delta log lambda - Delta mu| < 0.01 on q, k, v, gate.

**Balancing routine.** `bothnorm_tracked` in `scripts/p5v2_ea_kbi_trajectory_v1.py` (identical copies in the other extraction scripts): factors a, b start at one; each sweep divides rows by their RMS (a *= r) then columns by their RMS (b *= c); stop when the relative spread of row and column RMS is below 1e-6 or after 200 sweeps; final global factor g restores the original RMS and is split as a /= sqrt(g), b /= sqrt(g); asserts spread < 1e-3 and |diag(a) Z diag(b) - W| < 1e-9 x RMS(W). The returned core has row and column RMS equal to RMS(W); the paper's Z is that core divided by s.

**Pythia size series protocol** (from the companion study, ICLR submission appendix table): Pythia-70M/160M/410M GPT-NeoX architectures (70M: 6 layers, d=512, 8 heads, FFN 2048, rotary 0.25, fused QKV), HuggingFace GPT-NeoX default initialization (lambda_0 = 0.01776), batch 24 x seq 512, 200-step warm-up, cosine to 0.1 x peak, peak learning rate 3e-4, AdamW (0.9, 0.999, eps 1e-8), decoupled weight decay 0.01, no gradient clipping, fp32, random 512-token windows sampled with replacement, 8,000 steps (T = 98.3M tokens), one seed per cell. Differences from the LLaMA-style shared protocol: architecture and initialization, weight decay (0.01 vs 0.1), peak learning rate (3e-4 vs 1e-3), batching (random windows vs sequential stream).

## 6. Numbers removed from Appendix B.1 (2026-09-20)

The former subsection "The normalized-shape band, per kind and per setting" was replaced by a short calibration paragraph. Its per-kind and per-setting ranges are recoverable from the read-out files below; the removed text is kept verbatim in `figure_v2_removed_B11_260920.tex` beside `p5_compile/`.

| Removed statement | Source file | Keys |
|---|---|---|
| Controlled grid, 1,440 one-axis fits: per-kind k_row ranges (q 1.198–1.251, k 1.194–1.251, gate 1.187–1.211, up 1.169–1.211), pooled k_raw down to 0.92 | `06_hrow/kmix_bridge_v1_1.json` | per-matrix `k_row`, `k_raw` by kind |
| Controlled grid, 2,016 two-sided fits: per-kind k_bi medians 1.204–1.211 at every checkpoint; per-matrix k_bi 1.18–1.27; D3 v lowest block k_row 1.09 → k_bi 1.19 | `DATA/p5v2_twoaxis_bridge_v1.json` | `k_bi_range_by_family_step`, `k_bi_median`, `per_matrix` |
| Public Pythia endpoints: identity-axis medians 1.193–1.205 at every size; v 1.202–1.204 only two-sided; two raw q fits below the R² gate; q blocks up to 1.28 at 410M | `DATA/p5v2_figure_readouts_v1.json` | public-endpoint block |
| Initialization families: two-sided core 1.195–1.219 (Gaussian) at every checkpoint/layer/kind; Laplace/uniform start 1.00/1.37, within 0.01 of Gaussian by step 3,200 except Laplace v (step 10,000); per-matrix 1.191–1.218 at 30k; profile deviation 40.5%→0.09% (Laplace), 38.1%→0.25% (uniform), Gaussian ≤0.26%; 10–90% spread ±1% over 378 matrices | `DATA/p5v2_ea_kbi_trajectory_v1.json` | family × step medians and ranges |
| Pythia size series, 12 endpoints: k_row 1.194–1.217 at all three sizes | size-series read-out listed in §2 | per-run `k_row` |
| Single structured-data 70M run to 30k steps: q/k k_row 1.197–1.214 at all eight checkpoints while raw k moves 1.19→1.17 | `06_hrow/ea_mulhk_v1.json` | `med_traj` |
