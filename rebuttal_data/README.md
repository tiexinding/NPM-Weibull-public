# `rebuttal_data/` — data behind the author response

Machine-readable sources for the numbers cited in the author response. File names
carry the short battery codes used during preparation (`c2`, `c3`, `c5`, `loo`,
`s2`); the table below maps each file to the request it answers, the headline
numbers it backs, and where those numbers appear in the revised manuscript.
The companion battery `c1` lives in `../per_matrix_data/` (it is a per-matrix
table and is grouped with the other per-matrix files); it is listed here for
completeness.

| File | Contents | Headline numbers it backs | Where it is used |
|---|---|---|---|
| `c2_truncated_compare_v1.json` | Body-truncated likelihood comparison of Weibull vs. log-normal vs. gamma: binned MLE on the rank-[0.10, 0.90] body (CDF-renormalized), plus held-out quantiles at F ∈ {0.01, 0.05, 0.95, 0.99, 0.999}. Per-matrix records for all 1,220 cohort matrices (fields: `model`, `param_name`, `block`, `kind`, `cls`, per-family NLL and held-out errors). | Per-class body win rates (Weibull first in every component group except MHA W_q/W_k) and the held-out tail ranking; the fused mixed W_qkv row (win 93/5/2). | Appendix A.2 (Tables 7–8, Figure 15). Response to Reviewer Fjkw (reference-family comparison) and Reviewer egMc (Q1a). |
| `c3_k0_protocol_v1.json` | The analytic initialization anchor as a function of trim protocol, k₀(P): probability-plot OLS on exact half-Normal quantiles (n_grid = 200,001); `anchor_table` for central fractions 1.00 → 0.50 (e.g., k₀(0.80) = 1.2054, λ₀/σ = 0.8875); cohort `pairing` of raw k(80)−k(90/100) gaps against anchored residuals. | The protocol-geometry account of the body–tail fit gap: raw k₈₀ − k₁₀₀ ≈ 0.050–0.054 across entries, anchored residuals ≤ 0.004 in magnitude. | Appendix A.1 (Table 6) and Section 3.2. Response to Reviewer Fjkw (protocol dependence of the anchor). |
| `c5_raw_analysis_v2.json` | Raw stored-weight (bf16) verification of the Qwen Regime-B secondary population: 15 matrices, 10⁶-element subsamples gate-checked against the extraction histograms; per-record low-magnitude fraction (threshold 10⁻⁴), log₁₀\|w\| band masses, channel concentration, mode separation (0.4 dex criterion). | Channel-level localization of the secondary population and the significance statements (Bonferroni-corrected across the 15 tests) in Appendix D. | Appendix D (Figure 20). Response to Reviewer Fjkw (raw-weight verification) and Reviewer egMc (RC1). |
| `loo_family_cv_v1.json` | Leave-one-family-out robustness of the FFN k band: per-entry median FFN terminal k for the 12 entries / 7 families; entry-level and family-level CV; per-fold held-out in-band counts. | CV = 0.51% across 12 entries; family-level CV = 0.53%; leave-one-family-out range of the per-entry CV [0.47%, 0.57%]; every held-out family lands in band in all 7 folds. | Section 3.1. Response to Reviewer JzZy (W1), Reviewer Fjkw, and Reviewer egMc (Q2). |
| `pythia_displacement_s2_v1.json` | Initialization-to-terminal matrix displacement for the five Pythia sizes (360 matrices): per-matrix relative Frobenius displacement and flattened init–terminal cosine similarity, organized by stored kind, with extraction gate errors. | The "macroscopic only" reading of the band: large matrix-level displacement and low init–terminal cosine while the fitted body shape stays in band. | Appendix A.6 (Figure 16). Response to Reviewer Fjkw (S2) and Reviewer egMc (MC1). |
| `c1_stats_vs_k_scatter_v1.png` | Per-matrix scatter plots of the fitted shape parameter k against the simple-statistics battery (trimmed CV, IQR/median, MAD/median, full CV, excess kurtosis), one point per training-state matrix (n = 1092; the 128 initialization-checkpoint records in the battery are tabulated in `../per_matrix_data/c1_simple_stats_per_matrix_v1.json` but not plotted), with the exact-Weibull prediction overlaid as a zero-parameter reference curve. Rendered directly from that JSON. | Visualizes the Spearman \|ρ\| = 0.86–0.996 tracking of robust body summaries and the kurtosis exception (\|ρ\| = 0.555). | Response to Reviewer Fjkw (point 1: the requested per-matrix scatter plots). |
| `../per_matrix_data/c1_simple_stats_per_matrix_v1.json` | Simple-statistics battery: per-matrix trimmed CV, IQR/median, MAD/median, and excess kurtosis alongside the fitted (k, λ, R²) for all 1,220 matrices. | Robust body summaries track the fitted shape with Spearman \|ρ\| = 0.86–0.996; kurtosis is the unstable exception. | Appendix A.2 (Figure 14, Table 9). Response to Reviewer Fjkw (simple-statistics comparison) and Reviewer egMc (Q1a). |

Each JSON is self-describing: a `schema` and/or `method` field (or equivalent
top-level metadata) records the estimator, conventions, and sampling settings
used to produce it. All values quoted in the response documents are computed
from these files.

### Added 2608-14 (completing the number-to-source map)

| File | Response numbers it backs |
|---|---|
| `c5_channel_overlap_v1.json` | Qwen channel overlap: 5,031 of 18,944 intermediate channels (26.6%), gate–down overlap 0.979 of the smaller set, enrichment 9.4× over independence |
| `c5_decomposition_summary_v1.json` | 52%–61% of below-valley mass carried by the fixed suppressed-channel set (gate 0.61 / up 0.54 / down 0.52); suppressed set's own mass 97–98% below valley |
| `pythia160m_cos_trajectory_v2.json` | Pythia-160m time-resolved directions: 10 checkpoints, cosine-to-initialization range 0.05–0.19 at the terminal end |
| `lambda_step_trajectory.json` | Pythia per-checkpoint fitted (k, λ) trajectories behind the time-resolved analysis (k 1.204 → 1.098 for the fused W_qkv; λ growth curves) |
