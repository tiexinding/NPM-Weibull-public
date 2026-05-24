# DATABASE_v9_1 — Reference Table (12 family entries)

**Source**: cascade v3 (per-component Weibull fit, mid-80% trim)
**Generated**: by populate_database_v9_1.py

## 12 entries (5 Pythia size + 7 cross-family) — k median per component

| Entry | Family | Size | Architecture | n_q/n_kv | QK-Norm | tokens | k_q | k_k | k_v | k_o | k_gate | k_up | k_down | low-R²(q) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-70m | Pythia | 70m | MHA-merged | merged | no | 300B | 1.0488 | — | — | 1.1838 | 0.0000 | 0.0000 | 0.0000 | 1/6 |
| pythia-160m | Pythia | 160m | MHA-merged | merged | no | 300B | 1.0982 | — | — | 1.1917 | 0.0000 | 0.0000 | 0.0000 | 0/12 |
| pythia-410m | Pythia | 410m | MHA-merged | merged | no | 300B | 1.1466 | — | — | 1.1988 | 0.0000 | 0.0000 | 0.0000 | 0/24 |
| pythia-1b | Pythia | 1B | MHA-merged | merged | no | 300B | 1.1758 | — | — | 1.1990 | 0.0000 | 0.0000 | 0.0000 | 0/16 |
| pythia-6.9b | Pythia | 6.9B | MHA-merged | merged | no | 300B | 1.1726 | — | — | 1.1986 | 0.0000 | 0.0000 | 0.0000 | 0/32 |
| olmo-1-7b | OLMo-1 | 7B | MHA-separate | 32/32 | no | 2.5T | 0.8123 | 0.7601 | 1.0601 | 1.0409 | 1.2010 | 1.2039 | 1.2041 | 12/32 |
| olmo-2-7b | OLMo-2 | 7B | MHA-separate | 32/32 | yes | 5T | 0.9895 | 0.9716 | 1.1930 | 1.1958 | 1.1976 | 1.2032 | 1.2031 | 2/32 |
| llama-3-8b | Llama-3 | 8B | GQA-4:1 | 32/8 | no | 15T | 1.1352 | 1.1462 | 1.1710 | 1.1841 | 1.1890 | 1.1971 | 1.1931 | 0/32 |
| mistral-7b | Mistral | 7B | GQA-4:1 | 32/8 | no | 8T | 1.1485 | 1.1291 | 1.1702 | 1.1902 | 1.1947 | 1.1964 | 1.1926 | 1/32 |
| qwen2.5-7b | Qwen2.5 | 7B | GQA-7:1 | 28/4 | no | 18T | 1.1328 | 1.1033 | 1.1430 | 1.1665 | 1.1904 | 1.1888 | 1.1830 | 0/28 |
| qwen2.5-14b | Qwen2.5 | 14B | GQA-5:1 | 40/8 | no | 18T | 1.1598 | 1.1346 | 1.1636 | 1.1841 | 1.1909 | 1.1914 | 1.1885 | 0/27 |
| qwen3-8b | Qwen3 | 8B | GQA-4:1 | 32/8 | yes | 36T | 1.1623 | 1.1539 | 1.1581 | 1.1802 | 1.1872 | 1.1886 | 1.1846 | 0/36 |

## Training hyperparameters + T/τ + Physical State (Wang-Aitchison 2024 cycle ratio)

τ_iter = 1/(η · λ_wd) — EMA iteration time-constant.  T/τ_iter = T_steps / τ_iter — completed EMA cycles.

| Entry | η_peak | λ_wd | T_steps | τ_iter | **T/τ** | **Physical State** | hp source |
|---|---|---|---|---|---|---|---|
| pythia-70m | 1.0e-03 | 0.01 | 143000 | 100000 | **1.43** | **Saturated** | explicit |
| pythia-160m | 6.0e-04 | 0.01 | 143000 | 166667 | **0.86** | **Near-saturated** | explicit |
| pythia-410m | 3.0e-04 | 0.01 | 143000 | 333333 | **0.43** | **Approaching** | explicit |
| pythia-1b | 3.0e-04 | 0.01 | 143000 | 333333 | **0.43** | **Approaching** | explicit |
| pythia-6.9b | 1.2e-04 | 0.01 | 143000 | 833333 | **0.17** | **Transition** | explicit |
| olmo-1-7b | 3.0e-04 | 0.1 | 477000 | 33333 | **14.31** | **Saturated** | explicit |
| olmo-2-7b | 3.0e-04 | 0.1 | 600000 | 33333 | **18.00** | **Saturated** | inferred |
| llama-3-8b | 3.0e-04 | 0.1 | 1000000 | 33333 | **30.00** | **Saturated** | inferred |
| mistral-7b | 3.0e-04 | 0.1 | 500000 | 33333 | **15.00** | **Saturated** | estimated |
| qwen2.5-7b | 3.0e-04 | 0.1 | 1100000 | 33333 | **33.00** | **Saturated** | inferred |
| qwen2.5-14b | 3.0e-04 | 0.1 | 1100000 | 33333 | **33.00** | **Saturated** | estimated |
| qwen3-8b | 3.0e-04 | 0.1 | 2200000 | 33333 | **66.00** | **Saturated** | inferred |

**Physical State thresholds** (Wang-Aitchison 2024 cycle ratio):

- Saturated: T/τ ≥ 1.20
- Near-saturated: 0.80 ≤ T/τ < 1.20
- Approaching: 0.40 ≤ T/τ < 0.80
- Partial: 0.25 ≤ T/τ < 0.40
- Transition: T/τ < 0.25

**hp source confidence**:

- *explicit*: paper Table / official tech report directly states the value
- *inferred*: paper §3 states a quantity from which we derive it (e.g. tokens × batch / seq → steps)
- *estimated*: paper does not publish; same-family typical recipe used as fallback

---

## Verification

Per-entry per-component sanity check is recorded in
[`DATABASE_v9_1_report.md`](DATABASE_v9_1_report.md). All entries pass
R² ≥ 0.99 on the Transmission Class components.

**Transmission Class aggregated band** (median across components per
entry, then aggregated across the 12 entries): k ∈ [1.186, 1.204],
cross-family CV = 0.51%. See paper §3 for the strict-band definition
and protocol.

For per-block raw fits and the cascade pipeline that produces this
table, see the `npm-weibull-py` repository on GitHub.

---

## Companion: Qwen-family application cohort (Appendix C)

An 11-entry Qwen-family cohort surfacing the shallow-FFN bimodal
anomaly described in Appendix C of the paper is provided as a separate
companion table:

- [`DATABASE_v9_1_qwen_cohort.csv`](DATABASE_v9_1_qwen_cohort.csv)
- [`DATABASE_v9_1_qwen_cohort.md`](DATABASE_v9_1_qwen_cohort.md)

The Qwen cohort spans 3 generations (Qwen2, Qwen2.5, Qwen3), 5 sizes
(1.5B/3B/7B/8B/14B), 3 depths (28L/36L/48L), and includes 4 base-vs-Math-CPT
pairs. The partition is clean: 4 Regime A (clean) entries are all 1.5B
variants and 7 Regime B (shallow-FFN bimodal) entries are all 7B and above.
Generated by `populate_qwen_cohort.py` from the same cascade-v3 source.
