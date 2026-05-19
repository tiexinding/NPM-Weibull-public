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

**Physical State thresholds** (B2 spec aligned with §A.8 v7 T3 table):
- Saturated: T/τ ≥ 1.20
- Near-saturated: 0.80 ≤ T/τ < 1.20
- Approaching: 0.40 ≤ T/τ < 0.80
- Partial: 0.25 ≤ T/τ < 0.40
- Transition: T/τ < 0.25

**hp source confidence**:
- *explicit*: paper Table / official tech report 直接给出
- *inferred*: paper §3 给出但需要导出 (e.g. tokens × batch / seq → steps)
- *estimated*: paper 不公开, 用同 family typical recipe 估算

---

## §A.8.4 cap stone cross-verify (B2 实测 vs §A.8 v7 文本)

| Model | Arch | A §A.8.4 written | B2 measured (k_median_q / k_median_k) | Match? |
|---|---|---|---|---|
| OLMo-1 7B | MHA | 0.81 / 0.76 (strong) | 0.812 / 0.760 | ✅ |
| OLMo-2 7B | MHA+QKN | 0.99 / 0.97 (mild) | 0.990 / 0.972 | ✅ |
| Llama-3 8B | GQA 4:1 | 1.14 / 1.15 (transmission) | 1.135 / 1.146 | ✅ |
| Mistral 7B | GQA 4:1 | 1.15 / 1.13 (transmission) | 1.149 / 1.129 | ✅ |
| Qwen2.5 7B | GQA 5:1 | 1.16 / 1.13 (transmission) | 1.133 / 1.103 | ⚠️ |
| Qwen3 8B | GQA 4:1 | 1.16 / 1.15 (transmission) | 1.162 / 1.154 | ✅ |

*Tolerance: |Δk| < 0.025 ≈ 2.1% relative.*

---

## §A.7 transmission strict band cross-verify (W_v / W_o / W_up / W_down)

§A.7 v1 写: 跨 12 family, 训练后 k 严格落入 **[1.186, 1.204]**, CV = **0.51%** (paper §3 final 数据).

| Entry | k_v | k_o | k_up | k_down |
|---|---|---|---|---|
(W_v 仅 separate-Q/K family; Pythia 5 size W_v 在 merged W_qkv 内, 单独无 v median;
Pythia transmission FFN 用 W_ffn_in / W_ffn_out; separate-Q/K 用 W_up / W_down)

| Entry | k_v | k_o | k_up / ffn_in | k_down / ffn_out |
|---|---|---|---|---|
| pythia-70m | — | 1.1838 | 1.1903 | 1.1898 |
| pythia-160m | — | 1.1917 | 1.1927 | 1.1953 |
| pythia-410m | — | 1.1988 | 1.1991 | 1.2011 |
| pythia-1b | — | 1.1990 | 1.1997 | 1.1993 |
| pythia-6.9b | — | 1.1986 | 1.2026 | 1.2010 |
| olmo-1-7b | 1.0601 | 1.0409 | 1.2039 | 1.2041 |
| olmo-2-7b | 1.1930 | 1.1958 | 1.2032 | 1.2031 |
| llama-3-8b | 1.1710 | 1.1841 | 1.1971 | 1.1931 |
| mistral-7b | 1.1702 | 1.1902 | 1.1964 | 1.1926 |
| qwen2.5-7b | 1.1430 | 1.1665 | 1.1888 | 1.1830 |
| qwen2.5-14b | 1.1636 | 1.1841 | 1.1914 | 1.1885 |
| qwen3-8b | 1.1581 | 1.1802 | 1.1886 | 1.1846 |

**B2 实测 transmission band ranges**:
- W_v: [1.0601, 1.1930]  (CV=3.73%)
- W_o: [1.0409, 1.1990]  (CV=3.71%)
- W_up: [1.1886, 1.2039]  (CV=0.47%)
- W_down: [1.1830, 1.2041]  (CV=0.60%)

- **Combined band**: [1.0409, 1.2041]  (CV=2.74%)
