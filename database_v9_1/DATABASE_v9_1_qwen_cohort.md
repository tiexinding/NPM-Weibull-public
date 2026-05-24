# `DATABASE_v9_1` Qwen Cohort — Application Case (Appendix C)

11-entry Qwen-family cohort used in the diagnostic application case (Appendix C of arXiv:2605.18898). Surfaces the shallow-FFN bimodal anomaly in 7B+ Qwen entries.

## Schema

- `regime`: A (clean, shallow min $k \geq 1.0$) or B (bimodal, shallow min $k \leq 0.7$)
- `shallow_median_k_gate`: median $k$ for $W_\text{gate}$ across blocks 0–5
- `shallow_min_k`: minimum block-level $k$ across $W_\text{gate}, W_\text{up}, W_\text{down}$ within blocks 0–5
- `deep_median_k_gate`: median $k$ for $W_\text{gate}$ across blocks ≥ 10
- `k_median_*`, `lambda_median_*`: per-kind terminal-checkpoint medians across all layers

## Table (paper Table 7 schema)

| Entry | Layers | Tokens | Shal med $k_g$ | Shal min $k$ | Deep med $k_g$ | Regime |
|---|---:|---|---:|---:|---:|:---:|
| Qwen2-1.5B-base | 28 | 7T | 1.191 | 1.179 | 1.182 | A |
| Qwen2.5-1.5B | 28 | 18T | 1.178 | 1.167 | 1.174 | A |
| Qwen2-Math-1.5B | 28 | 7T + Math CPT | 1.194 | 1.187 | 1.191 | A |
| Qwen2.5-Math-1.5B | 28 | 18T + Math CPT | 1.195 | 1.117 | 1.19 | A |
| Qwen2-7B | 28 | 7T | 0.958 | 0.526 | 1.19 | B |
| Qwen2.5-7B | 28 | 18T | 0.969 | 0.566 | 1.189 | B |
| Qwen2-Math-7B | 28 | 7T + 200B Math | 0.958 | 0.519 | 1.197 | B |
| Qwen2.5-Math-7B | 28 | 18T + 700B Math | 1.009 | 0.619 | 1.196 | B |
| Qwen2.5-3B | 36 | 18T | 0.653 | 0.571 | 1.181 | B |
| Qwen3-8B | 36 | 36T | 0.692 | 0.341 | 1.189 | B |
| Qwen2.5-14B | 48 | 18T | 0.439 | 0.398 | 1.193 | B |

## Full per-kind medians (terminal checkpoint)

| Entry | $k_q$ | $k_k$ | $k_v$ | $k_o$ | $k_\text{gate}$ | $k_\text{up}$ | $k_\text{down}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2-1.5B-base | 1.1166 | 1.1142 | 1.1438 | 1.1416 | 1.1856 | 1.1876 | 1.1769 |
| Qwen2.5-1.5B | 1.0947 | 1.1155 | 1.1377 | 1.1375 | 1.1777 | 1.1796 | 1.1688 |
| Qwen2-Math-1.5B | 1.1277 | 1.1083 | 1.1446 | 1.1556 | 1.1931 | 1.1905 | 1.1874 |
| Qwen2.5-Math-1.5B | 1.1256 | 1.1011 | 1.1468 | 1.1564 | 1.1912 | 1.1881 | 1.1859 |
| Qwen2-7B | 1.1366 | 1.1096 | 1.1456 | 1.166 | 1.1895 | 1.1892 | 1.1838 |
| Qwen2.5-7B | 1.1321 | 1.1027 | 1.1422 | 1.1659 | 1.1882 | 1.1882 | 1.1825 |
| Qwen2-Math-7B | 1.1545 | 1.1249 | 1.1523 | 1.1808 | 1.1971 | 1.1946 | 1.1935 |
| Qwen2.5-Math-7B | 1.1547 | 1.1195 | 1.1515 | 1.1791 | 1.1959 | 1.1934 | 1.1916 |
| Qwen2.5-3B | 1.1277 | 1.12 | 1.1484 | 1.161 | 1.1781 | 1.1813 | 1.1747 |
| Qwen3-8B | 1.1623 | 1.1539 | 1.1581 | 1.1802 | 1.1872 | 1.1886 | 1.1846 |
| Qwen2.5-14B | 1.1638 | 1.1413 | 1.1628 | 1.1903 | 1.1929 | 1.1926 | 1.1897 |

## Citation

```bibtex
@misc{ding2026weibull,
  title={A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions},
  author={Ding, Tiexin},
  year={2026},
  eprint={2605.18898},
  archivePrefix={arXiv}
}
```
