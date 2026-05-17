# NPM-Weibull

Companion code and benchmark database for the paper:

> **A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions**
> Tiexin Ding (Independent Researcher)
> arXiv:2605.XXXXX (link to be updated after announcement; submission ID 7603867)

## Overview

This repository hosts the open-source artifacts described in the paper:

- **`npm-weibull-py` v0.4**: A pip-installable Python library for fitting and benchmarking Weibull `(k, λ)` parameters on transformer weight matrices. Eight diagnostic functions (F1--F8) for cross-family comparison, body--tail ablation, paired-correlation analysis, and architecture classification.
- **`DATABASE_v9_1`**: Per-component Weibull fits for **12 model entries** across **7 architectural families** (Pythia 70M/160M/410M/1B/6.9B, OLMo-1, OLMo-2, LLaMA-3, Mistral, Qwen2.5-7B/14B, Qwen3-8B), with per-layer and per-component breakdowns.
- **Reproducibility examples** (planned): Jupyter notebooks reproducing key paper figures.

## Status

🚧 **Initial release in progress** (May 2026).

This repository is being populated alongside the arXiv submission of the paper. Current state:

| Component | Status |
|---|---|
| Paper information and citation | ✅ Available |
| `npm-weibull-py` library source | 🚧 Coming next 1--2 days |
| `DATABASE_v9_1` benchmark JSON files | 🚧 Coming next 1--2 days |
| Reproducibility example notebooks | 🚧 Planned |
| Pip-installable release on PyPI | 🚧 Planned |

For now, please reference the paper for methodology details. Code and data will be uploaded shortly.

## Quick Reference (from the paper)

### Initialization anchor (Appendix A.1)
Half-Normal initialization yields a deterministic Weibull `(k₀, λ₀)` anchor under middle-80% probability-plot fit:

- `k₀ ≈ 1.2054` (universal across vendors and σ_init scales)
- `λ₀ ≈ 0.8875 · σ_init` (initialization-scheme-specific)

Verified at step-0 across 5 Pythia sizes within 0.13% relative error.

### Two functional classes (Section 2.2)

- **Transmission Class** (`W_o`, FFN modules `W_gate`, `W_up`, `W_down` for SwiGLU; `W_FFN_in`, `W_FFN_out` for GeLU): the shape parameter `k` stays within the band `[1.186, 1.204]` across architectures (cross-family CV = 0.51%, n = 12 entries).
- **Selection Class** (`W_q`, `W_k`): departs from the Weibull anchor during training; departure severity tracks attention storage architecture:
  - Separately-stored MHA (OLMo-1, OLMo-2): `k ∈ [0.76, 0.99]` (deep Selection)
  - GQA (LLaMA-3, Mistral, Qwen2.5, Qwen3): `k ∈ [1.10, 1.16]` (mild Selection)
  - Merged `W_qkv` (Pythia): `k ∈ [1.05, 1.18]` (transitional, tracks `T/τ` monotonically)

### λ scaling within Pythia (Section 5.4)
Terminal mean `λ` across the three Transmission Class kinds scales with `√(η/λ_wd)`:

- Pearson `r = 0.94` (n = 5 Pythia sizes)
- Linear fit through origin: `λ = 0.087 · √(η/λ_wd)`

Directionally consistent with the AdamW steady-state scaling analysis of Fan et al. (2025).

## Citation

```bibtex
@misc{ding2026weibull,
  title={A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions},
  author={Ding, Tiexin},
  year={2026},
  eprint={2605.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.XXXXX}
}
```

(BibTeX entry will be finalized with the final arXiv ID once the paper is announced.)

## License

Code and data in this repository are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license, matching the arXiv submission license.

## Contact

Questions, collaboration, or feedback:

- **Email**: tiexinding@gmail.com
- **GitHub issues**: please use this repository's Issues tab (after content upload)

---

*Repository identifier note: the `NPM-Weibull` name is the stable library and repository identifier introduced in early development. The paper title ("A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions") reflects the framework's empirical, methodology-first identity adopted in the final draft.*
