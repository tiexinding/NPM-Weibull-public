# NPM-Weibull

[![arXiv](https://img.shields.io/badge/arXiv-2605.18898-b31b1b.svg)](https://arxiv.org/abs/2605.18898)
[![PyPI](https://img.shields.io/pypi/v/npm-weibull-py.svg)](https://pypi.org/project/npm-weibull-py/)
[![Python](https://img.shields.io/pypi/pyversions/npm-weibull-py.svg)](https://pypi.org/project/npm-weibull-py/)
[![HF Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-NPM--Weibull--DATABASE--v9__1-yellow.svg)](https://huggingface.co/datasets/TiexinDing/NPM-Weibull-DATABASE-v9_1)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-blue.svg)](LICENSE)

Companion code and benchmark database for the paper:

> **A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions**
> Tiexin Ding (Independent Researcher)
> [arXiv:2605.18898](https://arxiv.org/abs/2605.18898) ([doi:10.48550/arXiv.2605.18898](https://doi.org/10.48550/arXiv.2605.18898))

## Overview

This repository hosts the open-source artifacts described in the paper:

- **`npm-weibull-py` v0.4**: A pip-installable Python library for fitting and benchmarking Weibull `(k, λ)` parameters on transformer weight matrices. Eight diagnostic functions (F1--F8) for cross-family comparison, body--tail ablation, paired-correlation analysis, and architecture classification.
- **`DATABASE_v9_1`** (main cohort): Per-component Weibull fits for **12 model entries** across **7 architectural families** (Pythia 70M/160M/410M/1B/6.9B, OLMo-1, OLMo-2, LLaMA-3, Mistral, Qwen2.5-7B/14B, Qwen3-8B), with per-layer and per-component breakdowns. Also published on the [Hugging Face Hub](https://huggingface.co/datasets/TiexinDing/NPM-Weibull-DATABASE-v9_1) as a `datasets`-loadable artifact for streaming / `load_dataset()` workflows.
- **`DATABASE_v9_1` Qwen-cohort companion**: 11-entry Qwen-family cohort (Qwen2 / Qwen2.5 / Qwen3, sizes 1.5B–14B, depths 28L/36L/48L, includes 4 base-vs-Math-CPT pairs) used in the diagnostic application case (Appendix C). Surfaces the shallow-FFN bimodal anomaly characteristic of 7B+ Qwen entries. See [`database_v9_1/DATABASE_v9_1_qwen_cohort.md`](database_v9_1/DATABASE_v9_1_qwen_cohort.md).
- **Reproducibility examples** (planned): Jupyter notebooks reproducing key paper figures.

## Status

**v0.4.0 released** (May 2026): library, benchmark database, examples, tests, and pip-install all available.

| Component | Status |
|---|---|
| Paper information and citation | ✅ Available ([arXiv:2605.18898](https://arxiv.org/abs/2605.18898)) |
| `npm-weibull-py` v0.4 library source | ✅ Available (`npm_weibull/`) |
| `DATABASE_v9_1` main cohort (12 entries) | ✅ Available (Python module + CSV) |
| `DATABASE_v9_1` Qwen-cohort companion (11 entries) | ✅ Available ([`database_v9_1/DATABASE_v9_1_qwen_cohort.md`](database_v9_1/DATABASE_v9_1_qwen_cohort.md)) |
| `DATABASE_v9_1` on Hugging Face Hub | ✅ Available ([`TiexinDing/NPM-Weibull-DATABASE-v9_1`](https://huggingface.co/datasets/TiexinDing/NPM-Weibull-DATABASE-v9_1)) |
| Quickstart examples | ✅ Available (`examples/`, 3 runnable scripts) |
| Tests | ✅ Available (`tests/`, 47 passing, 82% coverage) |
| Pip-installable release on PyPI | ✅ Available ([pypi.org/project/npm-weibull-py](https://pypi.org/project/npm-weibull-py/)) |
| API reference documentation | ✅ Available ([`docs/F1_F8_API.md`](docs/F1_F8_API.md)) |

## Install

```bash
pip install npm-weibull-py

# Optional extras
pip install "npm-weibull-py[torch]"   # transformers + safetensors for checkpoint extraction
pip install "npm-weibull-py[plot]"    # matplotlib for plotting helpers
```

For a development install (clone the repository, edit source, run tests):

```bash
git clone https://github.com/tiexinding/NPM-Weibull-public.git
cd NPM-Weibull-public
pip install -e ".[dev]"   # adds pytest, pytest-cov, ruff, mypy
```

Requires Python ≥ 3.9. Core dependencies are `numpy` and `scipy` only.

## Quick start

```python
from npm_weibull import weibull_fit, DATABASE_v9_1, compare_to_benchmark

# F1 — fit Weibull to a weight magnitude histogram
fit = weibull_fit({"edges": edges, "hist": counts}, trim="mid_80")
print(fit["k"], fit["lambda"], fit["R2"])

# Layer B — compare user-side per-component median k to the 12-entry benchmark
user = {
    "arch": {"arch": "GQA", "n_q": 32, "n_kv": 8},
    "median_k_per_kind": {"q": 1.14, "k": 1.13, "v": 1.19, "o": 1.19},
}
print(compare_to_benchmark(user)["nearest_neighbor"])
```

See `examples/` for three runnable demos covering F1 fit, benchmark comparison, and F3/F5 trajectory decomposition. For the full API surface (15 entries: F1–F8 core + 5 utilities + workflow + benchmark), see [`docs/F1_F8_API.md`](docs/F1_F8_API.md).

## Repository layout

```
NPM-Weibull-public/
├── npm_weibull/           # library (F1-F8 + workflow + benchmark)
│   ├── core/              # F1 weibull, F5 trajectory, F6_ext distfree, F8 architecture, ...
│   ├── utils/             # closed-form, histogram, cascade reader, KS/AIC
│   ├── workflow/          # diagnose_model wrapper (Layer A)
│   └── benchmark/         # DATABASE_v9_1 + compare_to_benchmark (Layer B)
├── tests/                 # synthetic + integration tests (12 passing)
├── examples/              # 01 synthetic fit, 02 benchmark, 03 trajectory
├── database_v9_1/         # populate_database_v9_1.py + generated CSV/MD
├── pyproject.toml         # pip install config (v0.4.0)
└── README.md
```

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
  title         = {A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions},
  author        = {Ding, Tiexin},
  year          = {2026},
  eprint        = {2605.18898},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.18898},
  url           = {https://arxiv.org/abs/2605.18898}
}
```

## License

Code and data in this repository are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license, matching the arXiv submission license.

## Contact

Questions, collaboration, or feedback:

- **Email**: tiexinding@gmail.com
- **GitHub issues**: please use this repository's Issues tab (after content upload)

---

*Repository identifier note: the `NPM-Weibull` name is the stable library and repository identifier introduced in early development. The paper title ("A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions") reflects the framework's empirical, methodology-first identity adopted in the final draft.*
