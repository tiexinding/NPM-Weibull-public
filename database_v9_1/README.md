---
license: cc-by-4.0
task_categories:
  - other
language:
  - en
size_categories:
  - n<1K
tags:
  - weibull
  - transformer
  - weight-statistics
  - model-diagnostics
  - npm-weibull
pretty_name: NPM-Weibull DATABASE v9_1
configs:
  - config_name: default
    data_files: DATABASE_v9_1.csv
---

# NPM-Weibull DATABASE v9_1

Benchmark database of Weibull (k, λ) fits for **12 transformer model entries** spanning **7 architectural families** (Pythia, OLMo-1/2, LLaMA-3, Mistral, Qwen2.5, Qwen3) — 70M–14B parameters, GeLU and SwiGLU activations, Pre-LN and QK-Norm placements.

- 📄 **Paper**: [A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions](https://arxiv.org/abs/2605.18898) (arXiv:2605.18898)
- 📦 **Source repo (primary)**: [github.com/tiexinding/NPM-Weibull-public](https://github.com/tiexinding/NPM-Weibull-public) — directory `database_v9_1/`
- 🔧 **Python library**: `pip install npm-weibull-py` — [PyPI](https://pypi.org/project/npm-weibull-py/)

---

## 📌 Source of truth

The **primary, authoritative location** of this dataset is the GitHub repository above (directory `database_v9_1/`). This Hugging Face Dataset is a **mirror** synced from GitHub on each tagged release, provided for `load_dataset(...)` convenience and Dataset Viewer discoverability.

- Updates are **batched per release**, not per commit — for the latest state, watch the GitHub repo.
- Issues and pull requests: please file on [GitHub Issues](https://github.com/tiexinding/NPM-Weibull-public/issues) rather than HF discussions (faster response).
- License, citation, and documentation are kept consistent across both surfaces.

---

## Quick start

### Option 1 — `load_dataset` (Hugging Face)

```python
from datasets import load_dataset

ds = load_dataset("TiexinDing/NPM-Weibull-DATABASE-v9_1")
print(ds["train"][0])  # First entry: pythia-70m
```

### Option 2 — pandas

```python
import pandas as pd

df = pd.read_csv("DATABASE_v9_1.csv")
print(df[["entry_id", "k_median_o", "lambda_median_o"]].head())
```

### Option 3 — `npm-weibull-py` library (recommended for analysis)

```python
from npm_weibull import DATABASE_v9_1, compare_to_benchmark

print(DATABASE_v9_1)  # 12 entries with per-component fits

cmp = compare_to_benchmark({
    "median_k_per_kind": {"q": 1.14, "k": 1.13, "v": 1.19, "o": 1.19}
})
print(cmp["nearest_neighbor"])  # nearest of the 12 benchmark entries
```

---

## Dataset structure

### 12 entries

| `entry_id`    | Family   | Size  | Architecture  | QK-Norm | Tokens |
|---------------|----------|-------|---------------|---------|--------|
| pythia-70m    | Pythia   | 70m   | MHA-merged    | False   | 300B   |
| pythia-160m   | Pythia   | 160m  | MHA-merged    | False   | 300B   |
| pythia-410m   | Pythia   | 410m  | MHA-merged    | False   | 300B   |
| pythia-1b     | Pythia   | 1B    | MHA-merged    | False   | 300B   |
| pythia-6.9b   | Pythia   | 6.9B  | MHA-merged    | False   | 300B   |
| olmo-1-7b     | OLMo-1   | 7B    | MHA-separate  | False   | 2.5T   |
| olmo-2-7b     | OLMo-2   | 7B    | MHA-separate  | True    | 5T     |
| llama-3-8b    | LLaMA-3  | 8B    | GQA-4:1       | False   | 15T    |
| mistral-7b    | Mistral  | 7B    | GQA-4:1       | False   | 8T     |
| qwen2.5-7b    | Qwen2.5  | 7B    | GQA-7:1       | False   | 18T    |
| qwen2.5-14b   | Qwen2.5  | 14B   | GQA-5:1       | False   | 18T    |
| qwen3-8b      | Qwen3    | 8B    | GQA-4:1       | True    | 36T    |

### Columns (55 total)

**Identifiers**: `entry_id`, `family`, `size`, `arch`

**Architecture**: `n_q`, `n_kv` (head counts for GQA), `qk_norm`

**Training hyperparameters**: `training_tokens`, `eta_peak` (peak learning rate), `lambda_wd` (weight decay), `T_steps`, `tau_iter = 1/(η·λ_wd)`, `T_over_tau` (Wang-Aitchison 2024 cycle ratio), `Physical_State`, `hp_confidence` (explicit / inferred / estimated)

**Per-component Weibull fits** (median across blocks within each model):

- Pythia entries (GeLU 2-projection FFN with merged W_qkv): populate `qkv`, `o`, `ffn_in`, `ffn_out`
- Non-Pythia entries (SwiGLU 3-projection FFN with separate Q/K/V): populate `q`, `k`, `v`, `o`, `gate`, `up`, `down`

Each component has four columns: `k_median_*`, `lambda_median_*`, `R2_median_*`, `R2_below_95_*` (count of blocks with R² < 0.95).

`n_records` = total number of per-block Weibull fits aggregated.

---

## Key findings (paper §3)

1. **FFN reference band**: the **median terminal k across the FFN matrices per entry**, aggregated across the 12 entries, falls in **[1.186, 1.204]** with cross-family CV = **0.51%** — the paper's strict band estimand ($W_v$/$W_o$ behave similarly in most entries but are not part of the band; see paper Section 3). Shared across SwiGLU and GeLU activations, Pre-LN and QK-Norm placements, and model sizes 70M → 14B.

2. **Selection Class** (W_q, W_k): k departs from initialization anchor (k_init ≈ 1.205), severity tracks attention storage architecture — **separately-stored MHA** (OLMo-1/2) deepest at k ∈ [0.76, 0.99], **GQA** (LLaMA-3, Mistral, Qwen2.5/3) milder at [1.10, 1.16], **Pythia merged W_qkv** transitional at [1.05, 1.18].

3. **λ scaling**: terminal λ ∝ √(η/λ_wd) within Pythia, Pearson r = 0.94 across the three Transmission Class component kinds, directionally consistent with Fan et al. (2025) AdamW steady-state analysis.

---

## Files in this dataset

| File | Purpose |
|---|---|
| `DATABASE_v9_1.csv` | Machine-readable dataset (56 columns × 12 entries) |
| `DATABASE_v9_1.md` | Human-readable reference table |
| `DATABASE_v9_1_report.md` | Per-entry sanity verification |
| `populate_database_v9_1.py` | Internal regeneration script (dev-only; requires cascade v3 raw per-block fits not shipped here — see GitHub repo for the cascade pipeline) |
| `LICENSE` | CC BY 4.0 license text |
| `README.md` | This file |

---

## License

[CC BY 4.0](LICENSE) — free for academic and commercial use with attribution.

---

## Citation

```bibtex
@article{ding2026twoparameterweibull,
  title   = {A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions},
  author  = {Ding, Tiexin},
  journal = {arXiv preprint arXiv:2605.18898},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.18898},
  url     = {https://arxiv.org/abs/2605.18898}
}
```

---

**Author**: Tiexin Ding · NeuralCAE

Issues / questions: [GitHub Issues](https://github.com/tiexinding/NPM-Weibull-public/issues)
