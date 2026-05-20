# `npm-weibull-py` v0.4 — F1–F8 API Reference

Companion code documentation for the paper
[*A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions*](https://arxiv.org/abs/2605.18898).
This reference consolidates the eight diagnostic functions (F1–F8) introduced in
paper Section 2 and Appendix B, plus the supporting utilities, workflow wrapper,
and benchmark database that ship with the library.

```bash
pip install npm-weibull-py
```

```python
import npm_weibull
print(npm_weibull.__version__)              # '0.4.0'
print(len(npm_weibull.DATABASE_v9_1))       # 12
```

---

## Conventions

Throughout this document:
- `|w|` denotes the element-wise absolute value of a weight matrix entry.
- A *histogram* over `|w|` is binned in `log10|w|` space. The library's standard
  binning is 1024 bins over `[-12, 2]` (cascade v3 convention); see
  [extract_to_histogram](#extract_to_histogram) for how to produce one.
- All functions return plain Python dicts so they are JSON-serialisable.
- The `(k, λ)` Weibull anchor at initialisation is `k0 ≈ 1.2054` under
  middle-80 % probability-plot fit on `|w| ~ HalfNormal(σ_init)` data;
  `λ0 ≈ 0.8875 · σ_init`. See paper Appendix A.1.

---

## Table of contents

**Core diagnostics (paper Sec. 2 / App. B)**

| ID  | Function | Module | One-line |
|-----|----------|--------|----------|
| F1  | [`weibull_fit`](#weibull_fit)                          | `core.weibull`      | Fit Weibull `(k, λ, R², KS)` from a histogram |
| F2  | [`classify_k`](#classify_k)                            | `core.classify`     | Classify `k` into transmission / mild / strong / init |
| F3  | [`sigma_decompose`](#sigma_decompose)                  | `core.trajectory`   | σ growth attribution: λ-dominated vs k-dominated vs mixed |
| F5  | [`k_drift_severity`](#k_drift_severity)                | `core.trajectory`   | Severity label for `k_init → k_trained` drift |
| F6  | [`compute_T_tau`](#compute_t_tau)                      | `core.training`     | Wang–Aitchison `τ_iter` cycle ratio + physical-state |
| F6e | [`per_block_metrics`](#per_block_metrics)              | `core.distfree`     | Distribution-free heaviness (Q90/Q10, P999/P50, Gini) |
| F8  | [`classify_attention_arch`](#classify_attention_arch)  | `core.architecture` | MHA / GQA / MQA classifier from head counts |

> *Numbering convention: F4 and F7 are intentionally absent — they exist in
> earlier internal specs but were merged into F3/F5 and F6_ext, respectively,
> before public release. F6_ext is the distribution-free extension of F6.*

**Utilities (paper App. B.2)**

| Function | Module | One-line |
|----------|--------|----------|
| [`extract_to_histogram`](#extract_to_histogram) | `utils.histogram`        | Weight tensor → 1024-bin log10\|w\| histogram (optionally save NPZ) |
| [`sigma_from_k_lambda`](#sigma_from_k_lambda)   | `utils.closed_form`      | Closed-form σ / mean / median / CV from `(k, λ)` |
| [`weibull_quantile`](#weibull_quantile)         | `utils.closed_form`      | Closed-form `Q(q)` from `(k, λ)` |
| [`compare_distributions`](#compare_distributions) | `utils.ks_aic`         | KS / AIC ranking among Weibull / Lognormal / Gamma |
| [`load_cascade_v3`](#load_cascade_v3)           | `utils.cascade_reader`   | Read cascade-v3 `fit_per_component_v3.json` directory |

**Workflow + benchmark (paper Sec. 6)**

| Function / object | Module | One-line |
|-------------------|--------|----------|
| [`diagnose_model`](#diagnose_model)                | `workflow.diagnose`            | One-shot Layer A diagnostic chain (no benchmark coupling) |
| [`DATABASE_v9_1`](#database_v9_1)                  | `benchmark.database_v9_1`      | 12 reference model entries across 7 architectural families |
| [`compare_to_benchmark`](#compare_to_benchmark)    | `benchmark.database_v9_1`      | Layer B utility: nearest-neighbour in `DATABASE_v9_1` |

---

## Core diagnostics

### `weibull_fit`

**F1 — Weibull `(k, λ, R², KS)` fit on a weight magnitude histogram.**

```python
from npm_weibull import weibull_fit
fit = weibull_fit(histogram, trim="mid_80")
```

Algorithm: weighted least-squares on the linearised log-log Weibull plot
`log(-ln(1 - F)) = k · log(x) - k · log(λ)`, where `F` is the empirical CDF on
bin midpoints. The middle-80 % trim (paper Appendix A.7) drops the lowest and
highest 10 % of mass before fitting; this is the protocol used throughout the
paper.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `histogram` | `str \| Path \| dict \| tuple` | NPZ file path with keys `edges`/`hist`, or a dict with the same keys, or an `(edges, hist)` tuple. `edges` is the length-`n+1` array of `log10|w|` bin boundaries; `hist` is the length-`n` count per bin. |
| `trim` | `str = "mid_80"` | Body-trim mode: `"mid_80"` (default, paper §A.7 standard), `"mid_90"` (drop 5 %/5 %), or `"full_range"`. |

**Returns** — `dict[str, Any]`

| Key | Type | Description |
|-----|------|-------------|
| `k` | `float` | Weibull shape parameter (≈ 1.205 at Half-Normal init). |
| `lambda` | `float` | Weibull scale parameter. |
| `R2` | `float` | Goodness-of-fit on the weighted log-log plot, ∈ [0, 1]. |
| `KS` | `float` | Kolmogorov–Smirnov statistic between empirical and fitted CDF. |
| `n_used` | `int` | Number of bins entering the fit after trim. |
| `trim` | `str` | Echo of input trim mode. |
| `ok` | `bool` | `True` iff fit converged with `k > 0`, `λ > 0`, and `R²` finite. |
| `reason` | `str` (only when `ok=False`) | Diagnostic string explaining the failure. |

**Example**

```python
import numpy as np
from npm_weibull import weibull_fit

w = np.abs(np.random.default_rng(0).normal(0, 0.02, size=200_000))
edges = np.linspace(-6.0, 1.0, 1025)
hist, _ = np.histogram(np.log10(w + 1e-12), bins=edges)
fit = weibull_fit({"edges": edges, "hist": hist.astype(float)})
print(fit["k"])           # ~1.205 (paper §A.1 anchor)
print(fit["lambda"])      # ~0.8875 · σ_init
```

---

### `classify_k`

**F2 — Discrete regime label from a single `k` value.**

```python
from npm_weibull import classify_k
cls = classify_k(k, R2=None, arch_type=None, is_init=False)
```

Bins `k` into one of four regimes based on the thresholds calibrated against
the 12-entry DATABASE\_v9\_1 cohort (paper §A.7 / §A.8.3):

- **transmission**:     `k ∈ [1.18, 1.21]` and `R² ≥ 0.95`
- **mild_selection**:   `0.95 ≤ k < 1.18`
- **strong_selection**: `k < 0.95`
- **init_baseline**:    `|k - 1.205| < 0.01` and `is_init=True`

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `k` | `float` | Weibull shape parameter. Must be positive. |
| `R2` | `float \| None = None` | Optional R² of the fit. If given and the value would otherwise land in *transmission*, requires `R² ≥ 0.95` to confirm. |
| `arch_type` | `str \| None = None` | Optional `"MHA"` / `"GQA"` / `"MQA"` for the `in_expected_range` check. |
| `is_init` | `bool = False` | Step-0 marker. When `True`, returns `"init_baseline"` if `k` is within tolerance of 1.205. |

**Returns** — `dict[str, Any]`

| Key | Type | Description |
|-----|------|-------------|
| `class` | `str` | Regime label (one of the four above). |
| `in_expected_range` | `bool \| None` | Whether the class matches the expected drift behaviour for `arch_type` (paper §3); `None` if `arch_type` not provided. |
| `deviation_from_init` | `float` | `k - 1.205`. |
| `deviation_pct` | `float` | `(k - 1.205) / 1.205 × 100`. |

---

### `sigma_decompose`

**F3 — σ growth attribution along a trajectory.**

```python
from npm_weibull import sigma_decompose
res = sigma_decompose(k_traj, lambda_traj, paired_lambda_traj=None)
```

Given parallel trajectories of `k` and `λ` over training steps, attributes the
implied σ growth to either λ-dominated, k-dominated, or mixed regimes (paper
§5). Uses the closed form `σ = λ · √(Γ(1+2/k) − Γ²(1+1/k))` from
[`sigma_from_k_lambda`](#sigma_from_k_lambda) under the hood.

Attribution thresholds (paper §3.1):

- **lambda_dominated**: `|k_drift_pct| < 1 %`
- **k_dominated**:      `|k_drift_pct| > 30 %`
- **mixed**:            otherwise

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `k_traj` | `Sequence[float] \| np.ndarray` | `k` values across training steps, length ≥ 2. |
| `lambda_traj` | `Sequence[float] \| np.ndarray` | `λ` values, same length as `k_traj`. |
| `paired_lambda_traj` | `Sequence[float] \| np.ndarray \| None = None` | Optional `λ` trajectory of another component; if provided, the log-log Pearson `r` between the two `λ` trajectories is returned (paper F156 cap stone: paired `r ≈ 0.997` within saturated Transmission Class). |

**Returns** — `dict[str, Any]`

| Key | Type | Description |
|-----|------|-------------|
| `k_init`, `k_final` | `float` | First and last entries of `k_traj`. |
| `k_drift_pct` | `float` | `(k_final − k_init) / |k_init| × 100`. |
| `lambda_init`, `lambda_final` | `float` | First and last entries of `lambda_traj`. |
| `lambda_growth_pct` | `float` | `(λ_final − λ_init) / |λ_init| × 100`. |
| `sigma_init`, `sigma_final` | `float` | Closed-form Weibull σ at the endpoints. |
| `sigma_attribution` | `str` | `"lambda_dominated"` / `"k_dominated"` / `"mixed"`. |
| `paired_r` | `float \| None` | Pearson `r` on log-log paired λ (if `paired_lambda_traj` given), else `None`. Also `None` when either trajectory has zero variance. |

---

### `k_drift_severity`

**F5 — Severity classifier for the `k_init → k_trained` drift.**

```python
from npm_weibull import k_drift_severity
res = k_drift_severity(k_init=1.205, k_trained=med_k)
```

Maps the relative drift `|k_trained − k_init| / k_init × 100` to a discrete
severity label:

- **invariant**: `< 1 %`  (Transmission Class signature)
- **mild**:      `1 %–30 %`
- **strong**:    `> 30 %`  (Selection Class departure)

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `k_init` | `float` | Initialisation `k` (typically 1.205 for Half-Normal init). |
| `k_trained` | `float` | Post-training `k`. |

**Returns** — `dict[str, Any]`

| Key | Type | Description |
|-----|------|-------------|
| `delta_k` | `float` | `k_trained − k_init`. |
| `delta_pct` | `float` | Absolute relative drift in %. |
| `severity` | `str` | `"invariant"` / `"mild"` / `"strong"`. |

---

### `compute_T_tau`

**F6 — Wang–Aitchison `τ_iter` cycle ratio + physical-state classification.**

```python
from npm_weibull import compute_T_tau
res = compute_T_tau(eta=1e-3, lambda_wd=0.01, T_steps=143_000)
```

Computes `τ_iter = 1 / (η · λ_wd)` (the AdamW EMA time constant, Wang &
Aitchison 2024) and the cycle ratio `T/τ_iter`. Classifies the training
state per paper §3.1:

- **saturated**:    `T/τ ≥ 1.0`  (`physical_terminal=True`)
- **approaching**:  `0.2 ≤ T/τ < 1.0`
- **transition**:   `T/τ < 0.2`  (carries a warning that the apparent
  endpoint is a budget artefact, not a physical terminal)

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `eta` | `float` | Peak (or final) learning rate. Must be positive. |
| `lambda_wd` | `float` | Weight-decay coefficient. Must be positive. |
| `T_steps` | `int` | Total training steps. Must be positive. |

**Returns** — `dict[str, Any]`

| Key | Type | Description |
|-----|------|-------------|
| `tau_iter` | `float` | `1 / (η · λ_wd)`, in steps. |
| `T_tau` | `float` | `T_steps / τ_iter`. |
| `state` | `str` | `"saturated"` / `"approaching"` / `"transition"`. |
| `physical_terminal` | `bool` | `True` iff `state == "saturated"`. |
| `warning` | `str \| None` | Diagnostic string if `state == "transition"`. |

---

### `per_block_metrics`

**F6_ext — Distribution-free heaviness metrics (fallback when `R² < 0.95`).**

```python
from npm_weibull import per_block_metrics
res = per_block_metrics(histogram)
```

Computes quantile-based heaviness measures directly from the histogram (no
distributional assumption). Used when the F1 fit is unreliable, e.g. mixture
distributions or Q/K specialisation tails (paper §A.8).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `histogram` | `str \| Path \| dict \| tuple` | Same input format as [`weibull_fit`](#weibull_fit). |

**Returns** — `dict[str, Any]`

| Key | Type | Description |
|-----|------|-------------|
| `q90_q10` | `float` | Q90 / Q10 ratio (Weibull-friendly heaviness, paper §A.8). |
| `q99_q50` | `float` | Severe-tail ratio. |
| `p999_p50` | `float` | Extreme-outlier ratio (paper F150 finding: ~7× more sensitive than Q90/Q10). |
| `gini` | `float` | Gini coefficient via Lorenz curve, ∈ [0, 1]. |
| `median_abs_w` | `float` | Median of `|w|` in the linear domain. |
| `ok` | `bool` | `False` (with `reason`) when total count `< 100`. |

---

### `classify_attention_arch`

**F8 — Attention architecture classifier from head counts.**

```python
from npm_weibull import classify_attention_arch
arch = classify_attention_arch(n_q, n_kv)
```

Architectural classifier independent of weight data:

- **MHA**: `n_kv == n_q` (each query head has its own K/V)
- **GQA**: `1 < n_kv < n_q` and `n_q % n_kv == 0`
- **MQA**: `n_kv == 1`

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `n_q` | `int` | `num_attention_heads` (HF AutoConfig). |
| `n_kv` | `int` | `num_key_value_heads`. For pre-GQA architectures this equals `n_q`. |

`ValueError` is raised if `n_q % n_kv != 0` (non-standard sharing).

**Returns** — `dict[str, Any]`

| Key | Type | Description |
|-----|------|-------------|
| `arch` | `str` | `"MHA"` / `"GQA"` / `"MQA"`. |
| `ratio` | `int` | `n_q // n_kv` (1 for MHA, `n_q` for MQA). |
| `n_q`, `n_kv` | `int` | Echo of inputs. |
| `expected_q_k_drift` | `str` | Descriptive expected drift band per architecture (paper §3). |
| `cap_stone_alignment` | `bool` | `True` for MHA / GQA (the families measured in paper #1); `False` for MQA (no paper #1 data). |

---

## Utilities

### `extract_to_histogram`

Convert a weight matrix (`torch.Tensor` or `np.ndarray`) into the library's
canonical 1024-bin `log10|w|` histogram, optionally saving to NPZ.

```python
from npm_weibull import extract_to_histogram
result = extract_to_histogram(weight, save_path=None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight` | — | Any object with `.detach().to(...).cpu().numpy()` (torch) or castable via `np.asarray`. |
| `n_bins` | `1024` | Number of bins. |
| `log_w_range` | `(-12.0, 2.0)` | Range of `log10|w|`. |
| `save_path` | `None` | If given, write an NPZ with `edges` / `hist` / `shape` / `param_name`. |

Returns a dict `{edges, hist, n_total, param_name, shape}`. Compatible directly
with [`weibull_fit`](#weibull_fit) and [`per_block_metrics`](#per_block_metrics).

---

### `sigma_from_k_lambda`

Closed-form Weibull moments from `(k, λ)`.

```python
from npm_weibull import sigma_from_k_lambda
res = sigma_from_k_lambda(k, lam)
```

Returns `{sigma, mean_abs_w, median_abs_w, cv, c_k}` where
`sigma = λ · √(Γ(1+2/k) − Γ²(1+1/k))`, `mean_abs_w = λ · Γ(1+1/k)`,
`median_abs_w = λ · (ln 2)^(1/k)`, `cv = σ/mean`, `c_k = σ/λ`.

Raises `ValueError` on non-positive `k` or `λ`.

---

### `weibull_quantile`

Closed-form Weibull quantile.

```python
from npm_weibull import weibull_quantile
x = weibull_quantile(k, lam, q)   # q ∈ (0, 1)
```

Returns `λ · (-ln(1 - q))^(1/k)`. Raises `ValueError` on non-positive `(k, λ)`
or `q` outside `(0, 1)`.

---

### `compare_distributions`

KS / AIC ranking among Weibull, Lognormal, and Gamma.

```python
from npm_weibull import compare_distributions
res = compare_distributions(histogram, candidates=None)
```

Performs binned MLE for each candidate (default: all three), then ranks by AIC.
Returns `{best, aic_ranking, ks_per_dist, params_per_dist}`. Used in paper
§A.7 for robustness analysis.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `histogram` | — | Same format as [`weibull_fit`](#weibull_fit). |
| `candidates` | `None` | Subset of `["weibull", "lognormal", "gamma"]`; default is all three. |

Raises `ValueError` on unknown candidate, insufficient counts (< 100), or
degenerate input (non-positive moments for Gamma init).

---

### `load_cascade_v3`

Read all `*_fit_per_component_v3.json` files from a cascade-v3 derived
directory into a single dict keyed by model-step identifier.

```python
from npm_weibull import load_cascade_v3
data = load_cascade_v3("/path/to/cascade_v3_pull/data/derived/")
```

Returns `dict[str, list[dict]]` where each list is the `per_component` entries
for that model+checkpoint. Files that fail to decode emit a
`warnings.warn(...)` rather than raising.

A companion filter helper, `npm_weibull.utils.cascade_reader.filter_per_component`,
narrows a list by `kind` and/or `block_range`.

---

## Workflow

### `diagnose_model`

**Layer A — one-shot diagnostic, no benchmark coupling.**

```python
from npm_weibull import diagnose_model
report = diagnose_model(
    model_id_or_path="my-model",
    histograms_dir="/path/to/per_layer_npzs/",  # OR derived_dir=
    training_config={"eta": 1e-3, "lambda_wd": 0.01, "T_steps": 100_000},
    arch_config={"n_q": 32, "n_kv": 8},
)
```

Chains F1 + F2 + F5 + F6 + F6_ext + F8 into a single Layer-A report. The
function is *universal*: any transformer checkpoint can be diagnosed without
the benchmark. For benchmark comparison, pipe the result into
[`compare_to_benchmark`](#compare_to_benchmark) (Layer B, separate call).

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id_or_path` | `str` | Identifier used to tag the report. |
| `histograms_dir` | `str \| Path \| None` | Directory of raw 1024-bin NPZ files (one per layer / component). Pass either this OR `derived_dir`. |
| `derived_dir` | `str \| Path \| None` | Cascade-v3 derived directory; if given, skips re-fitting and uses pre-computed `(k, λ, R²)`. |
| `training_config` | `dict \| None` | `{eta, lambda_wd, T_steps}` for F6. If `None`, F6 is skipped. |
| `arch_config` | `dict \| None` | `{n_q, n_kv}` for F8. If `None`, F8 is skipped. |

**Returns** — `dict[str, Any]` (Layer-A diagnosis, no benchmark)

| Key | Type | Description |
|-----|------|-------------|
| `model_id` | `str` | Echo of `model_id_or_path`. |
| `arch` | `dict \| None` | F8 output if `arch_config` given. |
| `T_tau` | `dict \| None` | F6 output if `training_config` given. |
| `per_layer_fits` | `dict[str, list[dict]]` | `{kind: [{block_idx, k, lambda, R2, KS, ok}]}`. |
| `per_component_summary` | `dict[str, float]` | Median `k` per kind. |
| `classifications` | `dict[str, dict]` | F2 output per kind. |
| `distfree` | `dict[str, list[dict]]` | F6_ext fallback when `R² < 0.95`. |
| `k_drift` | `dict[str, dict]` | F5 output per kind vs. the init anchor `k=1.205`. |
| `alerts` | `list[str]` | Combined warning strings from F6 and F2. |

---

## Benchmark

### `DATABASE_v9_1`

`dict[str, dict[str, Any]]` — 12 reference model entries across 7
architectural families (paper §3, §4). Keys are model identifiers (e.g.
`"olmo-7b-hf"`, `"pythia-410m-step143000"`, `"qwen2.5-14b"`).

Each value carries:

- `arch`: `"MHA"` / `"GQA"`
- `n_q`, `n_kv`, `ratio`
- `QK_Norm`: bool
- `training_tokens_T`: float (optional)
- `trajectory`: bool — whether per-step trajectory data is available
- `expected_q_med_k` / `expected_k_med_k` / `expected_v_o_med_k` /
  `expected_qkv_med_k`: `(lo, hi)` tuples expressing the paper's
  per-component `k` band for that family.

The 12 entries map to 7 families:

| Family | Sizes |
|--------|-------|
| Pythia (MHA-merged W_qkv) | 70m, 160m, 410m, 1B, 6.9B |
| OLMo-1 (MHA-separate)     | 7B |
| OLMo-2 (MHA-separate)     | 7B |
| Llama-3 (GQA)             | 8B |
| Mistral (GQA)             | 7B |
| Qwen2.5 (GQA)             | 7B, 14B |
| Qwen3 (GQA)               | 8B |

The standalone CSV at `database_v9_1/DATABASE_v9_1.csv` carries the full
per-component median `k` / `λ` / `R²` numbers; the Python dict carries the
architectural metadata used by [`compare_to_benchmark`](#compare_to_benchmark).

---

### `compare_to_benchmark`

**Layer B — nearest-neighbour search in `DATABASE_v9_1`.**

```python
from npm_weibull import compare_to_benchmark
match = compare_to_benchmark(
    user_diagnosis,
    benchmark=None,
    family_filter=None,
    model_filter=None,
)
```

Computes a weighted L1 distance between the user's per-component median `k`
and each benchmark entry's expected `k` band (weights: Q/K get 2×, V/O 1×, to
emphasise the Selection-Class signal). Returns the closest match plus alerts
when the user's architecture or distance looks anomalous.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_diagnosis` | `dict[str, Any]` | Either a [`diagnose_model`](#diagnose_model) output, or a hand-built dict containing `per_component_summary` / `per_layer_fits` / `median_k_per_kind`, plus optionally `arch` (F8-style nested dict or a flat string). |
| `benchmark` | `dict \| None = None` | Defaults to `DATABASE_v9_1`; pass an extended dict to compare against external entries. |
| `family_filter` | `str \| None = None` | Prefix filter on benchmark entry keys (`"olmo"`, `"qwen"`, etc.). |
| `model_filter` | `str \| None = None` | Exact-match filter on a single entry key. |

**Returns** — `dict[str, Any]`

| Key | Type | Description |
|-----|------|-------------|
| `nearest_neighbor` | `str` | Benchmark entry with the smallest weighted distance. |
| `k_distance` | `float` | Weighted L1 distance. |
| `family_class` | `str` | Descriptive label (e.g. `"GQA transmission (Llama-3/Mistral/Qwen 类, q median k≈1.14)"`). |
| `per_neighbor_distances` | `dict[str, float]` | Full distance map, sorted ascending. |
| `alerts` | `list[str]` | Warning strings when user arch ≠ neighbour arch, or when `k_distance > 0.5`. |

---

## Source pointers

| Module | Source file |
|--------|-------------|
| `npm_weibull.core.weibull`      | `npm_weibull/core/weibull.py`      |
| `npm_weibull.core.classify`     | `npm_weibull/core/classify.py`     |
| `npm_weibull.core.trajectory`   | `npm_weibull/core/trajectory.py`   |
| `npm_weibull.core.training`     | `npm_weibull/core/training.py`     |
| `npm_weibull.core.distfree`     | `npm_weibull/core/distfree.py`     |
| `npm_weibull.core.architecture` | `npm_weibull/core/architecture.py` |
| `npm_weibull.utils.histogram`   | `npm_weibull/utils/histogram.py`   |
| `npm_weibull.utils.closed_form` | `npm_weibull/utils/closed_form.py` |
| `npm_weibull.utils.ks_aic`      | `npm_weibull/utils/ks_aic.py`      |
| `npm_weibull.utils.cascade_reader` | `npm_weibull/utils/cascade_reader.py` |
| `npm_weibull.workflow.diagnose` | `npm_weibull/workflow/diagnose.py` |
| `npm_weibull.benchmark.database_v9_1` | `npm_weibull/benchmark/database_v9_1.py` |

All docstrings include a `Spec:` line citing the corresponding paper section
when the function maps onto a paper-internal F-id.

---

## See also

- Paper: [arXiv:2605.18898](https://arxiv.org/abs/2605.18898) — full theoretical
  background, the 12-entry empirical study, and Appendix B's formal F1–F8 spec.
- Examples: `examples/01_quickstart_synthetic.py`,
  `examples/02_compare_to_benchmark.py`,
  `examples/03_trajectory_decomposition.py`.
- Database: `database_v9_1/DATABASE_v9_1.md` (human-readable per-component
  `k` table).
- Tests: `tests/test_synthetic.py`, `tests/test_integration.py`,
  `tests/test_coverage_p1b.py`.

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
