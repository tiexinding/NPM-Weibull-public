"""npm-weibull-py v0.4 — NPM-Weibull Diagnostic Framework Python Toolkit.

Public API (跟 B2_Framework_实施Spec_v2 §5 严格对应):
  Core (5):
    weibull_fit       — F1 Weibull fit (k, λ, R², KS) from histogram
    sigma_decompose   — F3 σ growth attribution (λ vs k) across step trajectory
    per_block_metrics — F6 distribution-free metrics (Q90/Q10, P99/P50, P99.9/P50, Gini)
    classify_attention_arch — F8 MHA/GQA/MQA classifier from model config
    compute_T_tau     — F6 Wang-Aitchison τ_iter cycle ratio

  Utility (5):
    load_cascade_v3   — cascade v3+v2 fit_per_component_v3.json reader
    extract_to_histogram  — weight tensor → 1024-bin log10|w| histogram
    compare_distributions — KS/AIC ranking (Weibull / Lognormal / Gamma)
    sigma_from_k_lambda   — σ closed-form
    weibull_quantile      — quantile closed-form

  Workflow (1):
    diagnose_model    — one-shot diagnostic (combines all 8 functions)

  Benchmark:
    DATABASE_v9_1     — 12 model entries across 7 families (cascade v3 reference)
    compare_to_benchmark  — user diagnosis vs benchmark closest neighbor
"""

__version__ = "0.4.0"

# Core API
from npm_weibull.core.weibull import weibull_fit
from npm_weibull.core.architecture import classify_attention_arch
from npm_weibull.core.training import compute_T_tau
from npm_weibull.core.distfree import per_block_metrics
from npm_weibull.core.trajectory import sigma_decompose, k_drift_severity
from npm_weibull.core.classify import classify_k

# Utility API
from npm_weibull.utils.closed_form import sigma_from_k_lambda, weibull_quantile
from npm_weibull.utils.cascade_reader import load_cascade_v3
from npm_weibull.utils.histogram import extract_to_histogram
from npm_weibull.utils.ks_aic import compare_distributions

# Workflow
from npm_weibull.workflow.diagnose import diagnose_model

# Benchmark
from npm_weibull.benchmark.database_v9_1 import DATABASE_v9_1, compare_to_benchmark

__all__ = [
    "weibull_fit", "sigma_decompose", "per_block_metrics",
    "classify_attention_arch", "compute_T_tau", "classify_k", "k_drift_severity",
    "load_cascade_v3", "extract_to_histogram", "compare_distributions",
    "sigma_from_k_lambda", "weibull_quantile",
    "diagnose_model",
    "DATABASE_v9_1", "compare_to_benchmark",
]
