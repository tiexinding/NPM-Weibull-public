"""npm-weibull-py v0.4 — NPM-Weibull Diagnostic Framework Python Toolkit.

Public API (15 entries, aligned with the paper's API appendix: seven core diagnostics, historical F-numbering, F4/F7 merged):

  Core diagnostics (7):
    weibull_fit             — F1 Weibull (k, λ, R², KS) fit from histogram
    classify_k              — F2 k regime classifier (transmission / mild / strong / init)
    sigma_decompose         — F3 σ growth attribution (λ vs k) across step trajectory
    k_drift_severity        — F5 selection-signal severity from k_init → k_trained
    compute_T_tau           — F6 Wang-Aitchison τ_iter cycle ratio + physical state
    per_block_metrics       — F6_ext distribution-free heaviness (Q90/Q10, P999/P50, Gini)
    classify_attention_arch — F8 MHA / GQA / MQA classifier from head counts

  Utility (5):
    load_cascade_v3         — cascade v3 fit_per_component_v3.json reader
    extract_to_histogram    — weight tensor → 1024-bin log10|w| histogram (+ optional NPZ save)
    compare_distributions   — KS/AIC ranking among Weibull / Lognormal / Gamma
    sigma_from_k_lambda     — σ / mean / median / CV closed-form from (k, λ)
    weibull_quantile        — Q(q) closed-form from (k, λ)

  Workflow (1):
    diagnose_model          — Layer A one-shot diagnostic chaining F1+F2+F5+F6+F6_ext+F8

  Benchmark (2):
    DATABASE_v9_1           — 12 model entries × 7 architectural families (cascade v3 reference)
    compare_to_benchmark    — Layer B utility: user diagnosis → nearest-neighbor in DATABASE
"""

__version__ = "0.4.0"

from npm_weibull.benchmark.database_v9_1 import DATABASE_v9_1, compare_to_benchmark
from npm_weibull.core.architecture import classify_attention_arch
from npm_weibull.core.classify import classify_k
from npm_weibull.core.distfree import per_block_metrics
from npm_weibull.core.training import compute_T_tau
from npm_weibull.core.trajectory import k_drift_severity, sigma_decompose
from npm_weibull.core.weibull import weibull_fit
from npm_weibull.utils.cascade_reader import load_cascade_v3
from npm_weibull.utils.closed_form import sigma_from_k_lambda, weibull_quantile
from npm_weibull.utils.histogram import extract_to_histogram
from npm_weibull.utils.ks_aic import compare_distributions
from npm_weibull.workflow.diagnose import diagnose_model

__all__ = [
    "weibull_fit",
    "sigma_decompose",
    "per_block_metrics",
    "classify_attention_arch",
    "compute_T_tau",
    "classify_k",
    "k_drift_severity",
    "load_cascade_v3",
    "extract_to_histogram",
    "compare_distributions",
    "sigma_from_k_lambda",
    "weibull_quantile",
    "diagnose_model",
    "DATABASE_v9_1",
    "compare_to_benchmark",
]
