"""Weibull closed-form expressions (math primitives).

All paper §A.5 衍生 6 量表 + σ=λC_k computed from these primitives.

Spec: B2_Framework_实施Spec_v2 §5 API utility 4-5.
"""
from __future__ import annotations
import math
from scipy.special import gamma as gamma_func


def sigma_from_k_lambda(k: float, lam: float) -> dict:
    """Closed-form σ from Weibull (k, λ).

    σ = λ × √[Γ(1+2/k) - Γ²(1+1/k)]   (Weibull std dev)
    mean(|w|) = λ × Γ(1+1/k)         (Weibull mean)
    median = λ × (ln 2)^(1/k)        (Weibull median)

    Returns
    -------
    dict with σ_full (full-range theoretical) + meta closed-form values:
        sigma : float — std dev
        mean_abs_w : float — mean of |w|
        median_abs_w : float — median
        cv : float — coefficient of variation (σ / mean)
        c_k : float — std/λ shape factor (paper §A.3 b 题 σ=λ×C_k)
    """
    if k <= 0 or lam <= 0:
        raise ValueError(f"k and λ must be positive; got k={k}, λ={lam}")

    g1 = gamma_func(1.0 + 1.0 / k)
    g2 = gamma_func(1.0 + 2.0 / k)
    var_normalized = g2 - g1 * g1
    if var_normalized < 0:
        # Numerical edge case for small k
        var_normalized = 0.0
    c_k = math.sqrt(var_normalized)
    sigma = lam * c_k
    mean_abs = lam * g1
    median = lam * (math.log(2.0)) ** (1.0 / k)

    return {
        "sigma": float(sigma),
        "mean_abs_w": float(mean_abs),
        "median_abs_w": float(median),
        "cv": float(c_k / max(g1, 1e-30)),  # σ/mean = c_k / Γ(1+1/k)
        "c_k": float(c_k),
    }


def weibull_quantile(k: float, lam: float, q: float) -> float:
    """Closed-form Weibull quantile at probability q.

    Q(q) = λ × (-ln(1-q))^(1/k)

    Parameters
    ----------
    k, lam : float — Weibull (k, λ)
    q : float — probability in (0, 1)

    Returns
    -------
    float — quantile value (in |w| domain)
    """
    if k <= 0 or lam <= 0:
        raise ValueError(f"k and λ must be positive; got k={k}, λ={lam}")
    if not (0 < q < 1):
        raise ValueError(f"q must be in (0, 1); got q={q}")
    return float(lam * (-math.log(1.0 - q)) ** (1.0 / k))


def weibull_q90_q10(k: float) -> float:
    """Closed-form Q90/Q10 ratio (heaviness, paper §A.5 core).

    Q90/Q10 = ((-ln 0.1) / (-ln 0.9))^(1/k)  — independent of λ
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got k={k}")
    return float((-math.log(0.1) / -math.log(0.9)) ** (1.0 / k))
