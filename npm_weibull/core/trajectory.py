"""F3 σ growth attribution + F5 k drift severity (trajectory across step).

Decomposes σ growth into λ-dominated vs k-dominated contributions:
  Weibull σ = λ × C_k where C_k = √[Γ(1+2/k) - Γ²(1+1/k)]
  if k_drift << 1% (transmission): σ growth = λ growth (λ-dominated)
  if k_drift > 30% (selection): σ growth = mixed (λ + C_k both change)

Spec: B2_Framework_实施Spec_v2 §1 F3+F5 + §5 API 2.
Source: F156 paper §A.8.2 verify (paired r=0.9967 across saturated state).
"""
from __future__ import annotations
import numpy as np
from npm_weibull.utils.closed_form import sigma_from_k_lambda


def sigma_decompose(
    k_traj,                                 # list[float] | np.ndarray
    lambda_traj,                            # list[float] | np.ndarray
    paired_lambda_traj=None,                # list[float] | None — for paired correlation r
) -> dict:
    """Decompose σ growth across step trajectory: λ-dominated vs k-dominated vs mixed.

    Parameters
    ----------
    k_traj : array-like
        Weibull k values across training steps (e.g., k_O at step0, step1, ..., step143K)
    lambda_traj : array-like
        Weibull λ values, same length
    paired_lambda_traj : array-like or None
        Optional paired λ trajectory of another component (e.g., λ_FFN_out)
        for paired correlation analysis (F156 cap stone: r=0.9967 saturated state)

    Returns
    -------
    dict with:
        k_drift_pct : float — (k_final - k_init) / k_init × 100
        lambda_growth_pct : float — (λ_final - λ_init) / λ_init × 100
        sigma_attribution : str — 'lambda_dominated' / 'mixed' / 'k_dominated'
        paired_r : float | None — Pearson r on log-log (if paired_lambda_traj given)
        sigma_init : float — predicted σ at step 0 (from k_init, λ_init)
        sigma_final : float — predicted σ at step T
    """
    k_arr = np.asarray(k_traj, dtype=np.float64)
    lam_arr = np.asarray(lambda_traj, dtype=np.float64)
    if k_arr.size < 2 or lam_arr.size < 2 or k_arr.size != lam_arr.size:
        raise ValueError("k_traj and lambda_traj must be same length and len ≥ 2")

    k_init, k_final = float(k_arr[0]), float(k_arr[-1])
    lam_init, lam_final = float(lam_arr[0]), float(lam_arr[-1])

    k_drift_pct = (k_final - k_init) / abs(k_init) * 100.0
    lambda_growth_pct = (lam_final - lam_init) / abs(lam_init) * 100.0

    sigma_init = sigma_from_k_lambda(k_init, lam_init)["sigma"]
    sigma_final = sigma_from_k_lambda(k_final, lam_final)["sigma"]

    # Attribution thresholds: B2_Framework_实施Spec_v2 §3.1
    if abs(k_drift_pct) < 1.0:
        attribution = "lambda_dominated"
    elif abs(k_drift_pct) > 30.0:
        attribution = "k_dominated"
    else:
        attribution = "mixed"

    # Paired correlation (log-log) if available
    paired_r = None
    if paired_lambda_traj is not None:
        paired = np.asarray(paired_lambda_traj, dtype=np.float64)
        if paired.size == lam_arr.size and (lam_arr > 0).all() and (paired > 0).all():
            log_x = np.log10(lam_arr)
            log_y = np.log10(paired)
            paired_r = float(np.corrcoef(log_x, log_y)[0, 1])

    return {
        "k_init": k_init,
        "k_final": k_final,
        "k_drift_pct": float(k_drift_pct),
        "lambda_init": lam_init,
        "lambda_final": lam_final,
        "lambda_growth_pct": float(lambda_growth_pct),
        "sigma_init": float(sigma_init),
        "sigma_final": float(sigma_final),
        "sigma_attribution": attribution,
        "paired_r": paired_r,
    }


def k_drift_severity(k_init: float, k_trained: float) -> dict:
    """Classify k drift severity (F5 selection signal).

    Returns
    -------
    dict with:
        delta_k : float — k_trained - k_init
        delta_pct : float — relative drift (%)
        severity : str — 'invariant' (<1%) / 'mild' (1-30%) / 'strong' (>30%)
    """
    if k_init <= 0:
        raise ValueError(f"k_init must be positive; got {k_init}")

    delta_k = k_trained - k_init
    delta_pct = abs(delta_k) / k_init * 100.0

    if delta_pct < 1.0:
        severity = "invariant"
    elif delta_pct <= 30.0:
        severity = "mild"
    else:
        severity = "strong"

    return {
        "delta_k": float(delta_k),
        "delta_pct": float(delta_pct),
        "severity": severity,
    }
