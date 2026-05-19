"""KS / AIC distribution comparison: Weibull vs Lognormal vs Gamma.

For paper §A.7 KS/AIC robustness analysis.
Spec: B2_Framework_实施Spec_v2 §5 utility 3.

A 5-10 14:30 Q-A3 反馈: 仅 3 个 2-param candidates (不加 gengamma).

Algorithm:
  1. binned MLE on 1024-bin log10|w| histogram (NOT raw samples — cascade v3 standard)
  2. AIC = 2k_param - 2*log_likelihood (penalize multi-param)
  3. KS = max |F_emp - F_fit| at all bin edges
  4. Rank by AIC ascending, report ΔAIC vs best
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

_VALID_CANDIDATES = ("weibull", "lognormal", "gamma")

HistogramInput = str | Path | dict[str, Any] | tuple[Any, Any]


def compare_distributions(
    histogram: HistogramInput,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    """KS/AIC ranking of candidate distributions on weight magnitude histogram.

    Parameters
    ----------
    histogram : str | Path | dict | tuple
        Input format same as weibull_fit (NPZ path / dict / tuple of edges + hist).
        edges are log10|w| bin boundaries; hist is count per bin.
    candidates : list[str] or None
        Default ['weibull', 'lognormal', 'gamma'] (all 2-param).
        gengamma (3-param) NOT included per A 5-10 Q-A3.

    Returns
    -------
    dict with:
        best : str — winning distribution by AIC
        aic_ranking : list[dict] — sorted by AIC, each {name, aic, delta_aic, k_param, log_lik}
        ks_per_dist : dict — {dist_name: ks_value}
        params_per_dist : dict — {dist_name: {param_name: value}}
    """
    if candidates is None:
        candidates = list(_VALID_CANDIDATES)
    for c in candidates:
        if c not in _VALID_CANDIDATES:
            raise ValueError(f"unknown candidate {c!r}; allowed: {_VALID_CANDIDATES}")

    edges_log10, hist = _load_histogram(histogram)
    edges_w = np.power(10.0, edges_log10)
    total = float(hist.sum())
    if total < 100:
        raise ValueError(f"insufficient counts ({int(total)}); need >= 100")

    # Empirical CDF at upper bin edges
    cum = np.cumsum(hist)
    F_emp = cum / total

    results: dict[str, dict[str, Any]] = {}
    for c in candidates:
        if c == "weibull":
            params, log_lik = _fit_weibull_binned_mle(edges_w, hist)
            ks = _ks_weibull(edges_w, F_emp, params)
        elif c == "lognormal":
            params, log_lik = _fit_lognormal_binned_mle(edges_w, hist)
            ks = _ks_lognormal(edges_w, F_emp, params)
        elif c == "gamma":
            params, log_lik = _fit_gamma_binned_mle(edges_w, hist)
            ks = _ks_gamma(edges_w, F_emp, params)
        else:
            raise RuntimeError(f"unhandled candidate: {c}")

        k_param = 2  # all 3 candidates are 2-param
        aic = 2 * k_param - 2 * log_lik
        results[c] = {
            "params": params,
            "log_lik": log_lik,
            "aic": aic,
            "ks": ks,
            "k_param": k_param,
        }

    # Rank by AIC
    ranking = sorted(results.items(), key=lambda kv: kv[1]["aic"])
    best = ranking[0][0]
    best_aic = ranking[0][1]["aic"]

    aic_ranking = [
        {
            "name": name,
            "aic": float(r["aic"]),
            "delta_aic": float(r["aic"] - best_aic),
            "k_param": r["k_param"],
            "log_lik": float(r["log_lik"]),
        }
        for name, r in ranking
    ]

    return {
        "best": best,
        "aic_ranking": aic_ranking,
        "ks_per_dist": {name: float(r["ks"]) for name, r in results.items()},
        "params_per_dist": {name: r["params"] for name, r in results.items()},
    }


def _load_histogram(h: HistogramInput) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(h, (str, Path)):
        d = np.load(str(h))
        return np.asarray(d["edges"], dtype=np.float64), np.asarray(d["hist"], dtype=np.float64)
    if isinstance(h, dict):
        return np.asarray(h["edges"], dtype=np.float64), np.asarray(h["hist"], dtype=np.float64)
    if isinstance(h, tuple) and len(h) == 2:
        return np.asarray(h[0], dtype=np.float64), np.asarray(h[1], dtype=np.float64)
    raise TypeError(f"histogram must be path/dict/tuple, got {type(h).__name__}")


# =============================================================================
# Weibull binned MLE
# =============================================================================


def _fit_weibull_binned_mle(
    edges_w: np.ndarray, hist: np.ndarray
) -> tuple[dict[str, float], float]:
    """Binned MLE for Weibull(k, λ) on |w| domain. Reuse weibull_fit weighted lstsq as init,
    then optimize log-likelihood."""
    # Initial guess from weighted lstsq (same as F1)
    from npm_weibull.core.weibull import _weibull_fit_core

    edges_log10 = np.log10(np.maximum(edges_w, 1e-30))
    init_fit = _weibull_fit_core(hist, edges_log10, "mid_80")
    if not init_fit["ok"]:
        raise RuntimeError(f"weibull init fit failed: {init_fit.get('reason')}")
    k_init, lam_init = init_fit["k"], init_fit["lambda"]

    def neg_log_lik(params: np.ndarray) -> float:
        k, lam = params
        if k <= 0 or lam <= 0:
            return 1e18
        cdf_lower = 1.0 - np.exp(-((edges_w[:-1] / lam) ** k))
        cdf_upper = 1.0 - np.exp(-((edges_w[1:] / lam) ** k))
        p_bin = np.maximum(cdf_upper - cdf_lower, 1e-30)
        return -float(np.sum(hist * np.log(p_bin)))

    res = optimize.minimize(
        neg_log_lik,
        x0=[k_init, lam_init],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1000},
    )
    k_opt, lam_opt = float(res.x[0]), float(res.x[1])
    log_lik = -float(res.fun)
    return {"k": k_opt, "lambda": lam_opt}, log_lik


def _ks_weibull(edges_w: np.ndarray, F_emp: np.ndarray, params: dict[str, float]) -> float:
    F_fit = 1.0 - np.exp(-((edges_w[1:] / params["lambda"]) ** params["k"]))
    return float(np.max(np.abs(F_emp - F_fit)))


# =============================================================================
# Lognormal binned MLE
# =============================================================================


def _fit_lognormal_binned_mle(
    edges_w: np.ndarray, hist: np.ndarray
) -> tuple[dict[str, float], float]:
    """Binned MLE for Lognormal(μ, σ) on |w|. Use scipy.stats.lognorm CDF."""
    # Initial guess from log|w| moments
    midpoints_log = 0.5 * (np.log(edges_w[:-1] + 1e-30) + np.log(edges_w[1:] + 1e-30))
    weights = hist / hist.sum()
    mu_init = float(np.sum(weights * midpoints_log))
    sigma_init = float(np.sqrt(np.sum(weights * (midpoints_log - mu_init) ** 2)))

    def neg_log_lik(params: np.ndarray) -> float:
        mu, sigma = params
        if sigma <= 0:
            return 1e18
        # F_lognorm(x) = Φ((ln(x) - μ) / σ)
        cdf_lower = stats.norm.cdf((np.log(np.maximum(edges_w[:-1], 1e-30)) - mu) / sigma)
        cdf_upper = stats.norm.cdf((np.log(np.maximum(edges_w[1:], 1e-30)) - mu) / sigma)
        p_bin = np.maximum(cdf_upper - cdf_lower, 1e-30)
        return -float(np.sum(hist * np.log(p_bin)))

    res = optimize.minimize(
        neg_log_lik,
        x0=[mu_init, sigma_init],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1000},
    )
    mu_opt, sigma_opt = float(res.x[0]), float(res.x[1])
    return {"mu": mu_opt, "sigma": sigma_opt}, -float(res.fun)


def _ks_lognormal(edges_w: np.ndarray, F_emp: np.ndarray, params: dict[str, float]) -> float:
    F_fit = stats.norm.cdf(
        (np.log(np.maximum(edges_w[1:], 1e-30)) - params["mu"]) / params["sigma"]
    )
    return float(np.max(np.abs(F_emp - F_fit)))


# =============================================================================
# Gamma binned MLE
# =============================================================================


def _fit_gamma_binned_mle(edges_w: np.ndarray, hist: np.ndarray) -> tuple[dict[str, float], float]:
    """Binned MLE for Gamma(α=shape, β=scale) on |w|. Use scipy.stats.gamma CDF."""
    # Initial guess from method-of-moments
    midpoints = 0.5 * (edges_w[:-1] + edges_w[1:])
    weights = hist / hist.sum()
    mean_w = float(np.sum(weights * midpoints))
    var_w = float(np.sum(weights * (midpoints - mean_w) ** 2))
    if mean_w <= 0 or var_w <= 0:
        raise ValueError(
            "histogram has non-positive moments; cannot initialise Gamma fit "
            f"(expected positive |w| domain, got mean_w={mean_w:.3g}, var_w={var_w:.3g})"
        )
    alpha_init = mean_w * mean_w / var_w
    beta_init = var_w / mean_w

    def neg_log_lik(params: np.ndarray) -> float:
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return 1e18
        cdf_lower = stats.gamma.cdf(edges_w[:-1], a=alpha, scale=beta)
        cdf_upper = stats.gamma.cdf(edges_w[1:], a=alpha, scale=beta)
        p_bin = np.maximum(cdf_upper - cdf_lower, 1e-30)
        return -float(np.sum(hist * np.log(p_bin)))

    res = optimize.minimize(
        neg_log_lik,
        x0=[alpha_init, beta_init],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 1000},
    )
    a_opt, b_opt = float(res.x[0]), float(res.x[1])
    return {"alpha": a_opt, "beta": b_opt}, -float(res.fun)


def _ks_gamma(edges_w: np.ndarray, F_emp: np.ndarray, params: dict[str, float]) -> float:
    F_fit = stats.gamma.cdf(edges_w[1:], a=params["alpha"], scale=params["beta"])
    return float(np.max(np.abs(F_emp - F_fit)))
