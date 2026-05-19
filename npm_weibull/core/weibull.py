"""F1 Weibull Fit (核心 base API, 全部 downstream 依赖).

Algorithm: weighted lstsq on log-log Weibull plot
  log(-ln(1-F)) = k·log(x) - k·log(λ)
  Where F is empirical CDF, x is bin midpoint (log scale).

Spec: B2_Framework_实施Spec_v2 §1 F1 + §5 API 1.
Source: cascade v3 weibull_v3.py weibull_fit_from_hist (5-7 起 verified).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_VALID_TRIM = {"mid_80", "mid_90", "full_range"}


def weibull_fit(
    histogram: str | Path | dict | tuple,
    trim: str = "mid_80",
) -> dict:
    """Fit Weibull(k, λ) to weight magnitude histogram via weighted lstsq.

    Parameters
    ----------
    histogram : str | Path | dict | tuple
        - str/Path: NPZ file path with keys 'edges' (1025,) and 'hist' (1024,)
        - dict: {'edges': array, 'hist': array}
        - tuple: (edges, hist)
        edges: log10|w| bin boundaries (length n+1)
        hist: count per bin (length n)
    trim : str, default 'mid_80'
        Body fit trim — 'mid_80' (paper §A.7 standard, drop tails 10%/10%),
        'mid_90' (drop 5%/5%), or 'full_range'

    Returns
    -------
    dict with keys:
        k : float        — Weibull shape parameter (k=1.205 for Half-Normal init)
        lambda : float   — Weibull scale parameter
        R2 : float       — goodness of fit on weighted log-log plot, [0, 1]
        KS : float       — Kolmogorov-Smirnov stat (deviation between empirical/fitted CDF)
        n_used : int     — number of bins used in fit
        trim : str       — echo input trim
        ok : bool        — fit converged AND k>0 AND R2 finite
    """
    if trim not in _VALID_TRIM:
        raise ValueError(f"trim must be one of {_VALID_TRIM}, got {trim!r}")

    edges, hist = _load_histogram(histogram)
    return _weibull_fit_core(hist, edges, trim)


def _load_histogram(h):
    """Normalize input to (edges, hist) numpy arrays."""
    if isinstance(h, (str, Path)):
        d = np.load(str(h))
        return np.asarray(d["edges"], dtype=np.float64), np.asarray(d["hist"], dtype=np.float64)
    if isinstance(h, dict):
        return np.asarray(h["edges"], dtype=np.float64), np.asarray(h["hist"], dtype=np.float64)
    if isinstance(h, tuple) and len(h) == 2:
        return np.asarray(h[0], dtype=np.float64), np.asarray(h[1], dtype=np.float64)
    raise TypeError(f"histogram must be path/dict/tuple, got {type(h).__name__}")


def _weibull_fit_core(hist: np.ndarray, edges: np.ndarray, trim: str) -> dict:
    """Core weighted lstsq fit. Returns full result dict including KS."""
    total = float(hist.sum())
    if total < 100:
        return _nan_result(trim, "insufficient counts (<100)")

    # Empirical CDF at bin midpoints
    cdf_upper = np.cumsum(hist) / total
    cdf_lower = (np.cumsum(hist) - hist) / total
    F_mid = (cdf_upper + cdf_lower) / 2.0

    # bin centers in log scale (edges are log10|w| -> log scale by × ln(10))
    bc_log10 = (edges[:-1] + edges[1:]) / 2.0
    bc_log = bc_log10 * np.log(10.0)

    # Trim mask
    trim_pct = {"mid_80": 80, "mid_90": 90, "full_range": 100}[trim]
    if trim_pct >= 100:
        mask = (hist > 0) & (F_mid > 0) & (F_mid < 1)
    else:
        margin = (1 - trim_pct / 100.0) / 2.0
        mask = (F_mid >= margin) & (F_mid <= 1 - margin) & (hist > 0)

    if mask.sum() < 5:
        return _nan_result(trim, "insufficient bins after trim (<5)")

    x = bc_log[mask]
    y = np.log(-np.log(1 - F_mid[mask] + 1e-12))
    w = hist[mask].astype(np.float64)

    A = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(np.sqrt(w))
    Aw = W @ A
    yw = W @ y

    try:
        sol, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        k = float(sol[0])
        lam = float(np.exp(-sol[1] / k))
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as exc:
        return _nan_result(trim, f"lstsq failure: {exc}")

    if not (np.isfinite(k) and np.isfinite(lam) and k > 0 and lam > 0):
        return _nan_result(trim, "non-physical k/λ from fit")

    # R² on weighted log-log
    y_pred = A @ sol
    ss_res = float((w * (y - y_pred) ** 2).sum())
    y_w_mean = float((w * y).sum() / w.sum())
    ss_tot = float((w * (y - y_w_mean) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # KS statistic (max |F_emp - F_fit|)
    bc_w = 10.0**bc_log10  # convert back to |w| domain
    F_fit = 1.0 - np.exp(-((bc_w / lam) ** k))
    ks = float(np.abs(cdf_upper - F_fit).max())

    return {
        "k": k,
        "lambda": lam,
        "R2": float(r2),
        "KS": ks,
        "n_used": int(mask.sum()),
        "trim": trim,
        "ok": True,
    }


def _nan_result(trim: str, reason: str) -> dict:
    return {
        "k": float("nan"),
        "lambda": float("nan"),
        "R2": float("nan"),
        "KS": float("nan"),
        "n_used": 0,
        "trim": trim,
        "ok": False,
        "reason": reason,
    }
