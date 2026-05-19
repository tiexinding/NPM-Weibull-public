"""F6_extension Distribution-free metrics (robust fallback when R²<0.95).

Distribution-free quantile / Gini computation directly from histogram, no fit needed.
Used when:
  - F1 weibull_fit R² < 0.95 (mixture distribution, Q/K specialization)
  - Need quantile-based heaviness metric (Q90/Q10 跨 family 比较)

Spec: B2_Framework_实施Spec_v2 §1 F6_extension + §5 API 3.
Source: F148 + F150 + F153 实测 metric, paper §A.8 used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

HistogramInput = str | Path | dict[str, Any] | tuple[Any, Any]


def per_block_metrics(histogram: HistogramInput) -> dict[str, Any]:
    """Compute distribution-free heaviness metrics from histogram.

    Parameters
    ----------
    histogram : str | Path | dict | tuple
        Same input format as weibull_fit; edges are log10|w|, hist is count per bin.

    Returns
    -------
    dict with:
        q90_q10 : float — heaviness ratio (Weibull-friendly, paper §A.8)
        q99_q50 : float — severe tail ratio
        p999_p50 : float — extreme outlier ratio (7× more sensitive than q90_q10, F150)
        gini : float — concentration coefficient [0, 1]
        median_abs_w : float — |w| median (linear scale)
    """
    edges, hist = _load_histogram(histogram)
    return _per_block_metrics_core(hist, edges)


def _load_histogram(h: HistogramInput) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(h, (str, Path)):
        d = np.load(str(h))
        return np.asarray(d["edges"], dtype=np.float64), np.asarray(d["hist"], dtype=np.float64)
    if isinstance(h, dict):
        return np.asarray(h["edges"], dtype=np.float64), np.asarray(h["hist"], dtype=np.float64)
    if isinstance(h, tuple) and len(h) == 2:
        return np.asarray(h[0], dtype=np.float64), np.asarray(h[1], dtype=np.float64)
    raise TypeError(f"histogram must be path/dict/tuple, got {type(h).__name__}")


def _per_block_metrics_core(hist: np.ndarray, edges_log10: np.ndarray) -> dict[str, Any]:
    """Core metric computation. edges are log10|w|; convert to linear |w| domain."""
    if hist.sum() < 100:
        return {
            "q90_q10": float("nan"),
            "q99_q50": float("nan"),
            "p999_p50": float("nan"),
            "gini": float("nan"),
            "median_abs_w": float("nan"),
            "ok": False,
            "reason": "insufficient counts",
        }

    edges_w = np.power(10.0, edges_log10)
    midpoints = 0.5 * (edges_w[:-1] + edges_w[1:])

    q10 = _quantile(midpoints, hist, 0.10)
    q50 = _quantile(midpoints, hist, 0.50)
    q90 = _quantile(midpoints, hist, 0.90)
    q99 = _quantile(midpoints, hist, 0.99)
    q999 = _quantile(midpoints, hist, 0.999)

    # Gini coefficient via Lorenz curve
    total = hist.sum()
    cum_w = np.cumsum(midpoints * hist) / np.sum(midpoints * hist)
    cum_n = np.cumsum(hist) / total
    gini = 1.0 - 2.0 * np.trapezoid(cum_w, cum_n)

    return {
        "q90_q10": float(q90 / max(q10, 1e-30)),
        "q99_q50": float(q99 / max(q50, 1e-30)),
        "p999_p50": float(q999 / max(q50, 1e-30)),
        "gini": float(gini),
        "median_abs_w": float(q50),
        "ok": True,
    }


def _quantile(midpoints: np.ndarray, hist: np.ndarray, q: float) -> float:
    """Cumulative-count quantile interpolation (no fit assumption)."""
    cum = np.cumsum(hist)
    target = q * cum[-1]
    idx = int(np.searchsorted(cum, target))
    return float(midpoints[min(idx, len(midpoints) - 1)])
