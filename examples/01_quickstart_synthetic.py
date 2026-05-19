"""Example 1 — Quickstart: fit Weibull to synthetic Half-Normal data.

No external model required. Demonstrates the F1 Weibull fit API on synthetic
|w| ~ Half-Normal(sigma=0.02) data, which is the initialization anchor used
throughout the paper (k_0 ~ 1.2054 under mid-80% trim).

Run:
    python examples/01_quickstart_synthetic.py
"""
from __future__ import annotations

import numpy as np

from npm_weibull import weibull_fit


def main():
    rng = np.random.default_rng(0)
    sigma_init = 0.02
    n = 200_000

    # Half-Normal: |w| where w ~ Normal(0, sigma)
    samples = np.abs(rng.normal(0.0, sigma_init, size=n))

    # Histogram in log10|w| space (cascade v3 convention: 1024 bins, [-6, 1])
    edges = np.linspace(-6.0, 1.0, 1025)
    hist, _ = np.histogram(np.log10(samples + 1e-12), bins=edges)

    fit = weibull_fit({"edges": edges, "hist": hist.astype(float)}, trim="mid_80")

    print("=== F1 Weibull fit (mid-80% trim) ===")
    print(f"  k        = {fit['k']:.4f}    (expected ~1.205 for Half-Normal init)")
    print(f"  lambda   = {fit['lambda']:.4f}")
    print(f"  R^2      = {fit['R2']:.4f}")
    print(f"  KS stat  = {fit['KS']:.4f}")
    print(f"  bins used= {fit['n_used']}")

    # Paper Appendix A.1 anchor: for Half-Normal init,  lambda ~ 0.8875 * sigma_init
    ratio = fit["lambda"] / sigma_init
    print(f"\n  lambda / sigma_init = {ratio:.4f}    (paper Appendix A.1: ~0.8875)")


if __name__ == "__main__":
    main()
