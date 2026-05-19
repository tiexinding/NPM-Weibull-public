"""Example 3 — Trajectory decomposition (F3 sigma_decompose + F5 k_drift).

Demonstrates the paper's central training-dynamics finding: across the trajectory
from initialization to the terminal checkpoint, Transmission Class components
(W_o, FFN) exhibit lambda-dominated sigma growth (k stays near 1.20), while
Selection Class components (W_q, W_k) exhibit k-dominated drift (k departs).

This example uses synthetic trajectory data approximating the cohort medians
reported in paper Section 5 (Transmission anchor [1.186, 1.204]; OLMo-1 7B
Selection k around [0.76, 0.99]).

Run:
    python examples/03_trajectory_decomposition.py
"""

from __future__ import annotations

from npm_weibull import k_drift_severity, sigma_decompose


def main():
    # ---- Transmission Class: W_o on a GQA model (paper §3) ----
    # k stays in the band [1.186, 1.204], lambda grows ~3x across training
    k_traj_wo = [1.205, 1.201, 1.199, 1.198, 1.198]
    lam_traj_wo = [0.0089, 0.0150, 0.0220, 0.0270, 0.0294]

    res = sigma_decompose(k_traj_wo, lam_traj_wo)
    print("=== Transmission Class — W_o trajectory ===")
    print(f"  k_init = {res['k_init']:.4f}  k_final = {res['k_final']:.4f}")
    print(f"  k_drift          = {res['k_drift_pct']:.2f}%")
    print(f"  lambda_growth    = {res['lambda_growth_pct']:.2f}%")
    print(f"  sigma attribution: {res['sigma_attribution']}    (expected: lambda_dominated)")

    sev = k_drift_severity(k_traj_wo[0], k_traj_wo[-1])
    print(f"  k_drift severity : {sev['severity']}    (expected: invariant)\n")

    # ---- Selection Class: W_q on separately-stored MHA (OLMo-1 7B-ish) ----
    # k drops from ~1.20 (init) to ~0.81 (paper Table — separately-stored MHA deep selection)
    k_traj_wq = [1.205, 1.10, 0.97, 0.88, 0.81]
    lam_traj_wq = [0.0080, 0.0095, 0.0102, 0.0108, 0.0112]

    res = sigma_decompose(k_traj_wq, lam_traj_wq)
    print("=== Selection Class — W_q trajectory (separately-stored MHA) ===")
    print(f"  k_init = {res['k_init']:.4f}  k_final = {res['k_final']:.4f}")
    print(f"  k_drift          = {res['k_drift_pct']:.2f}%")
    print(f"  lambda_growth    = {res['lambda_growth_pct']:.2f}%")
    print(f"  sigma attribution: {res['sigma_attribution']}    (expected: k_dominated or mixed)")

    sev = k_drift_severity(k_traj_wq[0], k_traj_wq[-1])
    print(f"  k_drift severity : {sev['severity']}    (expected: strong)\n")

    # ---- Paired correlation: F156 cap stone (paper §5) ----
    # When two Transmission Class components share the same residual stream input,
    # their lambda trajectories track each other (paper §5: paired r ~ 0.99).
    lam_traj_wffn_out = [0.0091, 0.0152, 0.0223, 0.0275, 0.0298]  # synthetic paired
    res = sigma_decompose(k_traj_wo, lam_traj_wo, paired_lambda_traj=lam_traj_wffn_out)
    print("=== Paired-component correlation (W_o vs W_FFN_out) ===")
    print(
        f"  paired r (log-log lambda): {res['paired_r']:.4f}    "
        f"(expected: ~0.99 within Transmission Class)"
    )


if __name__ == "__main__":
    main()
