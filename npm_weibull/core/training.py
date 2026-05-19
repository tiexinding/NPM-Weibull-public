"""F6 Wang-Aitchison τ_iter cycle ratio (training state diagnostic).

τ_iter = 1 / (η · λ_wd) — number of training steps for AdamW EMA to fully decay
T/τ_iter — cycle ratio, indicates training saturation:
  T/τ ≥ 1.0  : saturated (≥1 EMA cycle)
  0.2 ≤ T/τ < 1.0 : approaching saturation
  T/τ < 0.2  : transition state (NOT physical terminal, budget-limited)

Reference: Wang & Aitchison 2024 (arXiv:2405.13698)

Spec: B2_Framework_实施Spec_v2 §1 F6 + §3.1 thresholds.
Critical (老丁 5-8 catch): T/τ < 0.2 = NOT terminal even if step trajectory ends here.
"""

from __future__ import annotations

_T_TAU_THRESHOLDS = {
    "saturated": 1.0,
    "approaching": 0.2,
}


def compute_T_tau(eta: float, lambda_wd: float, T_steps: int) -> dict:
    """Compute T/τ_iter cycle ratio + state classification.

    Parameters
    ----------
    eta : float
        learning rate (peak or final, typical 1e-4 to 1e-3 for transformer LM)
    lambda_wd : float
        weight decay coefficient (typical 0.01 to 0.1)
    T_steps : int
        total training steps

    Returns
    -------
    dict with:
        tau_iter : float — 1/(η·λ_wd) [steps]
        T_tau : float — T_steps / tau_iter
        state : str — 'saturated' / 'approaching' / 'transition'
        physical_terminal : bool — True if state == 'saturated'
        warning : str | None — caveat if transition state
    """
    if eta <= 0 or lambda_wd <= 0 or T_steps <= 0:
        raise ValueError(
            f"eta, lambda_wd, T_steps must be positive; "
            f"got eta={eta}, lambda_wd={lambda_wd}, T_steps={T_steps}"
        )

    tau_iter = 1.0 / (eta * lambda_wd)
    T_tau = T_steps / tau_iter

    if T_tau >= _T_TAU_THRESHOLDS["saturated"]:
        state = "saturated"
        warning = None
    elif T_tau >= _T_TAU_THRESHOLDS["approaching"]:
        state = "approaching"
        warning = None
    else:
        state = "transition"
        warning = (
            f"T/τ_iter={T_tau:.2f} < 0.2: transition state, NOT physical terminal. "
            f"Apparent stability may be budget artifact (e.g., Pythia-6.9B at step143K). "
            f"Continued training would likely produce more departure."
        )

    return {
        "tau_iter": float(tau_iter),
        "T_tau": float(T_tau),
        "state": state,
        "physical_terminal": state == "saturated",
        "warning": warning,
    }
