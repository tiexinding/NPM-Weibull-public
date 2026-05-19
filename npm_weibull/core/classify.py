"""F2 Shape Parameter Readout (k 类别) — dedicated helper.

A v1 Q1 反馈 (5-9 早) + B2 Q1 答复采纳: F2 加 dedicated function (跟 paper Table 1 8 项 API 一一对应).

Spec: B2_Framework_实施Spec_v2 §1 F2 + §3.1 thresholds + A 评审 Catch A1 (5-10 早 09:00 sign-off).

Thresholds (paper §A.7/A.8.3 一致, A 5-10 早 09:30 确认):
  transmission band: k ∈ [1.18, 1.21] AND R² ≥ 0.95  → Llama-3/Mistral/Qwen2.5/3 (GQA)
  mild selection:    0.95 ≤ k < 1.18                 → OLMo-2 (MHA + QK-Norm)
  strong selection:  k < 0.95                        → OLMo-1 (MHA, no QK-Norm)
  init baseline:     |k - 1.205| < 0.01 AND step=0   → all init Half-Normal
"""

from __future__ import annotations

from typing import Any

_THRESHOLDS = {
    "transmission_low": 1.18,
    "transmission_high": 1.21,
    "mild_selection_low": 0.95,
    "init_anchor": 1.205,
    "init_tol": 0.01,
}


def classify_k(
    k: float,
    R2: float | None = None,
    arch_type: str | None = None,
    is_init: bool = False,
) -> dict[str, Any]:
    """F2 Shape Parameter Readout — classify Weibull k into discrete regime.

    Parameters
    ----------
    k : float
        Weibull shape parameter
    R2 : float | None, default None
        Optional R² goodness of fit. If given, transmission requires R² ≥ 0.95.
    arch_type : str | None, default None
        Optional 'MHA' / 'GQA' / 'MQA' from F8 classify_attention_arch
        Used for in_expected_range check
    is_init : bool, default False
        If True (step=0 ckpt), check against init Half-Normal anchor k=1.205

    Returns
    -------
    dict with:
        class : str — 'transmission' / 'mild_selection' / 'strong_selection' / 'init_baseline'
        in_expected_range : bool | None — if arch_type given, MHA expects 0.76-0.99 q/k,
                                          GQA expects 1.13-1.16, etc.
        deviation_from_init : float — k - 1.205 (init Half-Normal anchor)
        deviation_pct : float — (k - 1.205) / 1.205 * 100
    """
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")

    deviation = k - _THRESHOLDS["init_anchor"]
    deviation_pct = deviation / _THRESHOLDS["init_anchor"] * 100.0

    # Init baseline check (step=0 only)
    if is_init and abs(deviation) < _THRESHOLDS["init_tol"]:
        cls = "init_baseline"
    elif k >= _THRESHOLDS["transmission_low"] and k <= _THRESHOLDS["transmission_high"]:
        if R2 is None or R2 >= 0.95:
            cls = "transmission"
        else:
            cls = "mild_selection"  # k 在 transmission 范围但 R²<0.95, 降级
    elif k >= _THRESHOLDS["mild_selection_low"]:
        cls = "mild_selection"
    else:
        cls = "strong_selection"

    # arch-aware expected range check
    in_expected = None
    if arch_type is not None:
        if arch_type == "MHA":
            # OLMo lineage: q/k median k 0.76-0.99 (selection)
            in_expected = cls in ("strong_selection", "mild_selection")
        elif arch_type == "GQA":
            # Llama/Mistral/Qwen: q/k median k 1.13-1.16 (transmission)
            in_expected = cls == "transmission"
        elif arch_type == "MQA":
            # No paper#1 data, predicted minimal drift like GQA
            in_expected = cls == "transmission"

    return {
        "class": cls,
        "in_expected_range": in_expected,
        "deviation_from_init": float(deviation),
        "deviation_pct": float(deviation_pct),
    }
