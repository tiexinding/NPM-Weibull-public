"""F8 Architectural Type Classifier (independent, no weight data needed).

Classifies attention architecture from model config:
  MHA (Multi-Head Attention): n_q == n_kv (each Q has its own K/V)
  GQA (Grouped Query Attention): n_kv < n_q (K/V shared across groups of Q heads)
  MQA (Multi-Query Attention): n_kv == 1 (single K/V head shared)

Spec: B2_Framework_实施Spec_v2 §1 F8 + §3.2 architecture-aware thresholds.
"""

from __future__ import annotations

# Expected post-train q/k median k drift range per architecture
# (cap stone finding 5-8 23:10: 7 model 100% MHA/GQA dichotomy verified)
_DRIFT_RANGE = {
    "MHA": "strong (median k 0.76-0.99, R²<0.95 in shallow layers)",
    "GQA": "minimal (median k 1.13-1.16, R² > 0.99 across layers)",
    "MQA": "minimal predicted (no paper#1 data; paper#2 verify)",
}


def classify_attention_arch(n_q: int, n_kv: int) -> dict:
    """Classify attention architecture from head counts.

    Parameters
    ----------
    n_q : int
        num_attention_heads (HF AutoConfig)
    n_kv : int
        num_key_value_heads (HF AutoConfig); for older transformers without GQA, equals n_q

    Returns
    -------
    dict with:
        arch : str — 'MHA' / 'GQA' / 'MQA'
        ratio : int — n_q / n_kv (1 for MHA)
        n_q : int
        n_kv : int
        expected_q_k_drift : str — descriptive expectation of k drift
        cap_stone_alignment : bool — does this arch type cleanly fit cascade v3+v2 cap stone?
    """
    if n_q <= 0 or n_kv <= 0:
        raise ValueError(f"n_q and n_kv must be positive, got n_q={n_q}, n_kv={n_kv}")
    if n_q % n_kv != 0:
        raise ValueError(
            f"n_q ({n_q}) must be divisible by n_kv ({n_kv}); non-standard attention configuration"
        )

    if n_kv == 1:
        arch = "MQA"
    elif n_kv == n_q:
        arch = "MHA"
    else:
        arch = "GQA"

    ratio = n_q // n_kv

    return {
        "arch": arch,
        "ratio": ratio,
        "n_q": n_q,
        "n_kv": n_kv,
        "expected_q_k_drift": _DRIFT_RANGE[arch],
        "cap_stone_alignment": arch in ("MHA", "GQA"),  # MQA has no paper#1 data
    }
