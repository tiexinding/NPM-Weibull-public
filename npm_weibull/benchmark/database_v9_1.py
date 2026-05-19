"""DATABASE_v9_1 — paper reference (12 model entries across 7 architectural families).

Source: cascade v3 per-component Weibull fit (mid-80% trim).
Companion to paper §3-§5 (see DATABASE_v9_1.csv for the full table).
"""

from __future__ import annotations

# 12 model × architectural metadata (cascade v3 reference)
DATABASE_v9_1 = {
    # MHA family (separate Q/K/V/O storage, no QK-Norm)
    "olmo-7b-hf": {
        "arch": "MHA",
        "n_q": 32,
        "n_kv": 32,
        "ratio": 1,
        "QK_Norm": False,
        "trajectory": False,
        "expected_q_med_k": (0.76, 0.85),  # cascade v3 verify: 0.81
        "expected_k_med_k": (0.70, 0.80),  # 0.76
        "expected_v_o_med_k": (1.04, 1.06),
        "training_tokens_T": 2.5,
    },
    "olmo2-7b-final": {
        "arch": "MHA",
        "n_q": 32,
        "n_kv": 32,
        "ratio": 1,
        "QK_Norm": True,
        "trajectory": False,
        "expected_q_med_k": (0.95, 1.05),  # 0.99 (QK-Norm mitigator)
        "expected_k_med_k": (0.92, 1.00),  # 0.97
        "expected_v_o_med_k": (1.18, 1.20),
        "training_tokens_T": 4.0,
    },
    # Pythia family (MHA, W_qkv merged tensor — V dilute)
    "pythia-70m-step143000": {
        "arch": "MHA",
        "n_q": 8,
        "n_kv": 8,
        "ratio": 1,
        "QK_Norm": False,
        "trajectory": True,
        "merged_qkv": True,
        "T_tau_iter": 1.43,
        "physical_state": "saturated",
        "expected_qkv_med_k": (1.04, 1.10),  # 1.05
    },
    "pythia-160m-step143000": {
        "arch": "MHA",
        "n_q": 12,
        "n_kv": 12,
        "merged_qkv": True,
        "T_tau_iter": 1.0,
        "physical_state": "saturated",
        "expected_qkv_med_k": (1.08, 1.12),
    },
    "pythia-410m-step143000": {
        "arch": "MHA",
        "n_q": 16,
        "n_kv": 16,
        "merged_qkv": True,
        "T_tau_iter": 0.7,
        "physical_state": "approaching",
        "expected_qkv_med_k": (1.13, 1.17),
    },
    "pythia-1b-step143000": {
        "arch": "MHA",
        "n_q": 8,
        "n_kv": 8,
        "merged_qkv": True,
        "T_tau_iter": 0.43,
        "physical_state": "approaching",
        "expected_qkv_med_k": (1.15, 1.19),
    },
    "pythia-6.9b-step143000": {
        "arch": "MHA",
        "n_q": 32,
        "n_kv": 32,
        "merged_qkv": True,
        "T_tau_iter": 0.17,
        "physical_state": "transition",
        "expected_qkv_med_k": (1.15, 1.19),
        "warning": "T/τ=0.17, NOT physical terminal — apparent stability is budget artifact",
    },
    # GQA family (separate Q/K/V/O, K shared across Q heads)
    "llama-3-8b": {
        "arch": "GQA",
        "n_q": 32,
        "n_kv": 8,
        "ratio": 4,
        "QK_Norm": False,
        "trajectory": False,
        "expected_q_med_k": (1.13, 1.16),  # 1.14
        "expected_k_med_k": (1.13, 1.17),  # 1.15
        "expected_v_o_med_k": (1.17, 1.19),
    },
    "mistral-7b": {
        "arch": "GQA",
        "n_q": 32,
        "n_kv": 8,
        "ratio": 4,
        "QK_Norm": False,
        "trajectory": False,
        "expected_q_med_k": (1.14, 1.16),
        "expected_k_med_k": (1.12, 1.14),
        "expected_v_o_med_k": (1.17, 1.19),
    },
    "qwen2.5-7b": {
        "arch": "GQA",
        "n_q": 28,
        "n_kv": 4,
        "ratio": 7,
        "QK_Norm": False,
        "trajectory": False,
        "expected_q_med_k": (1.13, 1.16),
        "expected_k_med_k": (1.10, 1.14),
        "expected_v_o_med_k": (1.16, 1.20),
    },
    "qwen2.5-14b": {
        "arch": "GQA",
        "n_q": 40,
        "n_kv": 8,
        "ratio": 5,
        "QK_Norm": False,
        "trajectory": False,
        "expected_q_med_k": (1.15, 1.17),
        "expected_k_med_k": (1.12, 1.15),
        "expected_v_o_med_k": (1.16, 1.19),
    },
    "qwen3-8b-base": {
        "arch": "GQA",
        "n_q": 32,
        "n_kv": 8,
        "ratio": 4,
        "QK_Norm": True,
        "trajectory": False,
        "expected_q_med_k": (1.15, 1.18),
        "expected_k_med_k": (1.14, 1.17),
        "expected_v_o_med_k": (1.16, 1.19),
        "note": "GQA + QK-Norm: redundant, behaves same as Qwen2.5 GQA without QK-Norm",
    },
}


def compare_to_benchmark(
    user_diagnosis: dict,
    benchmark: dict | None = None,
    family_filter: str | None = None,
    model_filter: str | None = None,
) -> dict:
    """Compare user model diagnosis to benchmark database, find nearest neighbor.

    **Layer B utility — independent from diagnose_model (Layer A).**
    Decoupling: user calls diagnose_model() first → raw diagnosis, then explicitly
    calls this function for benchmark comparison. Optional, not chained internally.

    Parameters
    ----------
    user_diagnosis : dict
        Output of diagnose_model() OR manually constructed dict with keys:
        'arch' (from F8), 'per_layer_fits' or 'per_component_summary' (median k per kind)
    benchmark : dict | None
        Default DATABASE_v9_1; can pass extended/external benchmark dict
    family_filter : str | None
        Filter benchmark by family prefix (e.g., 'olmo', 'qwen', 'pythia')
    model_filter : str | None
        Filter benchmark to single entry by model_id key

    Returns
    -------
    dict with:
        nearest_neighbor : str — model id with smallest k_distance
        k_distance : float — weighted L1 distance over q/k/v/o medians
        family_class : str — descriptive class label
        per_neighbor_distances : dict — full distance map
        alerts : list[str]
    """
    if benchmark is None:
        benchmark = DATABASE_v9_1

    # Filter benchmark
    bench_filtered = {}
    for model_id, meta in benchmark.items():
        if model_filter is not None and model_id != model_filter:
            continue
        if family_filter is not None and not model_id.lower().startswith(family_filter.lower()):
            continue
        bench_filtered[model_id] = meta
    if not bench_filtered:
        raise ValueError(
            f"No benchmark entries after filter "
            f"(family_filter={family_filter}, model_filter={model_filter})"
        )

    # Extract user per-component median k
    user_med_k = _extract_user_median_k(user_diagnosis)

    # Compute weighted L1 distance: q/k weight 2x (selection-class signal), v/o weight 1x
    weights_per_kind = {"q": 2.0, "k": 2.0, "v": 1.0, "o": 1.0, "qkv": 2.0}
    distances = {}
    for model_id, meta in bench_filtered.items():
        d = _benchmark_distance(user_med_k, meta, weights_per_kind)
        if d is not None:
            distances[model_id] = d

    if not distances:
        raise ValueError(
            "Could not compute distance to any benchmark entry (check user_diagnosis schema)"
        )

    nearest = min(distances.items(), key=lambda kv: kv[1])
    nearest_id, k_distance = nearest

    # Family classification
    user_arch = user_diagnosis.get("arch", {}).get("arch", "?")
    nearest_meta = bench_filtered[nearest_id]
    nearest_arch = nearest_meta.get("arch", "?")
    family_class = _classify_family(user_arch, user_med_k, nearest_arch)

    # Alerts
    alerts = []
    if user_arch != nearest_arch:
        alerts.append(
            f"⚠️ user arch={user_arch} but nearest neighbor arch={nearest_arch} — "
            f"potential mismatch, verify F8 classification"
        )
    if k_distance > 0.5:
        alerts.append(
            f"⚠️ k_distance={k_distance:.3f} > 0.5 — user model far from any benchmark entry, "
            f"consider extending benchmark via paper#2 cascade v5"
        )

    return {
        "nearest_neighbor": nearest_id,
        "k_distance": float(k_distance),
        "family_class": family_class,
        "per_neighbor_distances": {
            k: float(v) for k, v in sorted(distances.items(), key=lambda kv: kv[1])
        },
        "alerts": alerts,
    }


def _extract_user_median_k(diagnosis: dict) -> dict:
    """Extract per-kind median k from diagnose_model() output OR manual dict."""
    # Format A: dict with per-kind median already
    if "per_component_summary" in diagnosis:
        return diagnosis["per_component_summary"]
    # Format B: from per_layer_fits
    if "per_layer_fits" in diagnosis:
        out = {}
        import statistics

        for kind, fits in diagnosis["per_layer_fits"].items():
            ks = [f["k"] for f in fits if f.get("ok") and f.get("k")]
            if ks:
                out[kind] = statistics.median(ks)
        return out
    # Format C: direct user input
    if "median_k_per_kind" in diagnosis:
        return diagnosis["median_k_per_kind"]
    raise ValueError(
        "diagnosis must contain 'per_component_summary' OR 'per_layer_fits' OR 'median_k_per_kind'"
    )


def _benchmark_distance(user_med_k: dict, bench_meta: dict, weights: dict) -> float | None:
    """Weighted L1 distance over per-kind median k."""
    total_w = 0.0
    total_d = 0.0
    for kind, user_k in user_med_k.items():
        w = weights.get(kind, 0.5)
        # Try multiple key naming conventions in bench_meta
        bench_range = (
            bench_meta.get(f"expected_{kind}_med_k")
            or bench_meta.get(f"expected_{kind}k_med_k")  # qk
            or (bench_meta.get(f"expected_{kind.upper()}_med_k") if kind in ("q", "k") else None)
            or (bench_meta.get("expected_v_o_med_k") if kind in ("v", "o") else None)
            or (bench_meta.get("expected_qkv_med_k") if kind == "qkv" else None)
        )
        if bench_range is None:
            continue  # this kind not specified for this benchmark entry
        # bench_range is (lo, hi) tuple; compute distance to nearest edge
        lo, hi = bench_range
        if lo <= user_k <= hi:
            d = 0.0
        else:
            d = min(abs(user_k - lo), abs(user_k - hi))
        total_d += w * d
        total_w += w
    if total_w == 0:
        return None
    return total_d / total_w


def _classify_family(user_arch: str, user_med_k: dict, nearest_arch: str) -> str:
    """Descriptive family class label."""
    q_k = user_med_k.get("q", user_med_k.get("qkv"))
    if q_k is None:
        return f"{user_arch} (q/k median unknown)"

    if user_arch == "MHA":
        if q_k < 0.95:
            return "MHA strong selection (OLMo-1 类, no QK-Norm)"
        elif q_k < 1.10:
            return "MHA mild selection (OLMo-2 类, with QK-Norm)"
        else:
            return "MHA near-transmission (atypical, verify)"
    elif user_arch == "GQA":
        if q_k > 1.10:
            return f"GQA transmission (Llama-3/Mistral/Qwen 类, q median k≈{q_k:.2f})"
        else:
            return "GQA selection (atypical, verify K-sharing)"
    elif user_arch == "MQA":
        return f"MQA (no paper#1 reference, predicted transmission, q k≈{q_k:.2f})"
    return f"{user_arch} (q median k≈{q_k:.2f})"
