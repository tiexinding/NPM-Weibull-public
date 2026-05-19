"""diagnose_model: Layer A only, NO benchmark dependency.

工具 universal applicable, 跟 cascade v3/v4 benchmark 完全独立.
若需 benchmark 比较, 使用 compare_to_benchmark() (Layer B utility).
任何 model X (HF / local / 未在 benchmark 内) 都能调用本 API.

(B1 5-10 06:00 加固建议 — docstring 顶部强调 Layer A NO benchmark dependency)

Spec: B2_Framework_实施Spec_v2 §5 wrapper (5-9 23:00 老丁 catch 后修正解耦).
"""
from __future__ import annotations
import statistics
from pathlib import Path

from npm_weibull.core.weibull import weibull_fit
from npm_weibull.core.architecture import classify_attention_arch
from npm_weibull.core.training import compute_T_tau
from npm_weibull.core.distfree import per_block_metrics
from npm_weibull.core.classify import classify_k
from npm_weibull.core.trajectory import k_drift_severity
from npm_weibull.utils.cascade_reader import load_cascade_v3, filter_per_component


def diagnose_model(
    model_id_or_path: str,
    histograms_dir: str | Path | None = None,
    derived_dir: str | Path | None = None,
    training_config: dict | None = None,
    arch_config: dict | None = None,
) -> dict:
    """One-shot Layer A diagnostic — chain F1 + F2 + F4 + F5 + F6 + F7 + F8 + F6_extension.

    **NO benchmark coupling** — universal applicable to any transformer.
    For benchmark comparison, use compare_to_benchmark() separately.

    Parameters
    ----------
    model_id_or_path : str
        Identifier for naming the diagnosis (e.g., 'olmo-7b-hf' or path)
    histograms_dir : str | Path | None
        Optional directory containing raw 1024-bin log10|w| NPZ files.
        If None, must provide derived_dir.
    derived_dir : str | Path | None
        Optional cascade v3 derived directory containing fit_per_component_v3.json.
        If given, skips re-fitting and uses pre-computed (k, λ, R²) per layer.
    training_config : dict | None
        {'eta': float, 'lambda_wd': float, 'T_steps': int} for F6 T/τ.
        If None, F6 not computed.
    arch_config : dict | None
        {'n_q': int, 'n_kv': int} for F8 architectural classifier.
        If None, F8 not computed.

    Returns
    -------
    dict (Layer A raw diagnosis, NO benchmark comparison):
        model_id : str
        arch : dict | None — F8 output if arch_config given
        T_tau : dict | None — F6 output if training_config given
        per_layer_fits : dict {kind: [{block_idx, k, lambda, R2, KS, classification}]}
        per_component_summary : dict {kind: median_k}
        distfree : dict {kind: list[{block_idx, q90_q10, p999_p50, gini}]}  (R²<0.95 fallback)
        k_drift : dict {kind: {delta_pct, severity}}  (vs init Half-Normal anchor 1.205)
        alerts : list[str]
    """
    # Step 1: F8 architectural classifier
    arch = None
    if arch_config is not None:
        arch = classify_attention_arch(arch_config["n_q"], arch_config["n_kv"])

    # Step 2: F6 T/τ training progress
    T_tau = None
    if training_config is not None:
        T_tau = compute_T_tau(
            eta=training_config["eta"],
            lambda_wd=training_config["lambda_wd"],
            T_steps=training_config["T_steps"],
        )

    # Step 3: load fits (either pre-computed from derived_dir OR fresh from histograms_dir)
    per_layer_fits = {}
    distfree = {}

    if derived_dir is not None:
        # Use pre-computed cascade v3 fits
        all_data = load_cascade_v3(derived_dir)
        # Find matching model entry (heuristic prefix match)
        matching = [k for k in all_data if model_id_or_path.split("/")[-1].lower() in k.lower()]
        if not matching:
            raise ValueError(f"No matching cascade entry for {model_id_or_path!r} in {derived_dir}")
        pc = all_data[matching[0]]
        for kind in ("q", "k", "v", "o", "gate", "up", "down", "qkv"):
            rows = filter_per_component(pc, kind=kind)
            if rows:
                per_layer_fits[kind] = [
                    {
                        "block_idx": r.get("block_idx"),
                        "k": r.get("k"),
                        "lambda": r.get("lambda"),
                        "R2": r.get("R2"),
                        "KS": None,  # cascade v3 schema may not have KS
                        "ok": (r.get("k") is not None and r.get("lambda") is not None),
                    }
                    for r in rows
                ]

    elif histograms_dir is not None:
        # Fresh fit from raw NPZ files (per-layer + per-kind)
        # Schema: {model_step}_layer{NNN}.npz with 'param_name' field encoding kind
        hist_path = Path(histograms_dir)
        if not hist_path.is_dir():
            raise FileNotFoundError(f"histograms_dir not found: {histograms_dir}")
        # Group NPZ files by inferred kind from param_name
        import numpy as np
        for npz_fp in sorted(hist_path.glob("*.npz")):
            d = np.load(npz_fp)
            param_name = str(d.get("param_name", ""))
            kind = _infer_kind_from_param_name(param_name)
            if kind is None:
                continue
            block_idx = _infer_block_idx_from_param_name(param_name)
            fit = weibull_fit({"edges": d["edges"], "hist": d["hist"]}, trim="mid_80")
            entry = {
                "block_idx": block_idx,
                "k": fit.get("k"),
                "lambda": fit.get("lambda"),
                "R2": fit.get("R2"),
                "KS": fit.get("KS"),
                "ok": fit.get("ok", False),
            }
            per_layer_fits.setdefault(kind, []).append(entry)
            # F6_extension distfree fallback for R²<0.95
            if not fit["ok"] or (fit.get("R2") is not None and fit["R2"] < 0.95):
                metrics = per_block_metrics({"edges": d["edges"], "hist": d["hist"]})
                metrics["block_idx"] = block_idx
                distfree.setdefault(kind, []).append(metrics)
    else:
        raise ValueError(
            "must provide histograms_dir OR derived_dir for diagnose_model"
        )

    # Step 4: per-component median + classification
    per_component_summary = {}
    classifications = {}
    alerts = []
    for kind, fits in per_layer_fits.items():
        ks = [f["k"] for f in fits if f.get("ok") and f.get("k") is not None]
        if not ks:
            continue
        med_k = statistics.median(ks)
        per_component_summary[kind] = med_k
        # F2 classify
        arch_type = arch["arch"] if arch else None
        cls = classify_k(med_k, arch_type=arch_type)
        classifications[kind] = cls

    # Step 5: F5 k drift vs init Half-Normal anchor
    k_drift = {}
    for kind, med_k in per_component_summary.items():
        drift = k_drift_severity(k_init=1.205, k_trained=med_k)
        k_drift[kind] = drift

    # Step 6: alerts
    if T_tau and T_tau.get("warning"):
        alerts.append(T_tau["warning"])
    if arch is not None:
        for kind, cls in classifications.items():
            if cls.get("in_expected_range") is False and kind in ("q", "k"):
                alerts.append(
                    f"⚠️ {kind}_proj median k = {per_component_summary[kind]:.3f} "
                    f"falls outside expected range for {arch['arch']}"
                )

    return {
        "model_id": model_id_or_path,
        "arch": arch,
        "T_tau": T_tau,
        "per_layer_fits": per_layer_fits,
        "per_component_summary": per_component_summary,
        "classifications": classifications,
        "distfree": distfree,
        "k_drift": k_drift,
        "alerts": alerts,
    }


def _infer_kind_from_param_name(param_name: str) -> str | None:
    """Infer component kind from HF parameter name."""
    n = param_name.lower()
    if "q_proj" in n or "self_attn.query" in n:
        return "q"
    if "k_proj" in n or "self_attn.key" in n:
        return "k"
    if "v_proj" in n or "self_attn.value" in n:
        return "v"
    if "o_proj" in n or "self_attn.output" in n or "self_attn.dense" in n:
        return "o"
    if "qkv" in n or "query_key_value" in n:
        return "qkv"
    if "gate" in n:
        return "gate"
    if "up_proj" in n or "h_to_4h" in n:
        return "up"
    if "down_proj" in n or "4h_to_h" in n:
        return "down"
    return None


def _infer_block_idx_from_param_name(param_name: str) -> int:
    """Infer transformer block index from parameter name (e.g., 'model.layers.5.self_attn...')."""
    import re
    m = re.search(r"layers?\.(\d+)\.", param_name)
    if m:
        return int(m.group(1))
    return -1
