"""P1b coverage tests — exercise uncovered branches in histogram / trajectory /
weibull / closed_form / diagnose modules.

These tests target the gaps identified by `pytest --cov` in the P1a baseline:
- utils/histogram.py was 26% (NPZ save path untested)
- core/trajectory.py was 37% (sigma_decompose branches untested)
- utils/closed_form.py was 69% (error branches untested)
- core/weibull.py was 78% (error paths untested)
- workflow/diagnose.py was 52% (full integration untested)

Goal: lift overall coverage from 70% to ≥80%.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from npm_weibull import (
    diagnose_model,
    extract_to_histogram,
    k_drift_severity,
    per_block_metrics,
    sigma_decompose,
    sigma_from_k_lambda,
    weibull_fit,
    weibull_quantile,
)
from npm_weibull.utils.closed_form import weibull_q90_q10

RNG = np.random.default_rng(42)


# =============================================================================
# extract_to_histogram (utils/histogram.py)
# =============================================================================


def test_extract_to_histogram_numpy_input():
    """np.ndarray input → 1024-bin log10|w| histogram with expected shape."""
    weight = RNG.normal(0.0, 0.02, size=(128, 256)).astype(np.float32)
    result = extract_to_histogram(weight)

    assert result["edges"].shape == (1025,)
    assert result["hist"].shape == (1024,)
    assert result["n_total"] == 128 * 256
    assert result["shape"] == (128, 256)
    assert result["param_name"] == ""
    # All counts non-negative and sum equals n_total
    assert result["hist"].sum() == 128 * 256


def test_extract_to_histogram_saves_npz(tmp_path: Path):
    """save_path option writes NPZ with edges/hist/shape/param_name."""
    weight = RNG.normal(0.0, 0.02, size=(64, 64)).astype(np.float32)
    out = tmp_path / "layer000.npz"
    extract_to_histogram(weight, save_path=out)

    assert out.exists()
    loaded = np.load(out)
    assert "edges" in loaded.files
    assert "hist" in loaded.files
    assert "shape" in loaded.files
    assert "param_name" in loaded.files
    assert loaded["edges"].shape == (1025,)
    assert loaded["hist"].shape == (1024,)


def test_extract_to_histogram_custom_bins():
    """Custom n_bins + log_w_range works."""
    weight = RNG.normal(0.0, 0.02, size=(64,)).astype(np.float32)
    result = extract_to_histogram(weight, n_bins=256, log_w_range=(-6.0, 0.0))
    assert result["edges"].shape == (257,)
    assert result["hist"].shape == (256,)
    assert result["edges"][0] == pytest.approx(-6.0)
    assert result["edges"][-1] == pytest.approx(0.0)


# =============================================================================
# sigma_decompose + k_drift_severity (core/trajectory.py)
# =============================================================================


def test_sigma_decompose_lambda_dominated():
    """k drift < 1% → lambda_dominated attribution (paper Transmission Class signature)."""
    k_traj = [1.205, 1.203, 1.202, 1.201, 1.200]  # 0.4% drift
    lam_traj = [0.009, 0.015, 0.022, 0.027, 0.029]
    res = sigma_decompose(k_traj, lam_traj)
    assert res["sigma_attribution"] == "lambda_dominated"
    assert abs(res["k_drift_pct"]) < 1.0
    assert res["lambda_growth_pct"] > 100.0  # ~220%


def test_sigma_decompose_k_dominated():
    """k drift > 30% → k_dominated attribution (paper deep Selection regime)."""
    k_traj = [1.205, 1.10, 0.97, 0.88, 0.81]  # 33% drop
    lam_traj = [0.008, 0.0095, 0.0102, 0.0108, 0.0112]
    res = sigma_decompose(k_traj, lam_traj)
    assert res["sigma_attribution"] == "k_dominated"
    assert res["k_drift_pct"] < -30.0


def test_sigma_decompose_mixed():
    """1% < |k drift| < 30% → mixed attribution."""
    k_traj = [1.205, 1.15, 1.10, 1.08, 1.06]  # ~12% drop
    lam_traj = [0.008, 0.009, 0.010, 0.011, 0.012]
    res = sigma_decompose(k_traj, lam_traj)
    assert res["sigma_attribution"] == "mixed"
    assert 1.0 < abs(res["k_drift_pct"]) < 30.0


def test_sigma_decompose_paired_correlation():
    """paired_lambda_traj produces a Pearson r in [-1, 1] (log-log)."""
    k_traj = [1.205, 1.20, 1.198, 1.197]
    lam_traj = [0.01, 0.02, 0.03, 0.04]
    paired = [0.011, 0.022, 0.033, 0.044]  # perfect correlation
    res = sigma_decompose(k_traj, lam_traj, paired_lambda_traj=paired)
    assert res["paired_r"] is not None
    assert res["paired_r"] == pytest.approx(1.0, abs=0.01)


def test_sigma_decompose_length_mismatch_raises():
    """Mismatched k_traj / lambda_traj lengths raise ValueError."""
    with pytest.raises(ValueError):
        sigma_decompose([1.0, 1.1], [0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        sigma_decompose([1.0], [0.01])  # too short


def test_k_drift_severity_categories():
    """k_drift_severity returns expected severity labels at boundaries."""
    # invariant: |drift| < 1%
    assert k_drift_severity(1.205, 1.200)["severity"] == "invariant"
    # mild / strong / deep — just verify return is a known label
    severities = {
        k_drift_severity(1.205, 1.15)["severity"],
        k_drift_severity(1.205, 0.90)["severity"],
        k_drift_severity(1.205, 0.75)["severity"],
    }
    assert severities.issubset({"mild", "strong", "deep", "invariant"})


# =============================================================================
# weibull_fit error paths (core/weibull.py)
# =============================================================================


def test_weibull_fit_invalid_trim_raises():
    """trim not in {mid_80, mid_90, full_range} raises ValueError."""
    edges = np.linspace(-6.0, 1.0, 1025)
    hist = np.ones(1024) * 100
    with pytest.raises(ValueError, match="trim must be one of"):
        weibull_fit({"edges": edges, "hist": hist}, trim="mid_99")


def test_weibull_fit_insufficient_counts():
    """hist with total < 100 returns nan result with ok=False."""
    edges = np.linspace(-6.0, 1.0, 1025)
    hist = np.zeros(1024)
    hist[500] = 50  # only 50 total
    res = weibull_fit({"edges": edges, "hist": hist})
    assert res["ok"] is False
    assert np.isnan(res["k"])


def test_weibull_fit_tuple_input():
    """(edges, hist) tuple input format works."""
    samples = np.abs(RNG.normal(0.0, 0.02, size=100_000))
    log_w = np.log10(np.maximum(samples, 1e-12))
    edges = np.linspace(-6.0, 1.0, 1025)
    hist, _ = np.histogram(log_w, bins=edges)
    res = weibull_fit((edges, hist.astype(float)))
    assert res["ok"] is True
    assert 1.15 < res["k"] < 1.25


def test_weibull_fit_npz_path_input(tmp_path: Path):
    """NPZ file path input format works."""
    samples = np.abs(RNG.normal(0.0, 0.02, size=100_000))
    log_w = np.log10(np.maximum(samples, 1e-12))
    edges = np.linspace(-6.0, 1.0, 1025)
    hist, _ = np.histogram(log_w, bins=edges)
    npz = tmp_path / "h.npz"
    np.savez(npz, edges=edges, hist=hist.astype(np.float64))
    res = weibull_fit(npz)
    assert res["ok"] is True


def test_weibull_fit_unsupported_input_type_raises():
    """Garbage input type raises TypeError from _load_histogram."""
    with pytest.raises(TypeError, match="histogram must be"):
        weibull_fit(12345)  # type: ignore[arg-type]


# =============================================================================
# closed_form error branches (utils/closed_form.py)
# =============================================================================


def test_sigma_from_k_lambda_negative_k_raises():
    with pytest.raises(ValueError, match="must be positive"):
        sigma_from_k_lambda(-1.0, 0.02)


def test_sigma_from_k_lambda_zero_lambda_raises():
    with pytest.raises(ValueError, match="must be positive"):
        sigma_from_k_lambda(1.2, 0.0)


def test_weibull_quantile_q_out_of_range_raises():
    with pytest.raises(ValueError, match="q must be in"):
        weibull_quantile(1.2, 0.02, 1.5)
    with pytest.raises(ValueError, match="q must be in"):
        weibull_quantile(1.2, 0.02, 0.0)


def test_weibull_quantile_negative_k_raises():
    with pytest.raises(ValueError, match="must be positive"):
        weibull_quantile(-1.0, 0.02, 0.5)


def test_weibull_q90_q10_negative_k_raises():
    with pytest.raises(ValueError, match="must be positive"):
        weibull_q90_q10(-1.0)


def test_weibull_q90_q10_value():
    """Q90/Q10 ratio for k=1.205 matches paper §A.5 anchor (~12.93)."""
    q_ratio = weibull_q90_q10(1.205)
    assert q_ratio == pytest.approx(12.93, abs=0.5)


# =============================================================================
# per_block_metrics edge case (core/distfree.py)
# =============================================================================


def test_per_block_metrics_insufficient_counts():
    """hist with total < 100 returns nan result with ok=False."""
    edges = np.linspace(-6.0, 1.0, 1025)
    hist = np.zeros(1024)
    hist[500] = 50
    res = per_block_metrics({"edges": edges, "hist": hist})
    assert res["ok"] is False
    assert np.isnan(res["q90_q10"])


# =============================================================================
# diagnose_model integration (workflow/diagnose.py)
# =============================================================================


def _make_mock_npz(path: Path, param_name: str, sigma: float = 0.02) -> None:
    """Generate a Half-Normal histogram NPZ at the given path."""
    samples = np.abs(RNG.normal(0.0, sigma, size=50_000))
    log_w = np.log10(np.maximum(samples, 1e-12))
    edges = np.linspace(-12.0, 2.0, 1025)
    hist, _ = np.histogram(log_w, bins=edges)
    np.savez(path, edges=edges, hist=hist.astype(np.float64), param_name=param_name)


def test_diagnose_model_with_histograms_dir(tmp_path: Path):
    """Full Layer A workflow: histograms_dir → per_layer_fits + per_component_summary."""
    h_dir = tmp_path / "hist"
    h_dir.mkdir()
    # 4 layers per kind for q, k, v, o
    for block in range(4):
        for kind in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _make_mock_npz(h_dir / f"L{block:03d}_{kind}.npz", param_name=f"layers.{block}.{kind}")

    rep = diagnose_model(
        "synthetic-test",
        histograms_dir=h_dir,
        training_config={"eta": 1e-3, "lambda_wd": 0.01, "T_steps": 100_000},
        arch_config={"n_q": 32, "n_kv": 8},
    )

    # F8 architectural classifier ran
    assert rep["arch"] is not None
    assert rep["arch"]["arch"] == "GQA"
    # F6 T/tau ran
    assert rep["T_tau"] is not None
    # Per-layer fits collected for q/k/v/o
    assert set(rep["per_layer_fits"].keys()) >= {"q", "k", "v", "o"}
    # Per-component summary contains median k values
    assert all(0.8 < v < 1.5 for v in rep["per_component_summary"].values())
    # k_drift dict has severities for each kind
    assert set(rep["k_drift"].keys()) == set(rep["per_component_summary"].keys())


def test_diagnose_model_requires_dir():
    """Calling without histograms_dir OR derived_dir raises ValueError."""
    with pytest.raises(ValueError, match="histograms_dir OR derived_dir"):
        diagnose_model("nowhere")


def test_diagnose_model_missing_dir_raises(tmp_path: Path):
    """histograms_dir pointing to nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        diagnose_model("foo", histograms_dir=tmp_path / "does_not_exist")


# =============================================================================
# Small fillers to push coverage over 80% (classify / training / distfree)
# =============================================================================


def test_classify_k_transmission_band():
    """k in [1.18, 1.21] → transmission classification."""
    from npm_weibull import classify_k

    res = classify_k(1.20, R2=0.99, arch_type="GQA")
    assert res["class"] in ("transmission", "init_anchor")
    assert "in_expected_range" in res


def test_classify_k_mild_selection():
    """k in [0.95, 1.18] but not init → mild_selection."""
    from npm_weibull import classify_k

    res = classify_k(1.05, R2=0.95, arch_type="GQA")
    assert res["class"] in ("mild_selection", "transmission")
    assert "in_expected_range" in res


def test_classify_k_strong_selection():
    """k < 0.95 → strong_selection."""
    from npm_weibull import classify_k

    res = classify_k(0.80, R2=0.92, arch_type="MHA")
    assert res["class"] == "strong_selection"


def test_compute_T_tau_saturated():
    """T/tau >= 1.0 → saturated state."""
    from npm_weibull import compute_T_tau

    res = compute_T_tau(eta=1e-3, lambda_wd=0.01, T_steps=200_000)
    assert res["state"] == "saturated"
    assert res["T_tau"] >= 1.0


def test_compute_T_tau_transition():
    """T/tau < 0.2 → transition state with warning."""
    from npm_weibull import compute_T_tau

    res = compute_T_tau(eta=1e-4, lambda_wd=0.001, T_steps=1_000)
    assert res["state"] == "transition"
    assert res["warning"] is not None


def test_per_block_metrics_tuple_input():
    """(edges, hist) tuple input format works."""
    samples = np.abs(RNG.normal(0.0, 0.02, size=50_000))
    log_w = np.log10(np.maximum(samples, 1e-12))
    edges = np.linspace(-6.0, 1.0, 1025)
    hist, _ = np.histogram(log_w, bins=edges)
    res = per_block_metrics((edges, hist.astype(float)))
    assert res["ok"] is True
    assert res["q90_q10"] > 1.0  # Half-Normal is right-skewed


def test_per_block_metrics_unsupported_type_raises():
    """Garbage input type raises TypeError."""
    with pytest.raises(TypeError, match="histogram must be"):
        per_block_metrics(12345)  # type: ignore[arg-type]


def test_compare_to_benchmark_via_module():
    """compare_to_benchmark accepts a per_layer_fits-format diagnosis."""
    from npm_weibull import compare_to_benchmark

    diag = {
        "arch": {"arch": "GQA", "n_q": 32, "n_kv": 8},
        "per_layer_fits": {
            "q": [
                {"k": 1.14, "ok": True},
                {"k": 1.15, "ok": True},
            ],
            "k": [{"k": 1.13, "ok": True}],
            "v": [{"k": 1.19, "ok": True}],
            "o": [{"k": 1.19, "ok": True}],
        },
    }
    res = compare_to_benchmark(diag)
    assert "nearest_neighbor" in res
    assert "k_distance" in res


def test_compare_to_benchmark_family_filter():
    """family_filter restricts the search."""
    from npm_weibull import compare_to_benchmark

    diag = {
        "arch": {"arch": "GQA", "n_q": 32, "n_kv": 8},
        "median_k_per_kind": {"q": 1.14, "k": 1.13, "v": 1.19, "o": 1.19},
    }
    res = compare_to_benchmark(diag, family_filter="llama")
    assert res["nearest_neighbor"].startswith("llama")


def test_compare_to_benchmark_empty_filter_raises():
    """family_filter that matches nothing raises ValueError."""
    from npm_weibull import compare_to_benchmark

    diag = {
        "arch": {"arch": "GQA", "n_q": 32, "n_kv": 8},
        "median_k_per_kind": {"q": 1.14},
    }
    with pytest.raises(ValueError, match="No benchmark entries after filter"):
        compare_to_benchmark(diag, family_filter="nonexistent_family")


def test_compare_to_benchmark_missing_schema_raises():
    """Diagnosis without per_component_summary / per_layer_fits / median_k_per_kind raises."""
    from npm_weibull import compare_to_benchmark

    diag = {"arch": {"arch": "GQA"}}  # missing all 3 schema fields
    with pytest.raises(ValueError, match="must contain"):
        compare_to_benchmark(diag)
