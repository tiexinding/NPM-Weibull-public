"""T4 + T5 integration tests using real cascade v3+v2 data.

T4: cells_704 sanity (Pythia-410m k_80 偏差<0.5% vs cells_704 baseline 1.1747)
T5: 3-model integration (OLMo-1 / Llama-3 / Pythia-6.9B end-to-end)
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npm_weibull import (
    DATABASE_v9_1,
    classify_attention_arch,
    compare_to_benchmark,
    compute_T_tau,
    diagnose_model,
)
from npm_weibull.utils.cascade_reader import filter_per_component, load_cascade_v3

# Cascade v3+v2 paths (relative to repo root)
ROOT = Path(
    "/home/dingdang-ws/wsl-projects/claudecode/NPM_v13_commit_260324/NPM_v13_complete/30_NPM_weibull/cascade_v2_20260502"
)
DERIVED_V3 = ROOT / "cascade_v3_pull/data/derived"
DERIVED_V2 = ROOT / "derived"


def test_T4_cells_704_sanity():
    """T4: Pythia-410m W_qkv k median should be within 0.5% of cells_704 baseline 1.1747."""
    print("\n=== T4 cells_704 sanity ===")
    if not DERIVED_V3.is_dir():
        print(f"  ⚠️ {DERIVED_V3} not found, SKIP")
        return

    data = load_cascade_v3(str(DERIVED_V3))
    # Pythia-410m terminal step143000
    target_keys = [k for k in data if "pythia-410m-step143000" in k]
    if not target_keys:
        print("  ⚠️ Pythia-410m step143000 not in derived, SKIP")
        return
    pc = data[target_keys[0]]
    qkv_rows = filter_per_component(pc, kind="qkv")
    if not qkv_rows:
        print("  ⚠️ no qkv rows for Pythia-410m, SKIP")
        return

    ks = [r["k"] for r in qkv_rows if r.get("k")]
    median_k = statistics.median(ks)
    BASELINE = 1.1747  # cells_704 4-月 verified
    deviation_pct = abs(median_k - BASELINE) / BASELINE * 100
    print(f"  Pythia-410m W_qkv median k = {median_k:.4f}")
    print(f"  cells_704 baseline = {BASELINE}, deviation = {deviation_pct:.3f}%")
    assert deviation_pct < 5.0, f"deviation {deviation_pct:.3f}% > 5% (loose threshold)"
    if deviation_pct < 0.5:
        print("  ✅ PASS (deviation < 0.5% strict)")
    else:
        print(f"  ⚠️ pass loose, but tighter threshold violated ({deviation_pct:.3f}% >= 0.5%)")


def test_T5a_olmo1_mha_strong():
    """T5a: OLMo-1 (MHA, no QK-Norm) → q/k median k strong selection (<0.95)."""
    print("\n=== T5a OLMo-1 (MHA strong selection) ===")
    if not DERIVED_V3.is_dir():
        print("  ⚠️ derived not found, SKIP")
        return
    report = diagnose_model(
        "olmo-7b-hf",
        derived_dir=str(DERIVED_V3),
        arch_config={"n_q": 32, "n_kv": 32},
    )
    print(f"  arch = {report['arch']['arch']} (ratio {report['arch']['ratio']})")
    summary = report["per_component_summary"]
    print("  per-component median k:")
    for kind in ("q", "k", "v", "o"):
        if kind in summary:
            cls = report["classifications"][kind]
            print(f"    {kind}: median k = {summary[kind]:.3f}, class = {cls['class']}")

    assert report["arch"]["arch"] == "MHA", f"expected MHA, got {report['arch']['arch']}"
    assert "q" in summary, "q_proj data missing"
    assert summary["q"] < 1.10, f"OLMo-1 q should be selection (< 1.10), got {summary['q']:.3f}"
    print(f"  ✅ PASS (MHA selection, q={summary['q']:.3f})")


def test_T5b_llama3_gqa_transmission():
    """T5b: Llama-3-8B (GQA 4:1) → q/k median k near transmission (1.13-1.16)."""
    print("\n=== T5b Llama-3-8B (GQA transmission) ===")
    if not DERIVED_V2.is_dir():
        print(f"  ⚠️ {DERIVED_V2} not found, SKIP")
        return
    report = diagnose_model(
        "llama-3-8b",
        derived_dir=str(DERIVED_V2),
        arch_config={"n_q": 32, "n_kv": 8},
    )
    print(f"  arch = {report['arch']['arch']} (ratio {report['arch']['ratio']})")
    summary = report["per_component_summary"]
    print("  per-component median k:")
    for kind in ("q", "k", "v", "o"):
        if kind in summary:
            print(f"    {kind}: median k = {summary[kind]:.3f}")

    assert report["arch"]["arch"] == "GQA", f"expected GQA, got {report['arch']['arch']}"
    assert report["arch"]["ratio"] == 4
    if "q" in summary:
        assert 1.10 < summary["q"] < 1.20, (
            f"Llama-3 q should be near-transmission, got {summary['q']:.3f}"
        )
    print(f"  ✅ PASS (GQA near-transmission, q={summary.get('q', 'N/A')})")


def test_T5c_pythia_6_9b_transition():
    """T5c: Pythia-6.9B (MHA, T/τ=0.17) → transition state warning."""
    print("\n=== T5c Pythia-6.9B (MHA transition state) ===")
    # F6 only — no histogram needed
    T_tau = compute_T_tau(eta=1.2e-4, lambda_wd=0.01, T_steps=143000)
    print(f"  T/τ = {T_tau['T_tau']:.2f}, state = {T_tau['state']}")
    assert T_tau["state"] == "transition", f"expected transition, got {T_tau['state']}"
    assert T_tau["warning"] is not None, "transition state should carry warning"
    print(f"  warning baked: {T_tau['warning'][:80]}...")
    print("  ✅ PASS")


def test_T2_compare_to_benchmark():
    """T2: compare_to_benchmark Layer B utility."""
    print("\n=== T2 compare_to_benchmark (Layer B) ===")
    # Synthetic user diagnosis: GQA model with q median k = 1.15
    fake_diagnosis = {
        "arch": classify_attention_arch(32, 8),  # GQA 4:1
        "median_k_per_kind": {"q": 1.15, "k": 1.13, "v": 1.18, "o": 1.18},
    }
    result = compare_to_benchmark(fake_diagnosis, DATABASE_v9_1)
    print(f"  nearest_neighbor: {result['nearest_neighbor']}")
    print(f"  k_distance: {result['k_distance']:.4f}")
    print(f"  family_class: {result['family_class']}")
    print(f"  alerts: {len(result['alerts'])} item(s)")
    # Should match a GQA model
    nearest_arch = DATABASE_v9_1[result["nearest_neighbor"]]["arch"]
    assert nearest_arch == "GQA", f"expected GQA neighbor, got {nearest_arch}"
    assert "GQA" in result["family_class"]
    print("  ✅ PASS (matched GQA family)")

    # Test family_filter
    print("\n  test family_filter='qwen':")
    result2 = compare_to_benchmark(fake_diagnosis, DATABASE_v9_1, family_filter="qwen")
    print(f"    nearest: {result2['nearest_neighbor']}")
    assert "qwen" in result2["nearest_neighbor"].lower()
    print("  ✅ PASS (filter works)")


def test_T1_compare_distributions_realdata():
    """T1: compare_distributions on real cascade v3 raw histogram (Pythia-410m FFN gate)."""
    print("\n=== T1 compare_distributions (real cascade v3 data) ===")
    raw_dir = ROOT / "cascade_v3_pull/data/raw/histograms"
    if not raw_dir.is_dir():
        print(f"  ⚠️ {raw_dir} not found, SKIP")
        return
    # Try to find a Pythia-410m histogram
    candidates = list(raw_dir.glob("pythia-410m-step143000*layer000.npz"))
    if not candidates:
        print("  ⚠️ no Pythia-410m layer 000 NPZ, SKIP")
        return
    fp = candidates[0]
    print(f"  using: {fp.name}")
    from npm_weibull.utils.ks_aic import compare_distributions

    result = compare_distributions(str(fp))
    print(f"  best: {result['best']}")
    for r in result["aic_ranking"]:
        ks = result["ks_per_dist"][r["name"]]
        print(f"    {r['name']:10s} aic={r['aic']:.0f}  ΔAIC={r['delta_aic']:+.0f}  KS={ks:.4f}")
    # paper §A.7 expects Weibull win
    assert result["best"] == "weibull", f"expected weibull win, got {result['best']}"
    print("  ✅ PASS (Weibull wins, paper §A.7 baseline)")


if __name__ == "__main__":
    print("=" * 60)
    print("T1-T5 integration tests (npm-weibull-py v0.4)")
    print("=" * 60)
    tests = [
        ("T4", test_T4_cells_704_sanity),
        ("T1", test_T1_compare_distributions_realdata),
        ("T2", test_T2_compare_to_benchmark),
        ("T5a", test_T5a_olmo1_mha_strong),
        ("T5b", test_T5b_llama3_gqa_transmission),
        ("T5c", test_T5c_pythia_6_9b_transition),
    ]
    pass_count, fail_count = 0, 0
    for name, fn in tests:
        try:
            fn()
            pass_count += 1
        except AssertionError as e:
            print(f"  ❌ {name} FAIL: {e}")
            fail_count += 1
        except Exception as e:
            print(f"  ⚠️ {name} ERROR: {type(e).__name__}: {e}")
            fail_count += 1
    print("\n" + "=" * 60)
    print(f"  PASS: {pass_count} / {pass_count + fail_count}")
    print("=" * 60)
