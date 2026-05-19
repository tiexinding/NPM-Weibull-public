"""Example 2 — Inspect DATABASE_v9_1 and compare a synthetic diagnosis to it.

Demonstrates the 12-entry benchmark database shipped with the library and the
two-step pattern from the paper: user runs diagnose_model() (Layer A, here
mocked with a hand-built diagnosis), then calls compare_to_benchmark() (Layer
B) to find the nearest neighbor by per-component k distance.

Run:
    python examples/02_compare_to_benchmark.py
"""

from __future__ import annotations

from npm_weibull import DATABASE_v9_1, classify_attention_arch, compare_to_benchmark


def show_benchmark_overview():
    print("=== DATABASE_v9_1 entries ===")
    print(f"  total: {len(DATABASE_v9_1)} entries\n")
    print(f"  {'id':<28} {'arch':<6} {'n_q/n_kv':<10} {'QK-Norm':<8} {'tokens(T)'}")
    for model_id, meta in DATABASE_v9_1.items():
        nq = meta.get("n_q", "-")
        nkv = meta.get("n_kv", "-")
        qkn = "yes" if meta.get("QK_Norm") else "no"
        tok = meta.get("training_tokens_T", "-")
        print(f"  {model_id:<28} {meta.get('arch', '-'):<6} {f'{nq}/{nkv}':<10} {qkn:<8} {tok}")


def main():
    show_benchmark_overview()

    print("\n=== F8 attention architecture classification ===")
    for n_q, n_kv, label in [
        (40, 8, "Qwen2.5 14B"),
        (32, 8, "Llama-3 8B"),
        (32, 32, "OLMo-1 7B"),
        (32, 1, "hypothetical MQA"),
    ]:
        c = classify_attention_arch(n_q, n_kv)
        print(f"  {label:<22} (n_q={n_q}, n_kv={n_kv}): arch={c['arch']}, ratio={c['ratio']}:1")

    # Mock a diagnosis on a GQA 4:1 model with k values in the Transmission band
    user_diagnosis = {
        "arch": {"arch": "GQA", "n_q": 32, "n_kv": 8, "ratio": 4},
        "median_k_per_kind": {
            "q": 1.14,
            "k": 1.13,
            "v": 1.19,
            "o": 1.19,
        },
    }

    cmp = compare_to_benchmark(user_diagnosis)
    print("\n=== compare_to_benchmark result ===")
    print(f"  nearest_neighbor : {cmp['nearest_neighbor']}")
    print(f"  k_distance       : {cmp['k_distance']:.4f}")
    print(f"  family_class     : {cmp['family_class']}")
    for alert in cmp.get("alerts", []):
        print(f"  {alert}")


if __name__ == "__main__":
    main()
