"""Populate DATABASE_v9_1_qwen_cohort.csv + .md for the 11-entry Qwen-family cohort.

Companion to populate_database_v9_1.py (which generates the 12-entry main cohort).
Generates the Qwen application-case cohort used in Appendix C of the paper:
"A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions"
(arXiv:2605.18898).

Source: cascade_v3 per-component fits (JSON), one file per entry.
Output: CSV + MD with per-kind k/lambda medians + shallow/deep splits used in paper Table 7.

Usage:
    python populate_qwen_cohort.py [--cascade-root <path>]

Env override: NPM_CASCADE_ROOT
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path


# --- per-entry metadata (training tokens + Math CPT status + regime classification) ---
# These match paper Table 7 (tab:qwen-cohort).
QWEN_ENTRIES = [
    # (entry_id, family_gen, size, n_layers, tokens_str, math_cpt, regime, json_basename)
    ("Qwen2-1.5B-base",    "Qwen2",   "1.5B", 28, "7T",                False, "A", "qwen2-1.5b-base"),
    ("Qwen2.5-1.5B",       "Qwen2.5", "1.5B", 28, "18T",               False, "A", "qwen2.5-1.5b"),
    ("Qwen2-Math-1.5B",    "Qwen2",   "1.5B", 28, "7T + Math CPT",     True,  "A", "qwen2-math-1.5b"),
    ("Qwen2.5-Math-1.5B",  "Qwen2.5", "1.5B", 28, "18T + Math CPT",    True,  "A", "qwen2.5-math-1.5b"),
    ("Qwen2-7B",           "Qwen2",   "7B",   28, "7T",                False, "B", "qwen2-7b"),
    ("Qwen2.5-7B",         "Qwen2.5", "7B",   28, "18T",               False, "B", "qwen2.5-7b"),
    ("Qwen2-Math-7B",      "Qwen2",   "7B",   28, "7T + 200B Math",    True,  "B", "qwen2-math-7b"),
    ("Qwen2.5-Math-7B",    "Qwen2.5", "7B",   28, "18T + 700B Math",   True,  "B", "qwen2.5-math-7b"),
    ("Qwen2.5-3B",         "Qwen2.5", "3B",   36, "18T",               False, "B", "qwen2.5-3b"),
    ("Qwen3-8B",           "Qwen3",   "8B",   36, "36T",               False, "B", "qwen3-8b-base"),
    ("Qwen2.5-14B",        "Qwen2.5", "14B",  48, "18T",               False, "B", "qwen2.5-14b"),
]


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def load_per_component(path: Path) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    return d["per_component"]


def stats_for_entry(pc: list[dict]) -> dict:
    """Compute per-kind k/lambda medians + shallow/deep splits."""
    by_kind: dict[str, list[dict]] = {}
    for entry in pc:
        by_kind.setdefault(entry["kind"], []).append(entry)

    out = {}
    # Per-kind terminal medians (across all blocks)
    for kind in ["q", "k", "v", "o", "gate", "up", "down"]:
        entries = by_kind.get(kind, [])
        ks = [e["k"] for e in entries]
        lambdas = [e["lambda"] for e in entries]
        out[f"k_median_{kind}"] = round(median(ks), 4) if ks else None
        out[f"lambda_median_{kind}"] = round(median(lambdas), 6) if lambdas else None

    # Shallow (blocks 0-5) and deep (blocks 10+) splits used in paper Table 7
    gate_entries = by_kind.get("gate", [])
    shallow_gate_ks = [e["k"] for e in gate_entries if 0 <= e["block_idx"] <= 5]
    deep_gate_ks = [e["k"] for e in gate_entries if e["block_idx"] >= 10]
    out["shallow_median_k_gate"] = round(median(shallow_gate_ks), 3) if shallow_gate_ks else None
    out["deep_median_k_gate"] = round(median(deep_gate_ks), 3) if deep_gate_ks else None

    # Shallow min k across {gate, up, down}
    shallow_min_k = None
    for kind in ["gate", "up", "down"]:
        ks = [e["k"] for e in by_kind.get(kind, []) if 0 <= e["block_idx"] <= 5]
        if ks:
            kmin = min(ks)
            if shallow_min_k is None or kmin < shallow_min_k:
                shallow_min_k = kmin
    out["shallow_min_k"] = round(shallow_min_k, 3) if shallow_min_k is not None else None

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cascade-root",
        default=os.environ.get(
            "NPM_CASCADE_ROOT",
            "/home/dingdang-ws/wsl-projects/claudecode/NPM_v13_commit_260324/NPM_v13_complete/30_NPM_weibull/cascade_v2_20260502/cascade_v3_pull/data/derived",
        ),
    )
    args = parser.parse_args()
    cascade_root = Path(args.cascade_root)

    rows = []
    for entry_id, family_gen, size, n_layers, tokens, math_cpt, regime, basename in QWEN_ENTRIES:
        json_path = cascade_root / f"{basename}_main_fit_per_component_v3.json"
        if not json_path.exists():
            print(f"⚠️  Missing: {json_path}")
            continue
        pc = load_per_component(json_path)
        stats = stats_for_entry(pc)

        row = {
            "entry_id": entry_id,
            "family": family_gen,
            "size": size,
            "n_layers": n_layers,
            "tokens": tokens,
            "math_cpt": math_cpt,
            "regime": regime,
            "shallow_median_k_gate": stats["shallow_median_k_gate"],
            "shallow_min_k": stats["shallow_min_k"],
            "deep_median_k_gate": stats["deep_median_k_gate"],
        }
        # Add per-kind terminal medians
        for kind in ["q", "k", "v", "o", "gate", "up", "down"]:
            row[f"k_median_{kind}"] = stats[f"k_median_{kind}"]
            row[f"lambda_median_{kind}"] = stats[f"lambda_median_{kind}"]
        rows.append(row)

    # Output CSV
    out_dir = Path(__file__).parent
    csv_path = out_dir / "DATABASE_v9_1_qwen_cohort.csv"
    fieldnames = [
        "entry_id", "family", "size", "n_layers", "tokens", "math_cpt", "regime",
        "shallow_median_k_gate", "shallow_min_k", "deep_median_k_gate",
        "k_median_q", "lambda_median_q",
        "k_median_k", "lambda_median_k",
        "k_median_v", "lambda_median_v",
        "k_median_o", "lambda_median_o",
        "k_median_gate", "lambda_median_gate",
        "k_median_up", "lambda_median_up",
        "k_median_down", "lambda_median_down",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"✅ Wrote {csv_path} ({len(rows)} rows)")

    # Output Markdown summary (paper Table 7 schema)
    md_path = out_dir / "DATABASE_v9_1_qwen_cohort.md"
    with open(md_path, "w") as f:
        f.write("# `DATABASE_v9_1` Qwen Cohort — Application Case (Appendix C)\n\n")
        f.write("11-entry Qwen-family cohort used in the diagnostic application case "
                "(Appendix C of arXiv:2605.18898). Surfaces the shallow-FFN bimodal "
                "anomaly in 7B+ Qwen entries.\n\n")
        f.write("## Schema\n\n")
        f.write("- `regime`: A (clean, shallow min $k \\geq 1.0$) or B (bimodal, shallow min $k \\leq 0.7$)\n")
        f.write("- `shallow_median_k_gate`: median $k$ for $W_\\text{gate}$ across blocks 0–5\n")
        f.write("- `shallow_min_k`: minimum block-level $k$ across $W_\\text{gate}, W_\\text{up}, W_\\text{down}$ within blocks 0–5\n")
        f.write("- `deep_median_k_gate`: median $k$ for $W_\\text{gate}$ across blocks ≥ 10\n")
        f.write("- `k_median_*`, `lambda_median_*`: per-kind terminal-checkpoint medians across all layers\n\n")
        f.write("## Table (paper Table 7 schema)\n\n")
        f.write("| Entry | Layers | Tokens | Shal med $k_g$ | Shal min $k$ | Deep med $k_g$ | Regime |\n")
        f.write("|---|---:|---|---:|---:|---:|:---:|\n")
        for row in rows:
            f.write(f"| {row['entry_id']} | {row['n_layers']} | {row['tokens']} | "
                    f"{row['shallow_median_k_gate']} | {row['shallow_min_k']} | "
                    f"{row['deep_median_k_gate']} | {row['regime']} |\n")
        f.write("\n## Full per-kind medians (terminal checkpoint)\n\n")
        f.write("| Entry | $k_q$ | $k_k$ | $k_v$ | $k_o$ | $k_\\text{gate}$ | $k_\\text{up}$ | $k_\\text{down}$ |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['entry_id']} | {row['k_median_q']} | {row['k_median_k']} | "
                    f"{row['k_median_v']} | {row['k_median_o']} | {row['k_median_gate']} | "
                    f"{row['k_median_up']} | {row['k_median_down']} |\n")
        f.write("\n## Citation\n\n```bibtex\n@misc{ding2026weibull,\n"
                "  title={A Two-Parameter Weibull Framework for Diagnosing Transformer Weight Distributions},\n"
                "  author={Ding, Tiexin},\n  year={2026},\n  eprint={2605.18898},\n  archivePrefix={arXiv}\n}\n```\n")
    print(f"✅ Wrote {md_path}")


if __name__ == "__main__":
    main()
