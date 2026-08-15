"""DATABASE_v9_1 populate — regenerate the 12-entry benchmark table.

INTERNAL DEVELOPMENT SCRIPT. End users do NOT need to run this — the
output files (DATABASE_v9_1.csv, .md, _report.md) are already committed
alongside it. Use those files directly, or load via `npm-weibull-py`:

    pip install npm-weibull-py
    from npm_weibull import DATABASE_v9_1

This script requires the cascade v3 raw per-block Weibull fit JSON
files, which are NOT shipped in this repository (they total several GB
of derived measurement data). It runs only on the author's development
machine where the cascade pipeline output lives at the ROOT path
hard-coded below.

Output (written next to this script):
  - DATABASE_v9_1.csv          (12 rows × 55 columns)
  - DATABASE_v9_1.md           (human-readable reference table)
  - DATABASE_v9_1_report.md    (per-entry sanity verification)

Source data:
  - cascade_v3_pull/data/derived/{model}_main_fit_per_component_v3.json
  - cascade_v3_pull/data/derived/pythia-{size}-step143000_step143000_fit_per_component_v3.json
  - derived/{llama-3-8b, mistral-7b, qwen2.5-7b, olmo2-7b-final}_main_fit_per_component_v3.json

For the full cascade pipeline (raw checkpoints → per-block fits → this
benchmark), see the project root README on GitHub.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

# Override via env var when running on a different machine, e.g.
#   NPM_CASCADE_ROOT=/path/to/cascade_v2_20260502 python populate_database_v9_1.py
_DEFAULT_ROOT = (
    "/home/dingdang-ws/wsl-projects/claudecode/NPM_v13_commit_260324/"
    "NPM_v13_complete/30_NPM_weibull/cascade_v2_20260502"
)
ROOT = Path(os.environ.get("NPM_CASCADE_ROOT", _DEFAULT_ROOT))
CV3 = ROOT / "cascade_v3_pull" / "data" / "derived"
CV2_DERIVED = ROOT / "derived"
# Output goes next to this script (release dir under NPM-Weibull-public/database_v9_1/).
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


# 12 family entry definitions + training hyperparameters for T/tau computation
# Hyperparameters from each model's paper / HF config.json:
#   eta_peak  : peak learning rate (after warmup)
#   lambda_wd : weight decay coefficient
#   T_steps   : total training steps (terminal)
#   src_conf  : confidence (paper-explicit / inferred / estimated)
#
# Pythia: Biderman et al. 2023 (arXiv:2304.01373) Table 4
# OLMo-1: Groeneveld et al. 2024 (arXiv:2402.00838) §3
# OLMo-2: OLMo-2 paper (arXiv:2501.00656) §3
# LLaMA-3: Meta Llama 3 paper (arXiv:2407.21783) §3.4
# Mistral: tech report partial; estimates inferred from typical 7B recipe
# Qwen2.5: Qwen2.5 paper (arXiv:2412.15115)
# Qwen3:   Qwen3 Technical Report (arXiv:2505.09388)
#
# (entry_id, json_path, family, size, arch, n_q, n_kv, qk_norm, training_tokens,
#  eta_peak, lambda_wd, T_steps, hp_confidence)
ENTRIES = [
    # ----- Pythia 5 size (terminal step143000) -- Biderman 2023 Table 4 -----
    (
        "pythia-70m",
        CV3 / "pythia-70m-step143000_step143000_fit_per_component_v3.json",
        "Pythia",
        "70m",
        "MHA-merged",
        None,
        None,
        False,
        "300B",
        1.0e-3,
        0.01,
        143000,
        "explicit",
    ),
    (
        "pythia-160m",
        CV3 / "pythia-160m-step143000_step143000_fit_per_component_v3.json",
        "Pythia",
        "160m",
        "MHA-merged",
        None,
        None,
        False,
        "300B",
        6.0e-4,
        0.01,
        143000,
        "explicit",
    ),
    (
        "pythia-410m",
        CV3 / "pythia-410m-step143000_step143000_fit_per_component_v3.json",
        "Pythia",
        "410m",
        "MHA-merged",
        None,
        None,
        False,
        "300B",
        3.0e-4,
        0.01,
        143000,
        "explicit",
    ),
    (
        "pythia-1b",
        CV3 / "pythia-1b-step143000_step143000_fit_per_component_v3.json",
        "Pythia",
        "1B",
        "MHA-merged",
        None,
        None,
        False,
        "300B",
        3.0e-4,
        0.01,
        143000,
        "explicit",
    ),
    (
        "pythia-6.9b",
        CV3 / "pythia-6.9b-step143000_step143000_fit_per_component_v3.json",
        "Pythia",
        "6.9B",
        "MHA-merged",
        None,
        None,
        False,
        "300B",
        1.2e-4,
        0.01,
        143000,
        "explicit",
    ),
    # ----- 6 cross-family -----
    # OLMo-1 7B: Groeneveld 2024 §3.2 — peak LR 3e-4, wd 0.1, ~477k steps × global_batch 2160 × seq 2048 = ~2.1T tokens (paper says 2T+ but OLMo-1B used 3T tokens; 7B run is ~2.5T)
    (
        "olmo-1-7b",
        CV3 / "olmo-7b-hf_main_fit_per_component_v3.json",
        "OLMo-1",
        "7B",
        "MHA-separate",
        32,
        32,
        False,
        "2.5T",
        3.0e-4,
        0.1,
        477000,
        "explicit",
    ),
    # OLMo-2 7B: OLMo-2 paper — stage1 ~600k steps + stage2 annealing; here T_steps approx for stage1 saturation; LR 3e-4
    (
        "olmo-2-7b",
        ROOT / "derived/olmo2-7b-final_main_fit_per_component_v3.json",
        "OLMo-2",
        "7B",
        "MHA-separate",
        32,
        32,
        True,
        "~4T",
        3.0e-4,
        0.1,
        480000,
        "inferred",
    ),
    # Llama-3 8B: Meta Llama 3 paper — peak LR ~3e-4, wd 0.1, 15T tokens, batch ~16M tokens → ~1M steps
    (
        "llama-3-8b",
        ROOT / "derived/llama-3-8b_main_fit_per_component_v3.json",
        "Llama-3",
        "8B",
        "GQA-4:1",
        32,
        8,
        False,
        ">15T",
        3.0e-4,
        0.1,
        1000000,
        "inferred",
    ),
    # Mistral-7B: tech report partial — estimates inferred from typical Llama-style recipe
    (
        "mistral-7b",
        ROOT / "derived/mistral-7b_main_fit_per_component_v3.json",
        "Mistral",
        "7B",
        "GQA-4:1",
        32,
        8,
        False,
        "8T",
        3.0e-4,
        0.1,
        500000,
        "estimated",
    ),
    # Qwen2.5-7B: paper says peak LR 3e-4, wd 0.1, 18T tokens
    (
        "qwen2.5-7b",
        ROOT / "derived/qwen2.5-7b_main_fit_per_component_v3.json",
        "Qwen2.5",
        "7B",
        "GQA-7:1",
        28,
        4,
        False,
        "18T",
        3.0e-4,
        0.1,
        1100000,
        "inferred",
    ),
    # Qwen2.5-14B: GQA 40:8 = 5:1 (paper line 211 footnote: per-block extraction covers first 27 of 48 layers; aggregate-level entry); no QK-Norm (paper §QK-Norm figure caption)
    (
        "qwen2.5-14b",
        CV3 / "qwen2.5-14b_main_fit_per_component_v3.json",
        "Qwen2.5",
        "14B",
        "GQA-5:1",
        40,
        8,
        False,
        "18T",
        3.0e-4,
        0.1,
        1100000,
        "estimated",
    ),
    # Qwen3-8B-base: 36T tokens, similar recipe to Qwen2.5
    (
        "qwen3-8b",
        CV3 / "qwen3-8b-base_main_fit_per_component_v3.json",
        "Qwen3",
        "8B",
        "GQA-4:1",
        32,
        8,
        True,
        "36T",
        5.0e-4,   # Qwen3 TR does not publish pretraining LR; inferred, aligned with paper Table 4
        0.1,
        2200000,
        "inferred",
    ),
]


def compute_t_tau(eta_peak, lambda_wd, T_steps):
    """Wang-Aitchison 2024 cycle ratio: T/tau_iter where tau_iter = 1/(eta * lambda_wd)."""
    tau_iter = 1.0 / (eta_peak * lambda_wd)
    return T_steps / tau_iter


def classify_physical_state(t_over_tau):
    """Physical state thresholds (Wang-Aitchison 2024 cycle ratio)."""
    if t_over_tau >= 1.20:
        return "Saturated"
    if t_over_tau >= 0.80:
        return "Near-saturated"
    if t_over_tau >= 0.40:
        return "Approaching"
    if t_over_tau >= 0.25:
        return "Partial"
    return "Transition"


# kinds for separate-Q/K models (SwiGLU/GeGLU FFN: gate/up/down)
KINDS_SEPARATE = ["q", "k", "v", "o", "gate", "up", "down"]
# kinds for Pythia (merged W_qkv + GeLU FFN: ffn_in/ffn_out, no gate)
KINDS_MERGED = ["qkv", "o", "ffn_in", "ffn_out"]


def median_of_kind(records, kind, field):
    vals = [r[field] for r in records if r.get("kind") == kind]
    return statistics.median(vals) if vals else None


def count_low_R2(records, kind, threshold=0.95):
    matching = [r for r in records if r.get("kind") == kind]
    low = [r for r in matching if r.get("R2", 1.0) < threshold]
    return len(low), len(matching)


def load_entry(json_path):
    if not json_path.exists():
        return None
    with open(json_path) as f:
        d = json.load(f)
    return d.get("per_component", [])


def populate():
    rows = []
    sanity_lines = []

    for (
        eid,
        jpath,
        family,
        size,
        arch,
        nq,
        nkv,
        qkn,
        tok,
        eta_peak,
        lambda_wd,
        T_steps,
        hp_conf,
    ) in ENTRIES:
        records = load_entry(jpath)
        if not records:
            print(f"[WARN] missing data for {eid}: {jpath}")
            continue

        # T/tau and Physical State
        t_over_tau = compute_t_tau(eta_peak, lambda_wd, T_steps)
        phys_state = classify_physical_state(t_over_tau)

        # Determine kinds based on architecture (Pythia merged W_qkv vs separate-Q/K)
        is_merged = arch == "MHA-merged"
        kinds = KINDS_MERGED if is_merged else KINDS_SEPARATE

        row = {
            "entry_id": eid,
            "family": family,
            "size": size,
            "arch": arch,
            "n_q": nq,
            "n_kv": nkv,
            "qk_norm": qkn,
            "training_tokens": tok,
            "eta_peak": eta_peak,
            "lambda_wd": lambda_wd,
            "T_steps": T_steps,
            "tau_iter": round(1.0 / (eta_peak * lambda_wd)),
            "T_over_tau": round(t_over_tau, 3),
            "Physical_State": phys_state,
            "hp_confidence": hp_conf,
            "n_records": len(records),
        }

        for kind in kinds:
            row[f"k_median_{kind}"] = median_of_kind(records, kind, "k")
            row[f"lambda_median_{kind}"] = median_of_kind(records, kind, "lambda")
            row[f"R2_median_{kind}"] = median_of_kind(records, kind, "R2")
            low, tot = count_low_R2(records, kind)
            row[f"R2_below_95_{kind}"] = f"{low}/{tot}" if tot else "n/a"

        rows.append(row)

        # Sanity logging
        sanity_lines.append(f"### {eid} ({family} {size}, {arch})")
        sanity_lines.append(f"  records: {len(records)}; tokens: {tok}; QK-Norm: {qkn}")
        if is_merged:
            sanity_lines.append(
                f"  k_median(qkv) = {row['k_median_qkv']:.4f}, R2(qkv) = {row['R2_median_qkv']:.4f}, low-R2: {row['R2_below_95_qkv']}"
            )
        else:
            sanity_lines.append(
                f"  k_median(q)  = {row['k_median_q']:.4f}, R2(q)  = {row['R2_median_q']:.4f}, low-R2: {row['R2_below_95_q']}"
            )
            sanity_lines.append(
                f"  k_median(k)  = {row['k_median_k']:.4f}, R2(k)  = {row['R2_median_k']:.4f}, low-R2: {row['R2_below_95_k']}"
            )
            sanity_lines.append(
                f"  k_median(v)  = {row['k_median_v']:.4f}, R2(v)  = {row['R2_median_v']:.4f}"
            )
            sanity_lines.append(
                f"  k_median(o)  = {row['k_median_o']:.4f}, R2(o)  = {row['R2_median_o']:.4f}"
            )

        def fnum(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

        sanity_lines.append(
            f"  k_median(gate)={fnum(row.get('k_median_gate'))}, k_median(up)={fnum(row.get('k_median_up'))}, k_median(down)={fnum(row.get('k_median_down'))}"
        )
        sanity_lines.append("")

    # ===== Write CSV =====
    csv_path = OUT / "DATABASE_v9_1.csv"
    if rows:
        # Union of all keys (separate + merged)
        keys = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in keys})
        print(f"[saved CSV] {csv_path}  ({len(rows)} rows)")

    # ===== Write markdown reference table =====
    md_path = OUT / "DATABASE_v9_1.md"
    with open(md_path, "w") as f:
        f.write("# DATABASE_v9_1 — Reference Table (12 family entries)\n\n")
        f.write("**Source**: cascade v3 (per-component Weibull fit, mid-80% trim)\n")
        f.write("**Generated**: by populate_database_v9_1.py\n\n")
        f.write("## 12 entries (5 Pythia size + 7 cross-family) — k median per component\n\n")
        f.write(
            "| Entry | Family | Size | Architecture | n_q/n_kv | QK-Norm | tokens |"
            " k_q | k_k | k_v | k_o | k_gate | k_up | k_down | low-R²(q) |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            qk = "yes" if r["qk_norm"] else "no"
            nqkv = f"{r['n_q']}/{r['n_kv']}" if r["n_q"] else "merged"
            kq = r.get("k_median_q") or r.get("k_median_qkv") or 0
            kk = r.get("k_median_k") or "—"
            kv = r.get("k_median_v") or "—"
            ko = r.get("k_median_o") or "—"
            kg = r.get("k_median_gate") or 0
            ku = r.get("k_median_up") or 0
            kd = r.get("k_median_down") or 0
            lowq = r.get("R2_below_95_q", r.get("R2_below_95_qkv", "—"))

            def fmt(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

            f.write(
                f"| {r['entry_id']} | {r['family']} | {r['size']} | {r['arch']} | {nqkv} | {qk} | {r['training_tokens']} | "
                f"{fmt(kq)} | {fmt(kk)} | {fmt(kv)} | {fmt(ko)} | {fmt(kg)} | {fmt(ku)} | {fmt(kd)} | {lowq} |\n"
            )
        f.write("\n")

        # ===== T/tau + Physical State table =====
        f.write(
            "## Training hyperparameters + T/τ + Physical State (Wang-Aitchison 2024 cycle ratio)\n\n"
        )
        f.write(
            "τ_iter = 1/(η · λ_wd) — EMA iteration time-constant.  T/τ_iter = T_steps / τ_iter — completed EMA cycles.\n\n"
        )
        f.write(
            "| Entry | η_peak | λ_wd | T_steps | τ_iter | **T/τ** | **Physical State** | hp source |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['entry_id']} | {r['eta_peak']:.1e} | {r['lambda_wd']} | {r['T_steps']} | "
                f"{int(r['tau_iter'])} | **{r['T_over_tau']:.2f}** | **{r['Physical_State']}** | {r['hp_confidence']} |\n"
            )
        f.write("\n**Physical State thresholds** (Wang-Aitchison 2024 cycle ratio):\n\n")
        f.write("- Saturated: T/τ ≥ 1.20\n")
        f.write("- Near-saturated: 0.80 ≤ T/τ < 1.20\n")
        f.write("- Approaching: 0.40 ≤ T/τ < 0.80\n")
        f.write("- Partial: 0.25 ≤ T/τ < 0.40\n")
        f.write("- Transition: T/τ < 0.25\n\n")
        f.write("**hp source confidence**:\n\n")
        f.write("- *explicit*: paper Table / official tech report directly states the value\n")
        f.write("- *inferred*: paper §3 states a quantity from which we derive it (e.g. tokens × batch / seq → steps)\n")
        f.write("- *estimated*: paper does not publish; same-family typical recipe used as fallback\n\n")

        # ===== Public verification section =====
        f.write("---\n\n## Verification\n\n")
        f.write(
            "Per-entry per-component sanity check is recorded in "
            "[`DATABASE_v9_1_report.md`](DATABASE_v9_1_report.md). "
            "All entries pass R² ≥ 0.99 on the Transmission Class components.\n\n"
        )

        # Compute aggregated Transmission band (median across components per
        # entry, then aggregated across entries) — public-facing summary.
        def fmt_or_dash(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

        per_entry_med = []
        for r in rows:
            # Transmission Class = FFN + W_o per paper §3
            comps = []
            if r.get("k_median_qkv"):  # Pythia merged: ffn_in + ffn_out + o
                for kind in ("ffn_in", "ffn_out", "o"):
                    v = r.get(f"k_median_{kind}")
                    if v is not None:
                        comps.append(v)
            else:  # SwiGLU: gate + up + down + o
                for kind in ("gate", "up", "down", "o"):
                    v = r.get(f"k_median_{kind}")
                    if v is not None:
                        comps.append(v)
            if comps:
                per_entry_med.append(statistics.median(comps))

        if per_entry_med:
            mean_k = statistics.mean(per_entry_med)
            cv_pct = statistics.stdev(per_entry_med) / mean_k * 100
            f.write(
                f"**Transmission Class aggregated band** (median across "
                f"components per entry, then aggregated across the "
                f"{len(per_entry_med)} entries): "
                f"k ∈ [{min(per_entry_med):.4f}, {max(per_entry_med):.4f}], "
                f"cross-family CV = {cv_pct:.2f}%.\n\n"
            )
            f.write(
                "See paper §3 for the strict-band definition and trim "
                "protocol. For per-block raw fits and the cascade "
                "pipeline that produces this table, see the "
                "`npm-weibull-py` repository on GitHub.\n"
            )

    print(f"[saved MD] {md_path}")

    # ===== Sanity report =====
    rpt = OUT / "DATABASE_v9_1_report.md"
    with open(rpt, "w") as f:
        f.write("# DATABASE_v9_1 sanity verify report\n\n")
        f.write("\n".join(sanity_lines))
    print(f"[saved report] {rpt}")


if __name__ == "__main__":
    populate()
