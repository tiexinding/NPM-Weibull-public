"""cascade v3+v2 fit_per_component_v3.json reader.

Loads cascade benchmark data into unified dict for downstream analysis.

Spec: B2_Framework_实施Spec_v2 §5 utility 1 + §6 数据库 reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_cascade_v3(derived_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Auto-parse all fit_per_component_v3.json files in a derived/ dir.

    Parameters
    ----------
    derived_dir : str | Path
        Path to cascade_v3_pull/data/derived/ or cascade_v2_20260502/derived/

    Returns
    -------
    dict {model_step_key: per_component_list} where:
      - model_step_key e.g., "olmo-7b-hf_main", "pythia-70m-step143000_step143000"
      - per_component_list: list of dict {block_idx, kind, k, lambda, R2, k_80, k_90, k_100, lambda_over_mean, mean_abs, std, param_name}
    """
    p = Path(derived_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"derived_dir not found: {derived_dir}")

    out: dict[str, list[dict[str, Any]]] = {}
    for fp in sorted(p.glob("*_fit_per_component_v3.json")):
        key = fp.stem.replace("_fit_per_component_v3", "")
        try:
            with open(fp) as f:
                d = json.load(f)
            pc = d.get("per_component", [])
            if pc:
                out[key] = pc
        except (json.JSONDecodeError, OSError):
            continue
    return out


def filter_per_component(
    per_component: list[dict[str, Any]],
    kind: str | None = None,
    block_range: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Filter per_component list by kind and/or block_idx range.

    Parameters
    ----------
    per_component : list of dict from load_cascade_v3 value
    kind : str | None — 'q', 'k', 'v', 'o', 'qkv', 'gate', 'up', 'down'
    block_range : (lo, hi) | None — inclusive [lo, hi]

    Returns
    -------
    Filtered list
    """
    out = per_component
    if kind is not None:
        out = [r for r in out if r.get("kind") == kind]
    if block_range is not None:
        lo, hi = block_range
        out = [r for r in out if lo <= r.get("block_idx", -1) <= hi]
    return out
