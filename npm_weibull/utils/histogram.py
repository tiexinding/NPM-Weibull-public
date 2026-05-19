"""Histogram extraction utility — torch.Tensor / np.ndarray → 1024-bin log10|w| NPZ.

cascade v3 standard binning: log10|w| range [-12, 2], 1024 bins.

Spec: B2_Framework_实施Spec_v2 §1 F1 input + §5 utility 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

CASCADE_V3_RANGE = (-12.0, 2.0)
CASCADE_V3_N_BINS = 1024


def extract_to_histogram(
    weight: Any,  # torch.Tensor | np.ndarray (typed Any to avoid hard torch dep)
    n_bins: int = CASCADE_V3_N_BINS,
    log_w_range: tuple[float, float] = CASCADE_V3_RANGE,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute log10|w| histogram from weight matrix using cascade v3 standard binning.

    Parameters
    ----------
    weight : torch.Tensor or np.ndarray
        Weight matrix (any shape). Cast to FP32 for |w| computation.
    n_bins : int, default 1024
        Number of bins
    log_w_range : tuple, default (-12, 2)
        Range of log10|w|
    save_path : str | Path | None
        If not None, save NPZ to this path

    Returns
    -------
    dict with:
        edges : np.ndarray (n_bins+1,) — log10 bin edges
        hist : np.ndarray (n_bins,) — count per bin
        n_total : int — total count (= weight.numel())
        param_name : str — empty (caller can set)
        shape : tuple — original weight shape
    """
    # Cast to numpy FP32 |w|
    if hasattr(weight, "numpy"):
        try:
            arr = weight.detach().to(dtype=__import__("torch").float32).cpu().numpy()
        except Exception:
            arr = np.asarray(weight, dtype=np.float32)
    else:
        arr = np.asarray(weight, dtype=np.float32)

    abs_w = np.abs(arr).reshape(-1)
    # Avoid log(0) — clip tiny values
    abs_w_clip = np.maximum(abs_w, 1e-13)
    log_w = np.log10(abs_w_clip)

    edges = np.linspace(log_w_range[0], log_w_range[1], n_bins + 1)
    hist, _ = np.histogram(log_w, bins=edges)

    result = {
        "edges": edges.astype(np.float64),
        "hist": hist.astype(np.int64),
        "n_total": int(abs_w.size),
        "param_name": "",
        "shape": tuple(arr.shape),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            edges=np.asarray(result["edges"]),
            hist=np.asarray(result["hist"]),
            shape=np.array(result["shape"], dtype=np.int64),
            param_name=str(result["param_name"]),
        )

    return result
