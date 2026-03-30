from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np


def _configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "bioformer-matplotlib-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def save_predicted_vs_true_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.8, edgecolor="none")
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("True Final Potency")
    ax.set_ylabel("Predicted Final Potency")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_horizon_error_plot(
    horizons: Sequence[int],
    errors: Sequence[float],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(horizons, errors, marker="o")
    ax.set_title(title)
    ax.set_xlabel("History Horizon (hours)")
    ax.set_ylabel("Error")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
