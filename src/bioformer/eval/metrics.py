from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    pearson = stats.pearsonr(y_true, y_pred).statistic if len(y_true) > 1 else 0.0
    spearman = stats.spearmanr(y_true, y_pred).statistic if len(y_true) > 1 else 0.0

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(0.0 if np.isnan(pearson) else pearson),
        "spearman": float(0.0 if np.isnan(spearman) else spearman),
    }
    return metrics


def dump_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

