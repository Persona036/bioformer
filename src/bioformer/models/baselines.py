from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

from bioformer.datasets.efp import BatchSequence

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


SUMMARY_STATS = {"mean", "std", "min", "max", "last"}


def _nan_last(values: np.ndarray) -> float:
    finite = values[~np.isnan(values)]
    return float(finite[-1]) if finite.size else 0.0


def _apply_stat(values: np.ndarray, stat_name: str) -> float:
    if stat_name == "mean":
        return float(np.nanmean(values)) if not np.isnan(values).all() else 0.0
    if stat_name == "std":
        return float(np.nanstd(values)) if not np.isnan(values).all() else 0.0
    if stat_name == "min":
        return float(np.nanmin(values)) if not np.isnan(values).all() else 0.0
    if stat_name == "max":
        return float(np.nanmax(values)) if not np.isnan(values).all() else 0.0
    if stat_name == "last":
        return _nan_last(values)
    raise ValueError(f"Unsupported summary statistic: {stat_name}")


def build_summary_matrix(
    sequences: Sequence[BatchSequence],
    *,
    summary_stats: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    invalid = set(summary_stats).difference(SUMMARY_STATS)
    if invalid:
        raise ValueError(f"Unsupported summary statistics: {sorted(invalid)}")

    feature_names = sequences[0].feature_names
    column_names: list[str] = []
    for feature_name in feature_names:
        for stat_name in summary_stats:
            column_names.append(f"{feature_name}__{stat_name}")

    rows: list[np.ndarray] = []
    batch_ids: list[str] = []
    targets: list[float] = []
    for sequence in sequences:
        valid_timesteps = sequence.valid_timesteps
        values = sequence.x_num[valid_timesteps]
        masks = sequence.x_mask[valid_timesteps]

        observed = np.where(masks, values, np.nan)
        row_features: list[float] = []
        for feature_idx in range(observed.shape[1]):
            feature_values = observed[:, feature_idx]
            for stat_name in summary_stats:
                row_features.append(_apply_stat(feature_values, stat_name))

        rows.append(np.asarray(row_features, dtype=np.float32))
        batch_ids.append(sequence.batch_id)
        targets.append(sequence.y_final)

    return (
        np.vstack(rows).astype(np.float32),
        np.asarray(targets, dtype=np.float32),
        column_names,
        batch_ids,
    )


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_name: str,
    ridge_alpha: float,
    elasticnet_alpha: float,
    elasticnet_l1_ratio: float,
    xgboost_params: dict[str, Any] | None = None,
):
    if model_name == "ridge":
        model = Ridge(alpha=ridge_alpha)
    elif model_name == "elasticnet":
        model = ElasticNet(alpha=elasticnet_alpha, l1_ratio=elasticnet_l1_ratio, max_iter=10000)
    elif model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install project dependencies first.")
        model = XGBRegressor(**(xgboost_params or {}))
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    model.fit(X_train, y_train)
    return model

