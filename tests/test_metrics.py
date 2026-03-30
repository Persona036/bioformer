import numpy as np

from bioformer.eval.metrics import compute_regression_metrics


def test_compute_regression_metrics_keys() -> None:
    metrics = compute_regression_metrics(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([1.2, 1.8, 3.1], dtype=np.float32),
    )
    assert set(metrics) == {"mae", "rmse", "r2", "pearson", "spearman"}

