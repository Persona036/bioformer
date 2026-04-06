from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

from bioformer.datasets.efp import (
    add_elapsed_time_column,
    append_first_differences,
    build_sequences,
    filter_frame_by_batches,
    fit_feature_scaler,
    infer_numeric_feature_columns,
    load_efp_frame,
    split_batch_ids,
    write_split_frames,
)
from bioformer.eval.metrics import compute_regression_metrics, dump_metrics
from bioformer.eval.plots import save_predicted_vs_true_plot
from bioformer.models.baselines import build_summary_matrix, train_baseline_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models for the EFP task.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    model_cfg = config["model"]
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_efp_frame(
        data_cfg["raw_csv"],
        batch_id_col=data_cfg["batch_id_col"],
        time_col=data_cfg["time_col"],
    )
    frame, diff_cols = append_first_differences(
        frame,
        batch_id_col=data_cfg["batch_id_col"],
        diff_feature_cols=data_cfg.get("diff_feature_cols", []),
    )
    frame, model_time_col = add_elapsed_time_column(
        frame,
        batch_id_col=data_cfg["batch_id_col"],
        source_time_col=data_cfg["time_col"],
        derived_time_col=data_cfg.get("derived_time_col", "elapsed_hours"),
        rebase_time_by_batch=bool(data_cfg.get("rebase_time_by_batch", False)),
    )
    configured_feature_cols = list(data_cfg.get("feature_cols", []))
    feature_cols = infer_numeric_feature_columns(
        frame,
        batch_id_col=data_cfg["batch_id_col"],
        time_col=data_cfg["time_col"],
        target_col=data_cfg["target_col"],
        feature_cols=configured_feature_cols or None,
        extra_exclude=[model_time_col] if model_time_col != data_cfg["time_col"] else None,
    )
    if diff_cols:
        for diff_col in diff_cols:
            if diff_col not in feature_cols:
                feature_cols.append(diff_col)

    split_ids = split_batch_ids(
        frame,
        batch_id_col=data_cfg["batch_id_col"],
        test_size=float(data_cfg["test_size"]),
        val_size=float(data_cfg["val_size"]),
        seed=int(config["seed"]),
        target_col=data_cfg["target_col"],
        stratify=bool(data_cfg.get("stratify_splits", True)),
        stratify_bins=int(data_cfg.get("stratify_bins", 10)),
        stratify_tail_quantile=data_cfg.get("stratify_tail_quantile", 0.98),
    )
    write_split_frames(
        frame,
        batch_id_col=data_cfg["batch_id_col"],
        split_ids=split_ids,
        output_dir=data_cfg["processed_dir"],
    )

    train_frame = filter_frame_by_batches(
        frame,
        batch_id_col=data_cfg["batch_id_col"],
        batch_ids=split_ids["train"],
    )
    scaler = fit_feature_scaler(train_frame, feature_cols)
    normalized = scaler.transform(frame)

    sequences = {}
    for split_name, batch_ids in split_ids.items():
        split_frame = filter_frame_by_batches(
            normalized,
            batch_id_col=data_cfg["batch_id_col"],
            batch_ids=batch_ids,
        )
        sequences[split_name] = build_sequences(
            split_frame,
            batch_id_col=data_cfg["batch_id_col"],
            time_col=model_time_col,
            target_col=data_cfg["target_col"],
            feature_cols=feature_cols,
            horizon_hours=float(data_cfg["horizon_hours"]),
            max_seq_len=int(data_cfg["max_seq_len"]),
        )

    X_train, y_train, feature_names, _ = build_summary_matrix(
        sequences["train"],
        summary_stats=model_cfg["summary_stats"],
    )
    X_val, y_val, _, _ = build_summary_matrix(
        sequences["val"],
        summary_stats=model_cfg["summary_stats"],
    )
    X_test, y_test, _, batch_ids = build_summary_matrix(
        sequences["test"],
        summary_stats=model_cfg["summary_stats"],
    )

    model = train_baseline_model(
        X_train,
        y_train,
        model_name=model_cfg["name"],
        ridge_alpha=float(model_cfg["ridge_alpha"]),
        elasticnet_alpha=float(model_cfg["elasticnet_alpha"]),
        elasticnet_l1_ratio=float(model_cfg["elasticnet_l1_ratio"]),
        xgboost_params=model_cfg.get("xgboost_params", {}),
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    metrics = {
        "val": compute_regression_metrics(y_val, val_pred),
        "test": compute_regression_metrics(y_test, test_pred),
        "model_name": model_cfg["name"],
        "horizon_hours": float(data_cfg["horizon_hours"]),
        "num_features": len(feature_names),
    }

    prediction_rows = [
        {"batch_id": batch_id, "y_true": float(y_true), "y_pred": float(y_pred)}
        for batch_id, y_true, y_pred in zip(batch_ids, y_test, test_pred, strict=True)
    ]
    dump_metrics(metrics["val"], output_dir / "val_metrics.json")
    dump_metrics(metrics["test"], output_dir / "test_metrics.json")
    (output_dir / "predictions_test.json").write_text(
        json.dumps(prediction_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        save_predicted_vs_true_plot(
            y_test,
            test_pred,
            output_dir / "predicted_vs_true_test.png",
            title=f"{model_cfg['name']} holdout predictions",
        )
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"plot_warning": str(exc)}))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
