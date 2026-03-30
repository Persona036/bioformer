from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bioformer.datasets.efp import (
    EFPSequenceDataset,
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
from bioformer.models.transformer import TimeSeriesTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the transformer EFP model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloader(sequences, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = EFPSequenceDataset(sequences)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_norm: float,
    use_amp: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    total_loss = 0.0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    iterator = tqdm(loader, leave=False, disable=len(loader) < 2)
    for batch in iterator:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(
                    batch["x_num"],
                    batch["x_mask"],
                    batch["time_hours"],
                    batch["padding_mask"],
                    batch["dataset_id"],
                )
                loss = loss_fn(pred, batch["y_final"])

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        total_loss += float(loss.detach().cpu().item()) * batch["y_final"].shape[0]
        preds.append(pred.detach().cpu().numpy())
        targets.append(batch["y_final"].detach().cpu().numpy())

    dataset_size = len(loader.dataset)
    return (
        total_loss / max(dataset_size, 1),
        np.concatenate(preds).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
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

    batch_size = int(train_cfg["batch_size"])
    train_loader = build_dataloader(sequences["train"], batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(sequences["val"], batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(sequences["test"], batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(train_cfg["mixed_precision"]) and device.type == "cuda"
    model = TimeSeriesTransformer(
        input_dim=len(feature_cols),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        num_datasets=int(model_cfg.get("num_datasets", 1)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_mae = float("inf")
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss, train_pred, train_true = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=float(train_cfg["grad_clip_norm"]),
            use_amp=use_amp,
        )
        val_loss, val_pred, val_true = run_epoch(
            model,
            val_loader,
            optimizer=None,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=float(train_cfg["grad_clip_norm"]),
            use_amp=use_amp,
        )

        train_metrics = compute_regression_metrics(train_true, train_pred)
        val_metrics = compute_regression_metrics(val_true, val_pred)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_mae": float(train_metrics["mae"]),
            "val_mae": float(val_metrics["mae"]),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(train_cfg["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError("Training never produced a checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_pred, test_true = run_epoch(
        model,
        test_loader,
        optimizer=None,
        loss_fn=loss_fn,
        device=device,
        grad_clip_norm=float(train_cfg["grad_clip_norm"]),
        use_amp=use_amp,
    )
    test_metrics = compute_regression_metrics(test_true, test_pred)

    torch.save(best_state, output_dir / "best_model.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    dump_metrics(test_metrics, output_dir / "test_metrics.json")
    try:
        save_predicted_vs_true_plot(
            test_true,
            test_pred,
            output_dir / "predicted_vs_true_test.png",
            title="Transformer holdout predictions",
        )
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"plot_warning": str(exc)}))

    summary = {
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "horizon_hours": float(data_cfg["horizon_hours"]),
        "num_features": len(feature_cols),
        "device": device.type,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
