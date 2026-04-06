from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(slots=True)
class FeatureScaler:
    means: np.ndarray
    stds: np.ndarray
    feature_cols: list[str]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        safe_stds = np.where(self.stds == 0.0, 1.0, self.stds)
        transformed = transformed.astype({col: np.float32 for col in self.feature_cols}, copy=False)
        feature_values = transformed.loc[:, self.feature_cols].to_numpy(dtype=np.float32)
        transformed.loc[:, self.feature_cols] = (feature_values - self.means) / safe_stds
        return transformed


@dataclass(slots=True)
class BatchSequence:
    batch_id: str
    dataset_id: int
    feature_names: list[str]
    x_num: np.ndarray
    x_mask: np.ndarray
    time_hours: np.ndarray
    y_final: float
    sample_weight: float = 1.0

    @property
    def valid_timesteps(self) -> np.ndarray:
        return self.x_mask.any(axis=1)


class EFPSequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[BatchSequence]) -> None:
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        seq = self.sequences[index]
        valid = seq.valid_timesteps
        return {
            "batch_id": seq.batch_id,
            "dataset_id": torch.tensor(seq.dataset_id, dtype=torch.long),
            "x_num": torch.tensor(seq.x_num, dtype=torch.float32),
            "x_mask": torch.tensor(seq.x_mask, dtype=torch.bool),
            "time_hours": torch.tensor(seq.time_hours, dtype=torch.float32),
            "padding_mask": torch.tensor(~valid, dtype=torch.bool),
            "y_final": torch.tensor(seq.y_final, dtype=torch.float32),
            "sample_weight": torch.tensor(seq.sample_weight, dtype=torch.float32),
        }


def load_efp_frame(
    csv_path: str | Path,
    *,
    batch_id_col: str,
    time_col: str,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place the CSV there or update the config."
        )

    frame = pd.read_csv(path)
    required_cols = {batch_id_col, time_col}
    missing = required_cols.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame[batch_id_col] = frame[batch_id_col].astype(str)
    frame = frame.sort_values([batch_id_col, time_col]).reset_index(drop=True)
    return frame


def infer_numeric_feature_columns(
    frame: pd.DataFrame,
    *,
    batch_id_col: str,
    time_col: str,
    target_col: str,
    feature_cols: Sequence[str] | None = None,
    extra_exclude: Sequence[str] | None = None,
) -> list[str]:
    if feature_cols:
        missing = set(feature_cols).difference(frame.columns)
        if missing:
            raise ValueError(f"Configured feature columns not found: {sorted(missing)}")
        return list(feature_cols)

    exclude = {batch_id_col, time_col, target_col, *(extra_exclude or [])}
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def append_first_differences(
    frame: pd.DataFrame,
    *,
    batch_id_col: str,
    diff_feature_cols: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    if not diff_feature_cols:
        return frame.copy(), []

    missing = set(diff_feature_cols).difference(frame.columns)
    if missing:
        raise ValueError(f"Configured diff feature columns not found: {sorted(missing)}")

    augmented = frame.copy()
    created_cols: list[str] = []
    for col in diff_feature_cols:
        diff_col = f"{col}_diff"
        augmented[diff_col] = augmented.groupby(batch_id_col)[col].diff().fillna(0.0)
        created_cols.append(diff_col)
    return augmented, created_cols


def add_elapsed_time_column(
    frame: pd.DataFrame,
    *,
    batch_id_col: str,
    source_time_col: str,
    derived_time_col: str,
    rebase_time_by_batch: bool,
) -> tuple[pd.DataFrame, str]:
    transformed = frame.copy()
    active_time_col = source_time_col
    if rebase_time_by_batch:
        transformed[derived_time_col] = transformed.groupby(batch_id_col)[source_time_col].transform(
            lambda series: series - series.min()
        )
        active_time_col = derived_time_col
    return transformed, active_time_col


def _build_stratify_labels(
    batch_targets: pd.Series,
    *,
    num_bins: int,
    tail_quantile: float | None,
    max_classes: int,
) -> np.ndarray | None:
    if len(batch_targets) < 3:
        return None

    transformed_targets = np.log1p(batch_targets.to_numpy(dtype=np.float64))
    max_bins = min(max(int(num_bins), 2), len(batch_targets), max(max_classes, 2))
    for current_bins in range(max_bins, 1, -1):
        quantile_labels = pd.qcut(
            transformed_targets,
            q=current_bins,
            labels=False,
            duplicates="drop",
        )
        labels = pd.Series(quantile_labels, index=batch_targets.index, dtype="object")

        if tail_quantile is not None:
            tail_threshold = float(batch_targets.quantile(float(tail_quantile)))
            tail_mask = batch_targets >= tail_threshold
            if int(tail_mask.sum()) >= 2:
                labels.loc[tail_mask] = f"tail_{float(tail_quantile):.2f}"

        counts = labels.value_counts(dropna=False)
        if counts.empty or len(counts) > max_classes or int(counts.min()) < 2:
            continue
        return labels.astype(str).to_numpy()
    return None


def _resolve_split_size(num_items: int, split_size: float) -> int:
    return max(int(math.ceil(num_items * split_size)), 1)


def split_batch_ids(
    frame: pd.DataFrame,
    *,
    batch_id_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    target_col: str | None = None,
    stratify: bool = False,
    stratify_bins: int = 10,
    stratify_tail_quantile: float | None = 0.98,
) -> dict[str, set[str]]:
    batch_ids = np.array(sorted(frame[batch_id_col].astype(str).unique()))
    if len(batch_ids) < 3:
        raise ValueError("At least 3 unique batches are required to create train/val/test splits.")

    stratify_labels: np.ndarray | None = None
    label_lookup: dict[str, str] | None = None
    if stratify:
        if target_col is None:
            raise ValueError("target_col is required when stratify=True.")
        test_count = _resolve_split_size(len(batch_ids), test_size)
        train_val_count = len(batch_ids) - test_count
        relative_val_size = val_size / (1.0 - test_size)
        val_count = _resolve_split_size(train_val_count, relative_val_size)
        batch_targets = frame.groupby(batch_id_col, sort=False)[target_col].last().reindex(batch_ids)
        stratify_labels = _build_stratify_labels(
            batch_targets,
            num_bins=stratify_bins,
            tail_quantile=stratify_tail_quantile,
            max_classes=min(test_count, val_count),
        )
        if stratify_labels is not None:
            label_lookup = dict(zip(batch_ids.tolist(), stratify_labels.tolist(), strict=True))

    relative_val_size = val_size / (1.0 - test_size)
    try:
        train_val_ids, test_ids = train_test_split(
            batch_ids,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
        train_val_labels = None
        if label_lookup is not None:
            train_val_labels = np.asarray([label_lookup[batch_id] for batch_id in train_val_ids], dtype=object)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=relative_val_size,
            random_state=seed,
            shuffle=True,
            stratify=train_val_labels,
        )
    except ValueError as exc:
        if stratify_labels is None:
            raise
        warnings.warn(
            f"Falling back to unstratified batch split because stratification failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        train_val_ids, test_ids = train_test_split(
            batch_ids,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=relative_val_size,
            random_state=seed,
            shuffle=True,
        )
    return {
        "train": set(train_ids.tolist()),
        "val": set(val_ids.tolist()),
        "test": set(test_ids.tolist()),
    }


def fit_feature_scaler(frame: pd.DataFrame, feature_cols: Sequence[str]) -> FeatureScaler:
    means = frame.loc[:, feature_cols].mean().to_numpy(dtype=np.float32)
    stds = frame.loc[:, feature_cols].std(ddof=0).fillna(1.0).to_numpy(dtype=np.float32)
    return FeatureScaler(means=means, stds=stds, feature_cols=list(feature_cols))


def filter_frame_by_batches(
    frame: pd.DataFrame,
    *,
    batch_id_col: str,
    batch_ids: Iterable[str],
) -> pd.DataFrame:
    selected = set(batch_ids)
    return frame[frame[batch_id_col].astype(str).isin(selected)].copy()


def write_split_frames(
    frame: pd.DataFrame,
    *,
    batch_id_col: str,
    split_ids: dict[str, set[str]],
    output_dir: str | Path,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, batch_ids in split_ids.items():
        split_frame = filter_frame_by_batches(frame, batch_id_col=batch_id_col, batch_ids=batch_ids)
        split_frame.to_parquet(out_dir / f"{split_name}.parquet", index=False)


def build_sequences(
    frame: pd.DataFrame,
    *,
    batch_id_col: str,
    time_col: str,
    target_col: str,
    feature_cols: Sequence[str],
    horizon_hours: float,
    max_seq_len: int,
    dataset_id: int = 0,
) -> list[BatchSequence]:
    sequences: list[BatchSequence] = []

    grouped = frame.groupby(batch_id_col, sort=False)
    for batch_id, batch_frame in grouped:
        ordered = batch_frame.sort_values(time_col).reset_index(drop=True)
        target_series = ordered[target_col].dropna()
        if target_series.empty:
            continue

        history = ordered[ordered[time_col] <= horizon_hours].copy()
        if history.empty:
            continue

        history = history.iloc[:max_seq_len].copy()
        feature_values = history.loc[:, feature_cols]
        x_num = feature_values.fillna(0.0).to_numpy(dtype=np.float32)
        x_mask = (~feature_values.isna()).to_numpy(dtype=np.bool_)
        time_hours = history[time_col].to_numpy(dtype=np.float32)

        padded_x = np.zeros((max_seq_len, len(feature_cols)), dtype=np.float32)
        padded_mask = np.zeros((max_seq_len, len(feature_cols)), dtype=np.bool_)
        padded_time = np.zeros((max_seq_len,), dtype=np.float32)

        seq_len = len(history)
        padded_x[:seq_len] = x_num
        padded_mask[:seq_len] = x_mask
        padded_time[:seq_len] = time_hours

        sequences.append(
            BatchSequence(
                batch_id=str(batch_id),
                dataset_id=dataset_id,
                feature_names=list(feature_cols),
                x_num=padded_x,
                x_mask=padded_mask,
                time_hours=padded_time,
                y_final=float(target_series.iloc[-1]),
            )
        )

    if not sequences:
        raise ValueError("No batch sequences were created. Check the horizon and target column.")
    return sequences
