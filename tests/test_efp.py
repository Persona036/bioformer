import pandas as pd

from bioformer.datasets.efp import split_batch_ids


def test_split_batch_ids_stratifies_tail_batches() -> None:
    rows: list[dict[str, float | str]] = []
    final_targets = {
        **{str(idx): float(idx) for idx in range(1, 17)},
        "17": 100.0,
        "18": 110.0,
        "19": 120.0,
        "20": 130.0,
    }
    for batch_id, final_target in final_targets.items():
        rows.append({"batch_id": batch_id, "hh": 0.0, "cer": final_target / 2.0})
        rows.append({"batch_id": batch_id, "hh": 1.0, "cer": final_target})

    frame = pd.DataFrame(rows).sort_values(["batch_id", "hh"]).reset_index(drop=True)
    split_ids = split_batch_ids(
        frame,
        batch_id_col="batch_id",
        test_size=0.2,
        val_size=0.2,
        seed=42,
        target_col="cer",
        stratify=True,
        stratify_bins=5,
        stratify_tail_quantile=0.8,
    )

    batch_targets = frame.groupby("batch_id", sort=False)["cer"].last()
    tail_ids = set(batch_targets[batch_targets >= batch_targets.quantile(0.8)].index.astype(str))

    assert tail_ids.intersection(split_ids["train"])
    assert tail_ids.intersection(split_ids["val"])
    assert tail_ids.intersection(split_ids["test"])
    assert split_ids["train"].isdisjoint(split_ids["val"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["val"].isdisjoint(split_ids["test"])
