import torch

from bioformer.train.train_transformer import (
    build_regression_loss,
    build_target_transform,
    compute_tail_sample_weights,
)


def test_log1p_target_transform_round_trip() -> None:
    transform = build_target_transform("log1p")
    values = torch.tensor([0.0, 1.0, 12.5, 336.0], dtype=torch.float32)

    restored = transform.inverse(transform.forward(values))

    assert torch.allclose(restored, values)


def test_build_regression_loss_huber() -> None:
    loss_fn = build_regression_loss("huber", huber_delta=0.5)

    assert isinstance(loss_fn, torch.nn.HuberLoss)


def test_compute_tail_sample_weights_upweights_tail_targets() -> None:
    targets = torch.tensor([1.0, 10.0, 25.0, 120.0], dtype=torch.float32).numpy()

    weights = compute_tail_sample_weights(
        targets,
        mid_threshold=20.0,
        mid_weight=3.0,
        high_threshold=100.0,
        high_weight=12.0,
    )

    assert weights.tolist() == [1.0, 1.0, 3.0, 12.0]
