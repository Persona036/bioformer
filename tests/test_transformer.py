import torch

from bioformer.models.transformer import TimeSeriesTransformer


def test_transformer_forward_shape() -> None:
    model = TimeSeriesTransformer(
        input_dim=5,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ff_dim=64,
        dropout=0.1,
    )
    batch_size = 3
    seq_len = 8
    x_num = torch.randn(batch_size, seq_len, 5)
    x_mask = torch.ones(batch_size, seq_len, 5, dtype=torch.bool)
    time_hours = torch.arange(seq_len).repeat(batch_size, 1).float()
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    pred = model(x_num, x_mask, time_hours, padding_mask)
    assert pred.shape == (batch_size,)
