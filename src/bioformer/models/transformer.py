from __future__ import annotations

import torch
from torch import nn


def masked_mean_pool(embeddings: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    weights = keep_mask.unsqueeze(-1).to(embeddings.dtype)
    summed = (embeddings * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float,
        num_datasets: int = 1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim * 2, d_model)
        self.time_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.dataset_embedding = nn.Embedding(num_datasets, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_mask: torch.Tensor,
        time_hours: torch.Tensor,
        padding_mask: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dataset_id is None:
            dataset_id = torch.zeros(x_num.shape[0], dtype=torch.long, device=x_num.device)

        missingness = x_mask.to(x_num.dtype)
        features = torch.cat([x_num, missingness], dim=-1)
        encoded = self.input_projection(features)
        encoded = encoded + self.time_projection(time_hours.unsqueeze(-1))
        encoded = encoded + self.dataset_embedding(dataset_id).unsqueeze(1)
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        pooled = masked_mean_pool(encoded, ~padding_mask)
        pooled = self.output_norm(pooled)
        return self.head(pooled).squeeze(-1)
