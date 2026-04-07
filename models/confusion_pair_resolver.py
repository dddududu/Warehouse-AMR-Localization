from __future__ import annotations

import torch
from torch import nn


class ConfusionPairResolver(nn.Module):
    def __init__(
        self,
        num_pairs: int,
        pair_embedding_dim: int = 128,
        scalar_feature_dim: int = 5,
        hidden_dim: int = 128,
        pair_id_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.pair_embedding = nn.Embedding(num_pairs, pair_id_embedding_dim)
        input_dim = pair_embedding_dim * 2 + scalar_feature_dim * 2 + pair_id_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        pair_ids: torch.Tensor,
        embedding_a: torch.Tensor,
        scalar_a: torch.Tensor,
        embedding_b: torch.Tensor,
        scalar_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pair_features = self.pair_embedding(pair_ids)
        features = torch.cat((embedding_a, scalar_a, embedding_b, scalar_b, pair_features), dim=-1)
        logits = self.mlp(features)
        return {"logits": logits}
