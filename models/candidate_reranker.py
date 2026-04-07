from __future__ import annotations

import torch
from torch import nn


class CandidateReranker(nn.Module):
    def __init__(
        self,
        pair_embedding_dim: int = 128,
        scalar_feature_dim: int = 5,
        hidden_dim: int = 128,
        num_patch_ids: int = 0,
        patch_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.patch_embedding = (
            nn.Embedding(num_patch_ids, patch_embedding_dim)
            if int(num_patch_ids) > 0 and int(patch_embedding_dim) > 0
            else None
        )
        input_dim = pair_embedding_dim + scalar_feature_dim
        if self.patch_embedding is not None:
            input_dim += patch_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pair_embeddings: torch.Tensor,
        scalar_features: torch.Tensor,
        patch_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feature_parts = [pair_embeddings, scalar_features]
        if self.patch_embedding is not None and patch_ids is not None:
            feature_parts.append(self.patch_embedding(patch_ids))
        features = torch.cat(feature_parts, dim=-1)
        logits = self.mlp(features).squeeze(-1)
        return {"logits": logits}
