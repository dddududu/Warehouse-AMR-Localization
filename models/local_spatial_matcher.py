from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocalSpatialMatcher(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 48, max_shift_cells: int = 2) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
        )
        self.max_shift_cells = int(max_shift_cells)

    def project(self, feature_map: torch.Tensor) -> torch.Tensor:
        projected = self.projection(feature_map)
        return F.normalize(projected, dim=1)

    def _score_overlap(self, query_map: torch.Tensor, patch_map: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
        _, _, height, width = query_map.shape
        query_y_start = max(0, shift_y)
        query_y_end = min(height, height + shift_y)
        patch_y_start = max(0, -shift_y)
        patch_y_end = min(height, height - shift_y)
        query_x_start = max(0, shift_x)
        query_x_end = min(width, width + shift_x)
        patch_x_start = max(0, -shift_x)
        patch_x_end = min(width, width - shift_x)

        if query_y_start >= query_y_end or query_x_start >= query_x_end:
            batch_size = query_map.shape[0]
            return torch.full((batch_size,), -1.0e4, device=query_map.device, dtype=query_map.dtype)

        query_crop = query_map[:, :, query_y_start:query_y_end, query_x_start:query_x_end]
        patch_crop = patch_map[:, :, patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        return (query_crop * patch_crop).sum(dim=1).mean(dim=(1, 2))

    def score_pairs(self, query_map: torch.Tensor, patch_map: torch.Tensor) -> torch.Tensor:
        best_score = None
        for shift_y in range(-self.max_shift_cells, self.max_shift_cells + 1):
            for shift_x in range(-self.max_shift_cells, self.max_shift_cells + 1):
                score = self._score_overlap(query_map, patch_map, shift_y=shift_y, shift_x=shift_x)
                if best_score is None:
                    best_score = score
                else:
                    best_score = torch.maximum(best_score, score)
        if best_score is None:
            raise RuntimeError("LocalSpatialMatcher failed to evaluate any shift.")
        return best_score

    def score_pairwise(self, query_map: torch.Tensor, patch_maps: torch.Tensor) -> torch.Tensor:
        if patch_maps.ndim != 5:
            raise ValueError(f"Expected patch_maps with shape (B, N, C, H, W), got {patch_maps.shape}.")
        batch_size, num_patches, channels, height, width = patch_maps.shape
        flat_patches = patch_maps.reshape(batch_size * num_patches, channels, height, width)
        tiled_query = query_map[:, None, ...].expand(batch_size, num_patches, channels, height, width)
        flat_query = tiled_query.reshape(batch_size * num_patches, channels, height, width)
        scores = self.score_pairs(flat_query, flat_patches)
        return scores.reshape(batch_size, num_patches)

    def score_bank(self, query_map: torch.Tensor, patch_bank: torch.Tensor) -> torch.Tensor:
        if query_map.ndim != 4 or query_map.shape[0] != 1:
            raise ValueError(f"Expected query_map with shape (1, C, H, W), got {query_map.shape}.")
        if patch_bank.ndim != 4:
            raise ValueError(f"Expected patch_bank with shape (N, C, H, W), got {patch_bank.shape}.")
        tiled_query = query_map.expand(patch_bank.shape[0], -1, -1, -1)
        return self.score_pairs(tiled_query, patch_bank)
