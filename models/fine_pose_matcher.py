from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FineBEVEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, descriptor_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, descriptor_dim),
            nn.ReLU(inplace=True),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

    def forward(self, bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.stem(bev)
        pooled = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
        descriptor = F.normalize(self.proj(pooled), dim=1)
        return feature_map, descriptor


class FinePoseMatcher(nn.Module):
    def __init__(
        self,
        descriptor_dim: int = 128,
        hidden_dim: int = 128,
        local_submap_size_m: float = 30.0,
    ) -> None:
        super().__init__()
        self.encoder = FineBEVEncoder(in_channels=4, descriptor_dim=descriptor_dim)
        self.local_submap_size_m = float(local_submap_size_m)
        fused_channels = 128 * 4
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        pair_dim = descriptor_dim * 4 + hidden_dim
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.match_head = nn.Linear(hidden_dim, 1)
        self.pose_head = nn.Linear(hidden_dim, 3)
        self.pose_confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, query_bev: torch.Tensor, candidate_bev: torch.Tensor) -> dict[str, torch.Tensor]:
        query_feature_map, query_descriptor = self.encoder(query_bev)
        candidate_feature_map, candidate_descriptor = self.encoder(candidate_bev)
        candidate_feature_map = F.interpolate(
            candidate_feature_map,
            size=query_feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused_map = torch.cat(
            (
                query_feature_map,
                candidate_feature_map,
                torch.abs(query_feature_map - candidate_feature_map),
                query_feature_map * candidate_feature_map,
            ),
            dim=1,
        )
        fused_map = self.spatial_fusion(fused_map)
        fused_local = F.adaptive_avg_pool2d(fused_map, output_size=1).flatten(1)
        fused_global = torch.cat(
            (
                query_descriptor,
                candidate_descriptor,
                torch.abs(query_descriptor - candidate_descriptor),
                query_descriptor * candidate_descriptor,
            ),
            dim=1,
        )
        fused = self.pair_head(torch.cat((fused_global, fused_local), dim=1))
        pose_scale = torch.tensor(
            [self.local_submap_size_m / 2.0, self.local_submap_size_m / 2.0, math.pi],
            device=fused.device,
            dtype=fused.dtype,
        )
        pose = torch.tanh(self.pose_head(fused)) * pose_scale[None, :]
        pose_confidence = torch.sigmoid(self.pose_confidence_head(fused)).squeeze(1)
        match_logit = self.match_head(fused).squeeze(1)
        return {
            "match_logit": match_logit,
            "pose": pose,
            "pose_confidence": pose_confidence,
            "query_descriptor": query_descriptor,
            "candidate_descriptor": candidate_descriptor,
        }
