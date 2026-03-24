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


class QueryImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, descriptor_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        pooled = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        return F.normalize(self.proj(pooled), dim=1)


class FinePoseMatcher(nn.Module):
    def __init__(
        self,
        descriptor_dim: int = 128,
        hidden_dim: int = 128,
        local_submap_size_m: float = 30.0,
        num_xy_bins: int = 31,
        num_yaw_bins: int = 72,
        use_query_image: bool = True,
        use_stereo_query_image: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = FineBEVEncoder(in_channels=4, descriptor_dim=descriptor_dim)
        self.use_query_image = bool(use_query_image)
        self.use_stereo_query_image = bool(use_stereo_query_image and use_query_image)
        self.query_image_encoder = QueryImageEncoder(in_channels=3, descriptor_dim=descriptor_dim) if self.use_query_image else None
        self.local_submap_size_m = float(local_submap_size_m)
        self.num_xy_bins = int(num_xy_bins)
        self.num_yaw_bins = int(num_yaw_bins)
        fused_channels = 128 * 4
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(fused_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        num_global_parts = 4 + (2 if self.use_stereo_query_image else 1 if self.use_query_image else 0)
        pair_dim = descriptor_dim * num_global_parts + hidden_dim
        self.pair_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.match_head = nn.Linear(hidden_dim, 1)
        self.x_bin_head = nn.Linear(hidden_dim, self.num_xy_bins)
        self.y_bin_head = nn.Linear(hidden_dim, self.num_xy_bins)
        self.yaw_bin_head = nn.Linear(hidden_dim, self.num_yaw_bins)
        self.pose_residual_head = nn.Linear(hidden_dim, 3)
        self.pose_confidence_head = nn.Linear(hidden_dim, 1)
        xy_bin_centers = torch.linspace(
            -self.local_submap_size_m / 2.0,
            self.local_submap_size_m / 2.0,
            steps=self.num_xy_bins,
            dtype=torch.float32,
        )
        yaw_bin_centers = torch.linspace(-math.pi, math.pi, steps=self.num_yaw_bins + 1, dtype=torch.float32)[:-1]
        self.register_buffer("xy_bin_centers", xy_bin_centers, persistent=False)
        self.register_buffer("yaw_bin_centers", yaw_bin_centers, persistent=False)
        self.xy_bin_width = float(self.local_submap_size_m / max(1, self.num_xy_bins - 1))
        self.yaw_bin_width = float((2.0 * math.pi) / max(1, self.num_yaw_bins))

    def _decode_pose(
        self,
        x_bin_logits: torch.Tensor,
        y_bin_logits: torch.Tensor,
        yaw_bin_logits: torch.Tensor,
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        x_bin_idx = torch.argmax(x_bin_logits, dim=1)
        y_bin_idx = torch.argmax(y_bin_logits, dim=1)
        yaw_bin_idx = torch.argmax(yaw_bin_logits, dim=1)
        x_center = self.xy_bin_centers[x_bin_idx]
        y_center = self.xy_bin_centers[y_bin_idx]
        yaw_center = self.yaw_bin_centers[yaw_bin_idx]
        residual = torch.tanh(residuals)
        x = x_center + residual[:, 0] * (0.5 * self.xy_bin_width)
        y = y_center + residual[:, 1] * (0.5 * self.xy_bin_width)
        yaw = yaw_center + residual[:, 2] * (0.5 * self.yaw_bin_width)
        return torch.stack((x, y, yaw), dim=1)

    def forward(
        self,
        query_bev: torch.Tensor,
        candidate_bevs: torch.Tensor,
        query_image: torch.Tensor | None = None,
        query_image_right: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query_feature_map, query_descriptor = self.encoder(query_bev)
        query_image_descriptor = None
        query_image_right_descriptor = None
        if self.query_image_encoder is not None:
            if query_image is None:
                raise ValueError("query_image is required when use_query_image=True.")
            query_image_descriptor = self.query_image_encoder(query_image)
            if self.use_stereo_query_image:
                if query_image_right is None:
                    raise ValueError("query_image_right is required when use_stereo_query_image=True.")
                query_image_right_descriptor = self.query_image_encoder(query_image_right)
        if candidate_bevs.ndim == 4:
            candidate_bevs = candidate_bevs[:, None, ...]
        batch_size, num_candidates, channels, height, width = candidate_bevs.shape
        flat_candidates = candidate_bevs.reshape(batch_size * num_candidates, channels, height, width)
        candidate_feature_map, candidate_descriptor = self.encoder(flat_candidates)
        candidate_feature_map = F.interpolate(
            candidate_feature_map,
            size=query_feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        expanded_query_feature_map = query_feature_map[:, None, ...].repeat(1, num_candidates, 1, 1, 1).reshape(
            batch_size * num_candidates,
            *query_feature_map.shape[1:],
        )
        expanded_query_descriptor = query_descriptor[:, None, :].repeat(1, num_candidates, 1).reshape(
            batch_size * num_candidates,
            query_descriptor.shape[1],
        )
        fused_map = torch.cat(
            (
                expanded_query_feature_map,
                candidate_feature_map,
                torch.abs(expanded_query_feature_map - candidate_feature_map),
                expanded_query_feature_map * candidate_feature_map,
            ),
            dim=1,
        )
        fused_map = self.spatial_fusion(fused_map)
        fused_local = F.adaptive_avg_pool2d(fused_map, output_size=1).flatten(1)
        fused_global_parts = [
            expanded_query_descriptor,
            candidate_descriptor,
            torch.abs(expanded_query_descriptor - candidate_descriptor),
            expanded_query_descriptor * candidate_descriptor,
        ]
        if query_image_descriptor is not None:
            expanded_query_image_descriptor = query_image_descriptor[:, None, :].repeat(1, num_candidates, 1).reshape(
                batch_size * num_candidates,
                query_image_descriptor.shape[1],
            )
            fused_global_parts.append(expanded_query_image_descriptor)
            if query_image_right_descriptor is not None:
                expanded_query_image_right_descriptor = query_image_right_descriptor[:, None, :].repeat(1, num_candidates, 1).reshape(
                    batch_size * num_candidates,
                    query_image_right_descriptor.shape[1],
                )
                fused_global_parts.append(expanded_query_image_right_descriptor)
        fused_global = torch.cat(tuple(fused_global_parts), dim=1)
        fused = self.pair_head(torch.cat((fused_global, fused_local), dim=1))
        x_bin_logits = self.x_bin_head(fused)
        y_bin_logits = self.y_bin_head(fused)
        yaw_bin_logits = self.yaw_bin_head(fused)
        pose_residual = self.pose_residual_head(fused)
        pose = self._decode_pose(x_bin_logits, y_bin_logits, yaw_bin_logits, pose_residual)
        pose_confidence = torch.sigmoid(self.pose_confidence_head(fused)).squeeze(1)
        match_logit = self.match_head(fused).squeeze(1)
        return {
            "match_logit": match_logit.reshape(batch_size, num_candidates),
            "pose": pose.reshape(batch_size, num_candidates, 3),
            "x_bin_logits": x_bin_logits.reshape(batch_size, num_candidates, self.num_xy_bins),
            "y_bin_logits": y_bin_logits.reshape(batch_size, num_candidates, self.num_xy_bins),
            "yaw_bin_logits": yaw_bin_logits.reshape(batch_size, num_candidates, self.num_yaw_bins),
            "pose_residual": pose_residual.reshape(batch_size, num_candidates, 3),
            "pose_confidence": pose_confidence.reshape(batch_size, num_candidates),
            "query_descriptor": query_descriptor,
            "query_image_descriptor": query_image_descriptor,
            "query_image_right_descriptor": query_image_right_descriptor,
            "candidate_descriptor": candidate_descriptor.reshape(batch_size, num_candidates, -1),
        }
