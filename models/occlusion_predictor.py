from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StereoOcclusionPredictor(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
        )
        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.occlusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, stereo_image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(stereo_image)
        pooled = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        ratio = torch.sigmoid(self.ratio_head(pooled)).squeeze(1)
        occlusion_logit = self.occlusion_head(pooled).squeeze(1)
        return {
            "ratio": ratio,
            "occlusion_logit": occlusion_logit,
        }
